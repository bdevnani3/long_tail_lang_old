import copy
import enum

# from msilib.schema import SelfReg
import os
import pdb
import pickle
import random
from tabnanny import verbose
import time
import warnings
from re import X, template
from readline import set_pre_input_hook

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import clip
from clip import clip
from matplotlib.pyplot import phase_spectrum
from numpy.core.fromnumeric import cumprod
from pytz import NonExistentTimeError
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from algorithms import *
from optimizers import *
from classes import CLASSES, CUSTOM_TEMPLATES, GENERIC_PROMPT_COLLECTIONS
from diffgrad import diffgrad
from logger import Logger
from utils import *


def load_clip_to_cpu_pretrained(visual_backbone):
    backbone_name = visual_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url, os.path.expanduser("~/.cache/clip"))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def load_clip_to_cpu(visual_backbone, checkpoint=None):
    """
    Remember to provide checkpoint of same backbone as visual_backbone.
    """
    model = load_clip_to_cpu_pretrained(visual_backbone)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location="cuda")
        state_dict = checkpoint["state_dict"]
        state_dict = {key[len("module.") :]: value for key, value in state_dict.items()}
        model.load_state_dict(state_dict)
    return model


class model:
    def __init__(self, config, data, test=False):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.config = config
        self.training_opt = self.config["training_opt"]
        self.model_opt = self.config["model"]
        self.data = data
        self.test_mode = test
        self.num_gpus = torch.cuda.device_count()
        self.do_shuffle = config["shuffle"] if "shuffle" in config else False

        checkpoint = None
        if check_config(self.model_opt["clip"]["params"], "checkpoint"):
            checkpoint = self.model_opt["clip"]["params"]["checkpoint"]
            print(
                "----------> Loading CLIP model from {} <----------".format(checkpoint)
            )
        self.clip_model = load_clip_to_cpu(
            self.model_opt["clip"]["params"]["visual_backbone"], checkpoint
        )

        self.writer = SummaryWriter(log_dir="./runs/" + self.training_opt["tf_folder"])

        # Setup logger
        self.logger = Logger(self.training_opt["log_dir"])

        self.optimizer_variant = (
            config["optimizer_variant"] if "optimizer_variant" in config else None
        )

        self.variation = self.training_opt["variation"]

        if check_config(self.training_opt, "proj_path"):
            self.proj = (
                torch.tensor(np.load(self.training_opt["proj_path"])).cuda().float()
            )
            print(
                "----------> Loading projection matrix from {} <----------".format(
                    self.training_opt["proj_path"]
                )
            )
        else:
            self.proj = None

        # Initialize model
        self.init_models()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            print("Using steps for training.")
            self.training_data_num = len(self.data["train"].dataset)
            self.epoch_steps = int(
                self.training_data_num / self.training_opt["batch_size"]
            )

            # Initialize model optimizer and scheduler
            print("Initializing model optimizer.")
            self.scheduler_params = self.training_opt["scheduler_params"]
            self.model_optimizer, self.model_optimizer_scheduler = init_optimizers(
                self, self.model_optim_params_list
            )
            init_criterions(self)

            # Set up log file
            self.log_file = os.path.join(self.training_opt["log_dir"], "log.txt")
            if os.path.isfile(self.log_file):
                os.remove(self.log_file)
            self.logger.log_cfg(self.config)
        else:
            self.log_file = None

        self.plateau_lr_metric = 0.0

    def init_models(self, optimizer=True):
        # import pdb; pdb.set_trace()

        self.model_optim_params_list = []
        self.model_optim_params_list_LBFGS = []

        self.relu = nn.ReLU()

        print("Using", torch.cuda.device_count(), "GPUs.")

        # Initializing CLIP visual and Text models
        self.visual_model = torch.nn.DataParallel(self.clip_model.visual).cuda()
        text_model = TextEncoder(self.clip_model)
        self.text_model = torch.nn.DataParallel(text_model).cuda()

        in_dim = self.model_opt["fusion"]["params"]["in_dim"]
        out_dim = self.model_opt["fusion"]["params"]["out_dim"]
        self.fusion = torch.nn.DataParallel(
            nn.Linear(in_dim, out_dim, bias=False)
        ).cuda()

        if "proj" in self.variation:
            in_dim = self.model_opt["im_proj"]["params"]["in_dim"]
            out_dim = self.model_opt["im_proj"]["params"]["out_dim"]
            self.im_proj = torch.nn.DataParallel(
                nn.Linear(in_dim, out_dim, bias=False)
            ).cuda()

        if "model_dir" in self.config:
            print("Loading model weights from ", self.config["model_dir"])
            self.load_model(self.config["model_dir"])

        self.load_model()

        if self.training_opt["image_encoder_frozen"] is True:
            for param_name, param in self.visual_model.named_parameters():
                param.requires_grad = False

        if self.training_opt["text_encoder_frozen"] is True:
            for param_name, param in self.text_model.named_parameters():
                param.requires_grad = False

        optim_params_clip = self.model_opt["clip"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.visual_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.visual_model.parameters())

        self.model_optim_params_list.append(
            {
                "params": self.text_model.parameters(),
                "lr": optim_params_clip["lr"],
                "momentum": optim_params_clip["momentum"],
                "weight_decay": optim_params_clip["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.text_model.parameters())

        optim_params_fusion = self.model_opt["fusion"]["optim_params"]
        self.model_optim_params_list.append(
            {
                "params": self.fusion.parameters(),
                "lr": optim_params_fusion["lr"],
                "momentum": optim_params_fusion["momentum"],
                "weight_decay": optim_params_fusion["weight_decay"],
            }
        )
        self.model_optim_params_list_LBFGS.extend(self.fusion.parameters())

        if "proj" in self.variation:
            optim_params_im_proj = self.model_opt["im_proj"]["optim_params"]
            self.model_optim_params_list.append(
                {
                    "params": self.im_proj.parameters(),
                    "lr": optim_params_im_proj["lr"],
                    "momentum": optim_params_im_proj["momentum"],
                    "weight_decay": optim_params_im_proj["weight_decay"],
                }
            )
            self.model_optim_params_list_LBFGS.extend(self.im_proj.parameters())

    def batch_forward(self, inputs, labels, phase="train", paths=None):
        """
        This is a general single batch running function.
        """

        variation = self.training_opt["variation"]

        if variation == "clip":
            self.logits = regular_clip(self, inputs, labels, phase=phase)
        elif variation == "clip_with_linear_layer":
            self.logits = clip_with_linear_layer(self, inputs, labels, phase=phase)
        elif variation == "add_embs_then_linear1":
            self.logits = add_embs_then_linear1(self, inputs, labels, phase=phase)
        elif variation == "image_linear_probe":
            self.logits = image_linear_probe(self, inputs, labels, phase=phase)
        elif variation == "image_plus_text":
            self.logits = image_plus_text(self, inputs, labels, phase=phase)
        elif variation == "image_linear_probe_boosted":
            self.logits = image_linear_probe_boosted(
                self, inputs, labels, phase=phase, paths=paths, proj=self.proj
            )
        elif variation == "image_concat_text":
            self.logits = image_concat_text(self, inputs, labels, phase=phase)
        elif variation == "image_plus_text_all_prompts":
            self.logits = image_plus_text_all_prompts(
                self, inputs, labels, phase=phase, paths=paths
            )
        elif variation == "image_concat_text_all_prompts":
            self.logits = image_concat_text_all_prompts(
                self, inputs, labels, phase=phase, paths=paths
            )
        elif variation == "image_linear_probe_proj":
            self.logits = image_linear_probe_proj(self, inputs, labels, phase=phase)
        elif variation == "image_plus_text_with_tranformation_all_prompts":
            self.logits = image_plus_text_with_tranformation_all_prompts(
                self, inputs, labels, phase=phase, paths=paths, proj=self.proj
            )
        elif variation == "text_with_transformation":
            self.logits = text_with_transformation(
                self, inputs, labels, phase=phase, paths=paths, proj=self.proj
            )

    def batch_backward(self):
        # Zero out optimizer gradients
        self.model_optimizer.zero_grad()
        if self.criterion_optimizer:
            self.criterion_optimizer.zero_grad()
        # Back-propagation from loss outputs
        self.loss.backward()
        # Step optimizers

        self.model_optimizer.step()
        if self.criterion_optimizer:
            self.criterion_optimizer.step()

    def batch_loss(self, labels):
        self.loss = 0

        # First, apply performance loss
        if "PerformanceLoss" in self.criterions.keys():
            self.loss_perf = self.criterions["PerformanceLoss"](self.logits, labels)
            self.loss_perf *= self.criterion_weights["PerformanceLoss"]
            self.loss += self.loss_perf

        # Apply loss on features if set up
        if "FeatureLoss" in self.criterions.keys():
            self.loss_feat = self.criterions["FeatureLoss"](self.features, labels)
            self.loss_feat = self.loss_feat * self.criterion_weights["FeatureLoss"]
            # Add feature loss to total loss
            self.loss += self.loss_feat

    def shuffle_batch(self, x, y):
        index = torch.randperm(x.size(0))
        x = x[index]
        y = y[index]
        return x, y

    def train_step(self, step, epoch, inputs, labels, indexes, t):

        # Break when step equal to epoch step
        if step == self.epoch_steps:
            return
        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        with torch.set_grad_enabled(True):

            if self.training_opt["variation"] == "image_text_all_operations":
                self.image_plus_text(inputs, labels, phase="train", lam=0)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_plus_text(inputs, labels, phase="train", lam=1)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_plus_text(inputs, labels, phase="train", lam=0.5)
                self.batch_loss(labels)
                self.batch_backward()

                self.image_dot_text(inputs, labels, phase="train")
                self.batch_loss(labels)
                self.batch_backward()

            else:
                # If training, forward with loss, and no top 5 accuracy calculation
                self.batch_forward(inputs, labels, phase="train", paths=t)
                self.batch_loss(labels)
                self.batch_backward()

            # Tracking predictions
            _, preds = torch.max(self.logits, 1)
            self.total_preds.append(torch2numpy(preds))
            self.total_labels.append(torch2numpy(labels))
            if "boosted" in self.variation:
                self.total_paths.append(torch2numpy(t))

            # TODO uncomment
            self.minibatch_training_results(step, epoch, preds, labels)

    def train_step_LBFGS(self, step, epoch, inputs, labels, indexes, t):

        # Break when step equal to epoch step
        if step == self.epoch_steps:
            return
        if self.do_shuffle:
            inputs, labels = self.shuffle_batch(inputs, labels)
        inputs, labels = inputs.cuda(), labels.cuda()

        # If on training phase, enable gradients

        def closure():
            self.model_optimizer.zero_grad()
            if self.criterion_optimizer:
                self.criterion_optimizer.zero_grad()

            self.batch_forward(inputs, labels, phase="train", paths=t)
            self.batch_loss(labels)
            # Back-propagation from loss outputs
            if self.loss.requires_grad:
                self.loss.backward()
            return self.loss

        self.model_optimizer.step(closure)
        if self.criterion_optimizer:
            self.criterion_optimizer.step(closure)

        # Tracking predictions
        _, preds = torch.max(self.logits, 1)
        self.total_preds.append(torch2numpy(preds))
        self.total_labels.append(torch2numpy(labels))
        if "boosted" in self.variation:
            self.total_paths.append(torch2numpy(t))

        # TODO uncomment
        self.minibatch_training_results(step, epoch, preds, labels)

    def minibatch_training_results(self, step, epoch, preds, labels):

        # Output minibatch training results
        if step % self.training_opt["display_step"] == 0:

            minibatch_loss_feat = (
                self.loss_feat.item()
                if "FeatureLoss" in self.criterions.keys()
                else None
            )
            minibatch_loss_perf = (
                self.loss_perf.item() if "PerformanceLoss" in self.criterions else None
            )
            minibatch_loss_total = self.loss.item()
            minibatch_acc = mic_acc_cal(preds, labels)

            print_str = [
                "Config {}".format(self.training_opt["tf_folder"]),
                "Epoch: [%d/%d]" % (epoch, self.training_opt["num_epochs"]),
                "Step: %5d" % (step),
                "Minibatch_loss_feature: %.3f" % (minibatch_loss_feat)
                if minibatch_loss_feat
                else "",
                "Minibatch_loss_performance: %.3f" % (minibatch_loss_perf)
                if minibatch_loss_perf
                else "",
                "Minibatch_accuracy_micro: %.3f" % (minibatch_acc),
            ]
            print_write(print_str, self.log_file)

            loss_info = {
                "Epoch": epoch,
                "Step": step,
                "Total": minibatch_loss_total,
                "CE": minibatch_loss_perf,
                "feat": minibatch_loss_feat,
            }

            self.logger.log_loss(loss_info)
            self.epoch_loss += minibatch_loss_total

    def train_epoch(self, epoch):

        torch.cuda.empty_cache()

        # Set model modes and set scheduler
        # In training, step optimizer scheduler and set model to train()
        if check_config(self.config, "plateaulr"):
            self.model_optimizer_scheduler.step(metrics=self.plateau_lr_metric)
        else:
            self.model_optimizer_scheduler.step()

        if self.criterion_optimizer:
            self.criterion_optimizer_scheduler.step()

        # Iterate over dataset
        self.total_preds = []
        self.total_labels = []
        self.total_paths = []

        self.epoch_loss = 0.0

        def call_train_step(step, epoch, inputs, labels, indexes, t):
            if self.optimizer_variant == "LBFGS":
                self.train_step_LBFGS(step, epoch, inputs, labels, indexes, t)
            else:
                self.train_step(step, epoch, inputs, labels, indexes, t)

        for step, (inputs, labels, indexes, t) in enumerate(self.data["train"]):

            if self.training_opt["variation"] == "image_text_all_operations":
                self.training_opt["variation"] = "image_plus_text"
                self.training_opt["image_emb_weight"] = 0
                call_train_step(step, epoch, inputs, labels, indexes, t)
                self.training_opt["image_emb_weight"] = 1
                call_train_step(step, epoch, inputs, labels, indexes, t)
                self.training_opt["image_emb_weight"] = 0.5
                call_train_step(step, epoch, inputs, labels, indexes, t)

            else:
                call_train_step(step, epoch, inputs, labels, indexes, t)

            # Update priority weights if using PrioritizedSampler
            # if self.training_opt['sampler'] and \
            #    self.training_opt['sampler']['type'] == 'PrioritizedSampler':
            if hasattr(self.data["train"].sampler, "update_weights"):
                if hasattr(self.data["train"].sampler, "ptype"):
                    ptype = self.data["train"].sampler.ptype
                else:
                    ptype = "score"
                ws = get_priority(ptype, self.logits.detach(), labels)
                # ws = logits2score(self.logits.detach(), labels)
                inlist = [indexes.cpu().numpy(), ws]
                if self.training_opt["sampler"]["type"] == "ClassPrioritySampler":
                    inlist.append(labels.cpu().numpy())
                self.data["train"].sampler.update_weights(*inlist)
                # self.data['train'].sampler.update_weights(indexes.cpu().numpy(), ws)

        if hasattr(self.data["train"].sampler, "get_weights"):
            self.logger.log_ws(epoch, self.data["train"].sampler.get_weights())
        if hasattr(self.data["train"].sampler, "reset_weights"):
            self.data["train"].sampler.reset_weights(epoch)

        # After every epoch, validation
        rsls = {"epoch": epoch}
        rsls_train = self.eval_with_preds(
            self.total_preds, self.total_labels, self.total_paths
        )
        # rsls_train = self.eval(phase="train")
        rsls_eval = self.eval(phase="val")
        rsls.update(rsls_train)
        rsls.update(rsls_eval)

        # Reset class weights for sampling if pri_mode is valid
        if hasattr(self.data["train"].sampler, "reset_priority"):
            ws = get_priority(
                self.data["train"].sampler.ptype,
                self.total_logits.detach(),
                self.total_labels,
            )
            self.data["train"].sampler.reset_priority(
                ws, self.total_labels.cpu().numpy()
            )

        # Log results
        self.logger.log_acc(rsls)

        # Under validation, the best model need to be updated
        if self.eval_acc_mic_top1 > self.best_acc:
            self.best_epoch = epoch
            self.best_acc = self.eval_acc_mic_top1
            # best_centroids = self.centroids
            self.best_model_weights["visual_model"] = copy.deepcopy(
                self.visual_model.state_dict()
            )
            self.best_model_weights["text_model"] = copy.deepcopy(
                self.text_model.state_dict()
            )
            # if self.training_opt["phaseA"] is not True:
            self.best_model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())

            if "proj" in self.variation:
                self.best_model_weights["im_proj"] = copy.deepcopy(
                    self.im_proj.state_dict()
                )

        print("===> Saving checkpoint")

        self.plateau_lr_metric = rsls_eval["val_all"]

        self.writer.add_scalar("Loss/train", self.epoch_loss, epoch)
        self.writer.add_scalar("Acc/train_all", rsls_train["train_all"], epoch)
        self.writer.add_scalar("Acc/train_many", rsls_train["train_many"], epoch)
        self.writer.add_scalar("Acc/train_median", rsls_train["train_median"], epoch)
        self.writer.add_scalar("Acc/train_low", rsls_train["train_low"], epoch)

        if "boosted" in self.variation:
            self.writer.add_scalar(
                "Acc_boosted/train_median_image",
                rsls_train["train_median_image"],
                epoch,
            )
            self.writer.add_scalar(
                "Acc_boosted/train_median_text", rsls_train["train_median_text"], epoch
            )
            self.writer.add_scalar(
                "Acc_boosted/train_low_image", rsls_train["train_low_image"], epoch
            )
            self.writer.add_scalar(
                "Acc_boosted/train_low_text", rsls_train["train_low_text"], epoch
            )

        self.writer.add_scalar("Acc/val_all", rsls_eval["val_all"], epoch)
        self.writer.add_scalar("Acc/val_many", rsls_eval["val_many"], epoch)
        self.writer.add_scalar("Acc/val_median", rsls_eval["val_median"], epoch)
        self.writer.add_scalar("Acc/val_low", rsls_eval["val_low"], epoch)

        self.writer.add_scalar(
            "Learning Rate",
            float([group["lr"] for group in self.model_optimizer.param_groups][0]),
            epoch,
        )

        self.save_latest(epoch)
        self.save_model(epoch, self.best_epoch, self.best_model_weights, self.best_acc)

    def train(self):
        # When training the network
        print_str = ["Phase: train"]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        print_write(["Do shuffle??? --- ", self.do_shuffle], self.log_file)

        # Initialize best model
        self.best_model_weights = {}
        self.best_model_weights["visual_model"] = copy.deepcopy(
            self.visual_model.state_dict()
        )
        self.best_model_weights["text_model"] = copy.deepcopy(
            self.text_model.state_dict()
        )
        # if self.training_opt["phaseA"] is not True:
        self.best_model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())
        if "proj" in self.variation:
            self.best_model_weights["im_proj"] = copy.deepcopy(
                self.im_proj.state_dict()
            )
        self.best_acc = 0.0
        self.best_epoch = 0
        # best_centroids = self.centroids

        end_epoch = self.training_opt["num_epochs"]

        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            self.train_epoch(epoch)

            if epoch % 10 == 0:
                rsls_eval_test = self.eval("test" if "test" in self.data else "val")

                self.writer.add_scalar(
                    "Acc/test_all", rsls_eval_test["test_all"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_many", rsls_eval_test["test_many"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_median", rsls_eval_test["test_median"], epoch
                )
                self.writer.add_scalar(
                    "Acc/test_low", rsls_eval_test["test_low"], epoch
                )

        print()
        print("Training Complete.")

        print_str = [
            "Best validation accuracy is %.3f at epoch %d"
            % (self.best_acc, self.best_epoch)
        ]
        print_write(print_str, self.log_file)
        # Save the best model and best centroids if calculated
        self.save_model(epoch, self.best_epoch, self.best_model_weights, self.best_acc)

        # Test on the test set
        # self.reset_model(best_model_weights)
        rsls_eval_test = self.eval("test" if "test" in self.data else "val")

        self.writer.add_scalar("Acc/test_all", rsls_eval_test["test_all"], epoch)
        self.writer.add_scalar("Acc/test_many", rsls_eval_test["test_many"], epoch)
        self.writer.add_scalar("Acc/test_median", rsls_eval_test["test_median"], epoch)
        self.writer.add_scalar("Acc/test_low", rsls_eval_test["test_low"], epoch)

        print("Done")

    def eval_with_preds(self, preds, labels, paths=[]):
        # Count the number of examples
        n_total = sum([len(p) for p in preds])

        # Split the examples into normal and mixup
        normal_preds, normal_labels, normal_paths = [], [], []

        if len(paths) > 0:
            for p, l, t in zip(preds, labels, paths):
                normal_preds.append(p)
                normal_labels.append(l)
                normal_paths.append(t)
        else:
            for p, l in zip(preds, labels):
                normal_preds.append(p)
                normal_labels.append(l)

        # Calculate normal prediction accuracy
        rsl = {
            "train_all": 0.0,
            "train_many": 0.0,
            "train_median": 0.0,
            "train_low": 0.0,
            "train_median_image": 0.0,
            "train_median_text": 0.0,
            "train_low_image": 0.0,
            "train_low_text": 0.0,
        }
        if len(normal_preds) > 0:
            if len(normal_paths) > 0:
                normal_preds, normal_labels, normal_paths = list(
                    map(np.concatenate, [normal_preds, normal_labels, normal_paths])
                )
            else:
                normal_preds, normal_labels = list(
                    map(np.concatenate, [normal_preds, normal_labels])
                )
            n_top1 = mic_acc_cal(normal_preds, normal_labels)
            shot_acc_output = shot_acc(
                normal_preds,
                normal_labels,
                self.data["train_ltcount"],
                paths=normal_paths,
            )
            (n_top1_many, n_top1_median, n_top1_low,) = (
                shot_acc_output["many_shot"],
                shot_acc_output["median_shot"],
                shot_acc_output["low_shot"],
            )
            rsl["train_all"] += len(normal_preds) / n_total * n_top1
            rsl["train_many"] += len(normal_preds) / n_total * n_top1_many
            rsl["train_median"] += len(normal_preds) / n_total * n_top1_median
            rsl["train_low"] += len(normal_preds) / n_total * n_top1_low

            if "boosted" in self.variation:
                rsl["train_median_image"] += (
                    len(normal_preds) / n_total * shot_acc_output["median_shot_image"]
                )
                rsl["train_low_image"] += (
                    len(normal_preds) / n_total * shot_acc_output["low_shot_image"]
                )

                rsl["train_median_text"] += (
                    len(normal_preds) / n_total * shot_acc_output["median_shot_text"]
                )
                rsl["train_low_text"] += (
                    len(normal_preds) / n_total * shot_acc_output["low_shot_text"]
                )

        # Top-1 accuracy and additional string
        print_str = [
            "\n Training acc Top1: %.3f \n" % (rsl["train_all"]),
            "Many_top1: %.3f" % (rsl["train_many"]),
            "Median_top1: %.3f" % (rsl["train_median"]),
            "Low_top1: %.3f" % (rsl["train_low"]),
            "\n",
        ]
        if "boosted" in self.variation:
            print_str += [
                "Median_image_top1: %.3f" % (rsl["train_median_image"]),
                "Low_image_top1: %.3f" % (rsl["train_low_image"]),
                "Median_text_top1: %.3f" % (rsl["train_median_text"]),
                "Low_text_top1: %.3f" % (rsl["train_low_text"]),
                "\n",
            ]
        print_write(print_str, self.log_file)

        return rsl

    def eval(self, phase="val", openset=False, save_feat=False):

        print_str = ["Phase: %s" % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print(
                "Under openset test mode. Open threshold is %.1f"
                % self.training_opt["open_threshold"]
            )

        torch.cuda.empty_cache()

        self.total_logits = torch.empty((0, self.training_opt["num_classes"])).cuda()
        self.total_labels = torch.empty(0, dtype=torch.long).cuda()
        self.total_paths = np.empty(0)

        get_feat_only = save_feat
        feats_all, labels_all, idxs_all, logits_all = [], [], [], []
        featmaps_all = []
        # Iterate over dataset
        for inputs, labels, indexes, paths in tqdm(self.data[phase]):
            inputs, labels = inputs.cuda(), labels.cuda()

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                self.batch_forward(inputs, labels, phase=phase)
                if not get_feat_only:
                    self.total_logits = torch.cat((self.total_logits, self.logits))
                    self.total_labels = torch.cat((self.total_labels, labels))
                    self.total_paths = np.concatenate((self.total_paths, paths))

                if get_feat_only:
                    logits_all.append(self.logits.cpu().numpy())
                    feats_all.append(self.features.cpu().numpy())
                    labels_all.append(labels.cpu().numpy())
                    idxs_all.append(paths.numpy())

        if get_feat_only:
            typ = "feat"
            if phase == "train_plain":
                name = "train{}_all.pkl".format(typ)
            elif phase == "test":
                name = "test{}_all.pkl".format(typ)
            elif phase == "val":
                name = "val{}_all.pkl".format(typ)

            fname = os.path.join(self.training_opt["log_dir"], name)
            print("===> Saving feats to " + fname)
            with open(fname, "wb") as f:
                pickle.dump(
                    {
                        "feats": np.concatenate(feats_all),
                        "labels": np.concatenate(labels_all),
                        "idxs": np.concatenate(idxs_all),
                    },
                    f,
                    protocol=4,
                )
            return
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)

        if openset:
            preds[probs < self.training_opt["open_threshold"]] = -1
            self.openset_acc = mic_acc_cal(
                preds[self.total_labels == -1],
                self.total_labels[self.total_labels == -1],
            )
            print("\n\nOpenset Accuracy: %.3f" % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        self.eval_acc_mic_top1 = mic_acc_cal(
            preds[self.total_labels != -1], self.total_labels[self.total_labels != -1]
        )
        self.precision, self.recall, self.eval_f_measure = F_measure(
            preds,
            self.total_labels,
            openset=openset,
            theta=self.training_opt["open_threshold"],
        )
        shot_acc_output = shot_acc(
            preds[self.total_labels != -1],
            self.total_labels[self.total_labels != -1],
            self.data["train_ltcount"],
            acc_per_cls=True,
        )
        (
            self.many_acc_top1,
            self.median_acc_top1,
            self.low_acc_top1,
            self.cls_accs,
        ) = (
            shot_acc_output["many_shot"],
            shot_acc_output["median_shot"],
            shot_acc_output["low_shot"],
            shot_acc_output["class_accs"],
        )
        # Top-1 accuracy and additional string
        print_str = [
            "\n\n",
            "Phase: %s" % (phase),
            "\n\n",
            "Evaluation_accuracy_micro_top1: %.5f" % (self.eval_acc_mic_top1),
            "\n",
            "Averaged Precision: %.5f" % (self.precision),
            "\n",
            "Averaged Recall: %.5f" % (self.recall),
            "\n",
            "Averaged F-measure: %.5f" % (self.eval_f_measure),
            "\n",
            "Many_shot_accuracy_top1: %.5f" % (self.many_acc_top1),
            "Median_shot_accuracy_top1: %.5f" % (self.median_acc_top1),
            "Low_shot_accuracy_top1: %.5f" % (self.low_acc_top1),
            "\n",
        ]

        rsl = {
            phase + "_all": self.eval_acc_mic_top1,
            phase + "_many": self.many_acc_top1,
            phase + "_median": self.median_acc_top1,
            phase + "_low": self.low_acc_top1,
            phase + "_fscore": self.eval_f_measure,
            phase + "_precision": self.precision,
        }

        if phase == "val":
            print_write(print_str, self.log_file)
        else:
            acc_str = [
                "{:.3f} \t {:.3f} \t {:.3f} \t {:.3f}".format(
                    self.many_acc_top1 * 100,
                    self.median_acc_top1 * 100,
                    self.low_acc_top1 * 100,
                    self.eval_acc_mic_top1 * 100,
                )
            ]
            if self.log_file is not None and os.path.exists(self.log_file):
                print_write(print_str, self.log_file)
                print_write(acc_str, self.log_file)
            else:
                print(*print_str)
                print(*acc_str)

        if phase == "test":
            with open(
                os.path.join(self.training_opt["log_dir"], "cls_accs.pkl"), "wb"
            ) as f:
                pickle.dump(self.cls_accs, f)
        return rsl

    def load_model(self, model_dir=None):
        model_dir = self.training_opt["log_dir"] if model_dir is None else model_dir

        if os.path.isfile(model_dir + "/final_model_checkpoint.pth"):

            model_dir += "/final_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict_best"]
            epoch = checkpoint["epoch"]
            print(f"Loading best model which was trained for {epoch} epochs")

        elif os.path.isfile(model_dir + "/latest_model_checkpoint.pth"):

            model_dir += "/latest_model_checkpoint.pth"
            checkpoint = torch.load(model_dir, map_location="cpu")
            model_state = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]
            print(f"Training hasn't finished, loading model trained for {epoch} epochs")
        else:
            print("No pretrained model")
            return

        print("Loading model from %s" % (model_dir))

        # checkpoint = torch.load(model_dir, map_location="cpu")
        # model_state = checkpoint["state_dict_best"]

        self.visual_model.load_state_dict(model_state["visual_model"])
        self.text_model.load_state_dict(model_state["text_model"])
        self.fusion.load_state_dict(model_state["fusion"])

        if "proj" in self.variation:
            self.im_proj.load_state_dict(model_state["im_proj"])

        # if self.test_mode is True:
        #     self.adapter.load_state_dict(model_state["classifier"])

    def save_latest(self, epoch):
        model_weights = {}
        model_weights["visual_model"] = copy.deepcopy(self.visual_model.state_dict())
        model_weights["text_model"] = copy.deepcopy(self.text_model.state_dict())
        # if self.training_opt["phaseA"] is not True:
        model_weights["fusion"] = copy.deepcopy(self.fusion.state_dict())
        if "proj" in self.variation:
            model_weights["im_proj"] = copy.deepcopy(self.im_proj.state_dict())

        model_states = {"epoch": epoch, "state_dict": model_weights}

        model_dir = os.path.join(
            self.training_opt["log_dir"], "latest_model_checkpoint.pth"
        )
        torch.save(model_states, model_dir)

    def save_model(
        self, epoch, best_epoch, best_model_weights, best_acc, centroids=None
    ):

        model_states = {
            "epoch": epoch,
            "best_epoch": best_epoch,
            "state_dict_best": best_model_weights,
            "best_acc": best_acc,
            "centroids": centroids,
        }

        model_dir = os.path.join(
            self.training_opt["log_dir"], "final_model_checkpoint.pth"
        )

        torch.save(model_states, model_dir)

    def output_logits(self, openset=False):
        filename = os.path.join(
            self.training_opt["log_dir"], "logits_%s" % ("open" if openset else "close")
        )
        print("Saving total logits to: %s.npz" % filename)
        np.savez(
            filename,
            logits=self.total_logits.detach().cpu().numpy(),
            labels=self.total_labels.detach().cpu().numpy(),
            paths=self.total_paths,
        )
