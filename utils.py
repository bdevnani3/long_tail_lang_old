import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
import torch.nn.functional as F
import importlib
import pdb
import shutil
import os


def source_import(file_path):
    """This function imports python module directly from source code using importlib"""
    spec = importlib.util.spec_from_file_location("", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_show(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def print_write(print_str, log_file):
    print(*print_str)
    if log_file is None:
        return
    with open(log_file, "a") as f:
        print(*print_str, file=f)


def init_weights(model, weights_path, caffe=False, classifier=False):
    """Initialize weights"""
    print(
        "Pretrained %s weights path: %s"
        % ("classifier" if classifier else "feature model", weights_path)
    )
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {
                k: weights[k] if k in weights else model.state_dict()[k]
                for k in model.state_dict()
            }
        else:
            weights = weights["state_dict_best"]["feat_model"]
            weights = {
                k: weights["module." + k]
                if "module." + k in weights
                else model.state_dict()[k]
                for k in model.state_dict()
            }
    else:
        weights = weights["state_dict_best"]["classifier"]
        weights = {
            k: weights["module.fc." + k]
            if "module.fc." + k in weights
            else model.state_dict()[k]
            for k in model.state_dict()
        }
    model.load_state_dict(weights)
    return model


def shot_acc(
    preds,
    labels,
    train_data_count,
    many_shot_thr=100,
    low_shot_thr=20,
    acc_per_cls=False,
    paths=[],
):
    # if isinstance(train_data, np.ndarray):
    #     training_labels = np.array(train_data).astype(int)
    # else:
    #     training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError("Type ({}) of preds not supported".format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []

    test_class_count_images = []
    test_class_count_text = []
    image_correct = []
    text_correct = []

    for l in np.unique(labels):
        train_class_count.append(train_data_count[l])
        test_class_count.append(len(labels[labels == l]))
        class_correct.append((preds[labels == l] == labels[labels == l]).sum())

        if len(paths) > 0:
            test_class_count_images.append(len(labels[(labels == l) & (paths == 1)]))
            test_class_count_text.append(len(labels[(labels == l) & (paths != 1)]))
            image_correct.append(
                (
                    preds[(labels == l) & (paths == 1)]
                    == labels[(labels == l) & (paths == 1)]
                ).sum()
            )
            text_correct.append(
                (
                    preds[(labels == l) & (paths != 1)]
                    == labels[(labels == l) & (paths != 1)]
                ).sum()
            )

    many_shot = []
    median_shot = []
    low_shot = []

    median_shot_text = []
    median_shot_image = []

    low_shot_text = []
    low_shot_image = []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))

            if len(paths) > 0:
                # fine grained info for boosted variants
                low_shot_image.append((image_correct[i] / test_class_count_images[i]))
                low_shot_text.append((text_correct[i] / test_class_count_text[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

            if len(paths) > 0:
                # fine grained info for boosted variants
                median_shot_image.append(
                    (image_correct[i] / test_class_count_images[i])
                )
                median_shot_text.append((text_correct[i] / test_class_count_text[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    out = {
        "many_shot": np.mean(many_shot),
        "median_shot": np.mean(median_shot),
        "low_shot": np.mean(low_shot),
        "low_shot_image": np.mean(low_shot_image),
        "low_shot_text": np.mean(low_shot_text),
        "median_shot_image": np.mean(median_shot_image),
        "median_shot_text": np.mean(median_shot_text),
    }

    class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)]
    if acc_per_cls:
        out["class_accs"] = class_accs
    return out


def F_measure(preds, labels, openset=False, theta=None):

    if openset:
        # f1 score for openset evaluation
        true_pos = 0.0
        false_pos = 0.0
        false_neg = 0.0

        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += (
                1 if preds[i] != labels[i] and labels[i] != -1 and preds[i] != -1 else 0
            )
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return (
            precision_score(
                labels.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                average="macro",
            ),
            recall_score(
                labels.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                average="macro",
            ),
            f1_score(
                labels.detach().cpu().numpy(),
                preds.detach().cpu().numpy(),
                average="macro",
            ),
        )


def mic_acc_cal(preds, labels):
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (
            lam * preds.eq(targets_a.data).cpu().sum().float()
            + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()
        ) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


# New Added
def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == "score":
        ws = 1 - logits2score(logits, labels)
    elif ptype == "entropy":
        ws = logits2entropy(logits)
    elif ptype == "CE":
        ws = logits2CE(logits, labels)

    return ws


def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


def copy_current_codebase_to_path(path):
    # Taken from https://stackoverflow.com/questions/28985921/exclude-some-of-subfolders-while-copying-im-using-copytree-in-python

    to_exclude = [
        "/nethome/bdevnani3/flash1/long_tail_lang/data",
        "/nethome/bdevnani3/flash1/long_tail_lang/datasets",
        "/nethome/bdevnani3/flash1/long_tail_lang/results",
        "/nethome/bdevnani3/flash1/long_tail_lang/runs",
    ]

    # ignores excluded directories and .exe files
    def get_ignored(path, filenames):
        ret = []
        for filename in filenames:
            if os.path.join(path, filename) in to_exclude:
                ret.append(filename)
        return ret

    shutil.copytree(
        "/nethome/bdevnani3/flash1/long_tail_lang", path, ignore=get_ignored
    )


def check_config(conf, field):
    if field not in conf:
        return False
    return conf[field]
