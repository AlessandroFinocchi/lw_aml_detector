import statistics
import time

import torch
import torch.nn as nn

from libs.model.lwad_config import DEFAULT_THRESHOLD_DET, ScoreMode, DEFAULT_SCORE_MODE
from libs.attacks.lwad_attack import (generate_attack, DEFAULT_EVAL_ATTACK,
                              DEFAULT_SCORE_REDUCE)


# ===========================================================================
# Experiment comparison metric within different models
#
# The two models cannot be ranked on their own metrics: adv. training model
# defends by FLAGGING adversarial samples, det. moel by classifying them. 
# The end-to-end view makes them comparable:
#
#   clean_acc_e2e  = clean sample classified right AND not flagged
#   robust_acc_e2e = adv sample classified right OR flagged
# ===========================================================================
def combined_score(metrics: dict, mode=DEFAULT_SCORE_MODE) -> float:
    """Collapses the end-to-end metrics into the single number every
    experiment is ranked on. `mode` accepts a ScoreMode or its string value."""
    mode = ScoreMode(mode)
    clean, robust = metrics["clean_acc_e2e"], metrics["robust_acc_e2e"]
    if mode is ScoreMode.CLEAN:
        return clean
    if mode is ScoreMode.ROBUST:
        return robust
    return 0.5 * (clean + robust)

@torch.no_grad()
def predict(model, x, threshold_det=DEFAULT_THRESHOLD_DET, reduce=DEFAULT_SCORE_REDUCE):
    """Returns (predicted labels, adversarial score, clean-adversarial flags).
    For models of type 2 (CloserAL) score and flags are None."""
    model.eval()
    return _infer(model, x, threshold_det, reduce)

def _infer(model, x, threshold_det, reduce):
    logits, state = model(x)
    score = state.adv_score(reduce=reduce)
    flags = (score > threshold_det) if score is not None else None
    return logits.argmax(-1), score, flags

@torch.no_grad()
def _hidden_preacts(model, x):
    """Pre-activations a^(l) of every hidden Linear, in network order. Runs
    the bare base modules (no detectors); the output Linear (logits) is dropped."""
    acts = []
    for layer in model.layers:
        x = layer.base(x)
        if isinstance(layer.base, nn.Linear):
            acts.append(x)
    return acts[:-1]

def _binary_metrics(pred, true, positive=1):
    """Accuracy, precision and recall for binary classification.
    `positive` indicates which class counts as "positive" for precision/recall"""
    pred = pred.long()
    true = true.long()
    tp = int(((pred == positive) & (true == positive)).sum())
    fp = int(((pred == positive) & (true != positive)).sum())
    fn = int(((pred != positive) & (true == positive)).sum())
    tn = int(((pred != positive) & (true != positive)).sum())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"acc": acc, "precision": precision, "recall": recall}

def evaluate(model, X_te, y_te, eps, attack_mask=None, attack=DEFAULT_EVAL_ATTACK,
             device="cpu", batch_size=4096, threshold_det=DEFAULT_THRESHOLD_DET,
             attack_kwargs=None, reduce=DEFAULT_SCORE_REDUCE):
    """
    Task and Detector metrics on batch set

    Return a dictionary with acc, prec and recall for both classifications

      task_clean    : task metrics on clean data
      task_adv      : task metrics on adv data
      det_clean_acc : detector accuracy on clean data  (= 1 - FPR)
      det_adv_acc   : detector accuracy on adv data (= TPR = recall)
      detector      : detector accuracy, precision and recall on the clean and
                      adversarial mixed set
      clean_acc_e2e : clean sample classified right AND not flagged
      robust_acc_e2e: adv sample classified right OR flagged
      slav          : SLAV of every hidden layer, list of N floats
      slav_rel      : SLAV over the clean activation scale, list of N floats

    For models of type 2 (CloserAL) detector voices are None"""
    model.eval()
    attack_kwargs = attack_kwargs or {}
    res = {"lab_c": [], "lab_a": [], "sc_c": [], "sc_a": [], "dev": [], "ref": []}
    for i in range(0, len(X_te), batch_size):
        x = X_te[i:i + batch_size]
        y = y_te[i:i + batch_size]
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask,
                                **attack_kwargs)  #  needs grad
        lab_c, sc_c, _ = predict(model, x, threshold_det=threshold_det, reduce=reduce)
        lab_a, sc_a, _ = predict(model, x_adv, threshold_det=threshold_det, reduce=reduce)
        res["lab_c"].append(lab_c)
        res["lab_a"].append(lab_a)
        if sc_c is not None:
            res["sc_c"].append(sc_c); res["sc_a"].append(sc_a)

        # per sample, per layer 
        a_c, a_a = _hidden_preacts(model, x), _hidden_preacts(model, x_adv)
        # slav = (1/d_l) ||a(x_adv) - a(x)||^2
        res["dev"].append(torch.stack([(a - c).pow(2).mean(-1) for c, a in zip(a_c, a_a)]))
        # rho =  (1/d_l) ||a(x)||^2
        res["ref"].append(torch.stack([c.pow(2).mean(-1) for c in a_c]))

    lab_c, lab_a = torch.cat(res["lab_c"]), torch.cat(res["lab_a"])
    slav = torch.cat(res["dev"], dim=1).mean(dim=1)
    rho = torch.cat(res["ref"], dim=1).mean(dim=1)

    # --- TASK (positive = attack = label 1) --------------------------------
    task_clean = _binary_metrics(lab_c, y_te, positive=1)
    task_adv = _binary_metrics(lab_a, y_te, positive=1)

    correct_clean = lab_c.long() == y_te.long()
    correct_adv = lab_a.long() == y_te.long()

    out = {"task_clean": task_clean,
           "task_adv": task_adv,
           "det_clean_acc": None,
           "det_adv_acc": None,
           "detector": None,
           "score_clean": None,
           "score_adv": None,
           # no detector -> nothing is ever flagged, so the end-to-end view
           # degenerates into the plain task accuracies
           "clean_acc_e2e": correct_clean.float().mean().item(),
           "robust_acc_e2e": correct_adv.float().mean().item(),
           "slav": slav.tolist(),
           "slav_rel": (slav / rho).tolist()}

    # --- DETECTOR (positive = adversarial) ---------------------------------
    # only for detector models
    if res["sc_c"]:
        sc_c, sc_a = torch.cat(res["sc_c"]), torch.cat(res["sc_a"])

        # ground truth: clean = 0, adversarial = 1
        det_pred_clean = (sc_c > threshold_det).long()   # should be 0
        det_pred_adv = (sc_a > threshold_det).long()     # should be 1
        det_true_clean = torch.zeros_like(det_pred_clean)
        det_true_adv = torch.ones_like(det_pred_adv)

        out["det_clean_acc"] = (det_pred_clean == det_true_clean).float().mean().item()
        out["det_adv_acc"] = (det_pred_adv == det_true_adv).float().mean().item()

        det_pred = torch.cat([det_pred_clean, det_pred_adv])
        det_true = torch.cat([det_true_clean, det_true_adv])
        out["detector"] = _binary_metrics(det_pred, det_true, positive=1)
        out["score_clean"] = sc_c.mean().item()
        out["score_adv"] = sc_a.mean().item()

        # a false alarm on a clean sample is a missclassification, 
        # an adversarial alarm is a success even when the classifier gets fooled
        flag_c, flag_a = det_pred_clean.bool(), det_pred_adv.bool()
        out["clean_acc_e2e"] = (correct_clean & ~flag_c).float().mean().item()
        out["robust_acc_e2e"] = (correct_adv | flag_a).float().mean().item()

    return out


# ===========================================================================
# Inference latency
# ===========================================================================
DEFAULT_INFER_SAMPLES = 1024    # samples timed, one forward each
DEFAULT_INFER_WARMUP = 256      # untimed forwards: cuBLAS init, allocator, caches

@torch.no_grad()
def inference_time(model, X, n_samples=DEFAULT_INFER_SAMPLES,
                   warmup=DEFAULT_INFER_WARMUP, threshold_det=DEFAULT_THRESHOLD_DET,
                   reduce=DEFAULT_SCORE_REDUCE):
    """Single-sample inference latency on the first n_samples rows of X.

    Every sample goes through the full decision (label, adversarial score and
    flag) alone, as a flow would when it reaches a deployed detector: batching
    them would hide the per-sample overhead and leave no per-sample spread to
    measure. CUDA launches are asynchronous, hence the synchronize on both
    sides of the timed call.

    Returns inference time in ms."""
    model.eval()
    n = min(n_samples, len(X))
    sync = torch.cuda.synchronize if X.is_cuda else (lambda: None)
    for i in range(min(warmup, len(X))):
        _infer(model, X[i:i + 1], threshold_det, reduce)

    times = []
    for i in range(n):
        x = X[i:i + 1]
        sync()
        t0 = time.perf_counter()
        _infer(model, x, threshold_det, reduce)
        sync()
        times.append((time.perf_counter() - t0) * 1e3)

    if not times:
        return {"infer_ms": float("nan"), "infer_ms_std": float("nan"), "infer_n": 0}
    return {"infer_ms": statistics.fmean(times),
            "infer_ms_std": statistics.stdev(times) if n > 1 else 0.0,
            "infer_n": n}