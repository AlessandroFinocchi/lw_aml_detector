import torch

from libs.model.lwad_config import DEFAULT_THRESHOLD_DET, ScoreMode, DEFAULT_SCORE_MODE
from libs.attacks.lwad_attack import (generate_attack, DEFAULT_EVAL_ATTACK,
                              DEFAULT_SCORE_REDUCE)


# ===========================================================================
# Experiment comparison metric within different architectures
#
# The two architectures cannot be ranked on their own metrics: ali. architecture
# defends by FLAGGING adversarial samples, det. architecture by CLASSIFYING them. 
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
    For architectures of type 2 (NearestAL) score and flags are None."""
    model.eval()
    logits, state = model(x)
    score = state.adv_score(reduce=reduce)
    flags = (score > threshold_det) if score is not None else None
    return logits.argmax(-1), score, flags

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

    For architectures of type 2 (NearestAL) detector voices are None"""
    model.eval()
    attack_kwargs = attack_kwargs or {}
    res = {"lab_c": [], "lab_a": [], "sc_c": [], "sc_a": []}
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

    lab_c, lab_a = torch.cat(res["lab_c"]), torch.cat(res["lab_a"])

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
           # degenerates into the plain task accuracies (architecture 2)
           "clean_acc_e2e": correct_clean.float().mean().item(),
           "robust_acc_e2e": correct_adv.float().mean().item()}

    # --- DETECTOR (positive = adversarial) ---------------------------------
    # only for architecture of type 1
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