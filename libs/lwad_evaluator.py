import torch

from libs.lwad_config import THRESHOLD
from libs.lwad_attack import generate_attack, EVAL_ATTACK

@torch.no_grad()
def predict(model, x, threshold=THRESHOLD, reduce="mean"):
    """returns (predicted labels, adversarial score, clean-adversarial flags)."""
    model.eval()
    logits, state = model(x)
    score = state.adv_score(reduce=reduce)
    return logits.argmax(-1), score, score > threshold

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

def evaluate(model, X_te, y_te, eps, attack_mask=None, attack=EVAL_ATTACK,
             device="cpu", batch_size=4096, threshold=THRESHOLD):
    """
    Task and Detector metrics on batch set
    
    Return a dictionary with acc, prec and recall for both classifications
 
      task_clean    : task metrics on clean data
      task_adv      : task metrics on adv data
      det_clean_acc : detector accuracy on clean data  (= 1 - FPR)
      det_adv_acc   : detector accuracy on adv data (= TPR = recall)
      detector      : detector accuracy, precision and recall on the clean and
                      adversarial mixed set
    """
    model.eval()
    res = {"lab_c": [], "lab_a": [], "sc_c": [], "sc_a": []}
    for i in range(0, len(X_te), batch_size):
        x = X_te[i:i + batch_size].to(device)
        y = y_te[i:i + batch_size].to(device)
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask)  #  needs grad
        lab_c, sc_c, _ = predict(model, x, threshold=threshold)
        lab_a, sc_a, _ = predict(model, x_adv, threshold=threshold)
        res["lab_c"].append(lab_c.cpu()); res["sc_c"].append(sc_c.cpu())
        res["lab_a"].append(lab_a.cpu()); res["sc_a"].append(sc_a.cpu())
 
    lab_c, lab_a = torch.cat(res["lab_c"]), torch.cat(res["lab_a"])
    sc_c, sc_a = torch.cat(res["sc_c"]), torch.cat(res["sc_a"])
 
    # --- TASK (positive = attacco = label 1) --------------------------------
    task_clean = _binary_metrics(lab_c, y_te, positive=1)
    task_adv = _binary_metrics(lab_a, y_te, positive=1)
 
    # --- DETECTOR (positive = adversarial) ---------------------------------
    # ground truth: clean = 0, adversarial = 1
    det_pred_clean = (sc_c > threshold).long()   # should be 0
    det_pred_adv = (sc_a > threshold).long()     # should be 1
    det_true_clean = torch.zeros_like(det_pred_clean)
    det_true_adv = torch.ones_like(det_pred_adv)
 
    det_clean_acc = (det_pred_clean == det_true_clean).float().mean().item()
    det_adv_acc = (det_pred_adv == det_true_adv).float().mean().item()
 
    det_pred = torch.cat([det_pred_clean, det_pred_adv])
    det_true = torch.cat([det_true_clean, det_true_adv])
    detector = _binary_metrics(det_pred, det_true, positive=1)
 
    return {"task_clean": task_clean,
            "task_adv": task_adv,
            "det_clean_acc": det_clean_acc,
            "det_adv_acc": det_adv_acc,
            "detector": detector,
            "score_clean": sc_c.mean().item(),
            "score_adv": sc_a.mean().item()}
