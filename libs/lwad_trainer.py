import torch
import torch.nn.functional as F

from libs.lwad_config import DEFAULT_THRESHOLD
from libs.lwad_attack import generate_attack, DEFAULT_TRAIN_ATTACK
from libs.lwad_evaluator import predict


def train_epoch(model, loader, optimizer, eps, lambda_det=1.0, lambda_act=1.0,
                task_loss_on_adv=False, class_weights=None,
                attack_mask=None, attack=DEFAULT_TRAIN_ATTACK, threshold=DEFAULT_THRESHOLD,
                attack_kwargs=None, device="cpu"):
    """Agnostic training loop:

        loss = task + lambda_det * det_loss + lambda_act * act_loss

    where det_loss and act_loss appear only if the network contains the layers
    that produce them (DetectorLayer for det_loss, FurtherAL/NearestAL for act_loss).
    """
    
    model.train()
    attack_kwargs = attack_kwargs or {}
    tot_task_loss, tot_det_loss, tot_act_loss = 0.0, 0.0, 0.0
    tot_task_correct_clean_preds, tot_task_correct_adv_preds = 0, 0
    tot_det_correct_clean_preds, tot_det_correct_adv_preds = 0, 0
    tot_n = 0
    has_det, has_act = False, False

    for x, y in loader:
        # 1) advrsarial version of real batch
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask,
                                **attack_kwargs)

        # 2) mixed batch + real(0)/adversarial(1) flag
        xb = torch.cat([x, x_adv])
        yb = torch.cat([y, y])
        adv_flag = torch.cat([torch.zeros(len(x), device=x.device),
                              torch.ones(len(x_adv), device=x.device)])

        # 3) forward
        logits, state = model(xb, labels=yb, is_adv=adv_flag)

        # 4) task loss
        if task_loss_on_adv:
            task_loss = F.cross_entropy(logits, yb, weight=class_weights)
        else:
            real = adv_flag == 0
            task_loss = F.cross_entropy(logits[real], yb[real],
                                        weight=class_weights)

        # 5) total loss: contributions exist only if layers produce them
        loss = task_loss
        if state.det_loss is not None:
            loss = loss + lambda_det * state.det_loss
        if state.act_loss is not None:
            loss = loss + lambda_act * state.act_loss

        optimizer.zero_grad()   # gradients accumulate, alweys reset them before backward pass
        loss.backward()         # compute the gradients traversing autgrad graph
        optimizer.step()        # updates model parameters based on their lr

        n = len(x)
        tot_n += n
        tot_task_loss += task_loss.item() * n
        if state.det_loss is not None:
            has_det = True
            tot_det_loss += state.det_loss.item() * n
        if state.act_loss is not None:
            has_act = True
            tot_act_loss += state.act_loss.item() * n

        tot_task_correct_clean_preds += (logits[:n].argmax(-1) == y).sum().item()
        tot_task_correct_adv_preds += (logits[n:].argmax(-1) == y).sum().item()

        with torch.no_grad():
            score = state.adv_score()
        if score is not None:
            det_pred_adv = score > threshold # => adversarial sample
            tot_det_correct_clean_preds += (~det_pred_adv[:n]).sum().item()
            tot_det_correct_adv_preds += det_pred_adv[n:].sum().item()

    return {"task_loss": tot_task_loss / tot_n,
            "det_loss": tot_det_loss / tot_n if has_det else None,
            "act_loss": tot_act_loss / tot_n if has_act else None,
            "task_clean_acc": tot_task_correct_clean_preds / tot_n,
            "task_adv_acc": tot_task_correct_adv_preds / tot_n,
            "det_clean_acc": tot_det_correct_clean_preds / tot_n if has_det else None,
            "det_adv_acc": tot_det_correct_adv_preds / tot_n if has_det else None}




def select_threshold(model, X_val, y_val, eps, attack_mask=None, attack=DEFAULT_TRAIN_ATTACK,
                     device="cpu", batch_size=4096, grid=99, attack_kwargs=None):
    """
    Chooses the detector threashold maximizing its balanced accuracy, as the mean
    between adversarial attack (score above threshold) and clean data correctly
    classified (score under the threshold)"""

    if not getattr(model, "has_detectors", True):
        raise ValueError("select_threshold requires a detector-based architecture")

    model.eval()
    attack_kwargs = attack_kwargs or {}
    sc_c, sc_a = [], []
    for i in range(0, len(X_val), batch_size):
        x = X_val[i:i + batch_size]
        y = y_val[i:i + batch_size]
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask,
                                **attack_kwargs)
        _, s_c, _ = predict(model, x)
        _, s_a, _ = predict(model, x_adv)
        sc_c.append(s_c); sc_a.append(s_a)
    sc_c, sc_a = torch.cat(sc_c), torch.cat(sc_a)

    ts = torch.linspace(0.01, 0.99, grid, device=sc_c.device)
    tpr = (sc_a.unsqueeze(0) > ts.unsqueeze(1)).float().mean(dim=1)   # (grid,)
    tnr = (sc_c.unsqueeze(0) <= ts.unsqueeze(1)).float().mean(dim=1)  # (grid,)
    bal = 0.5 * (tpr + tnr)
    best_idx = int(bal.argmax())
    return ts[best_idx].item(), bal[best_idx].item()