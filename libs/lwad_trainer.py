import torch
import torch.nn.functional as F

from libs.lwad_config import THRESHOLD
from libs.lwad_attack import generate_attack, TRAIN_ATTACK
from libs.lwad_evaluator import predict


def train_epoch(model, loader, optimizer, eps, lambda_det=1.0,
                task_loss_on_adv=False, class_weights=None,
                attack_mask=None, attack=TRAIN_ATTACK, device="cpu"):
    model.train()
    tot_task_loss, tot_det_loss = 0.0, 0.0
    tot_task_correct_clean_preds, tot_task_correct_adv_preds = 0, 0
    tot_det_correct_clean_preds, tot_det_correct_adv_preds = 0, 0
    tot_n = 0

    for x, y in loader:
        # 1) advrsarial version of real batch 
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask)

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

        loss = task_loss + lambda_det * state.det_loss
        optimizer.zero_grad()   # gradients accumlate, so always reset them before backward pass
        loss.backward()         # compute the gradients traversing autgrad graph
        optimizer.step()        # updates model parameters based on their lr

        n = len(x)
        tot_n += n
        tot_task_loss += task_loss.item() * n
        tot_det_loss += state.det_loss.item() * n

        tot_task_correct_clean_preds += (logits[:n].argmax(-1) == y).sum().item()
        tot_task_correct_adv_preds += (logits[n:].argmax(-1) == y).sum().item()

        with torch.no_grad():
            det_pred_adv = state.adv_score() > THRESHOLD # => adversarial sample
        tot_det_correct_clean_preds += (~det_pred_adv[:n]).sum().item()
        tot_det_correct_adv_preds += det_pred_adv[n:].sum().item()

    return {"task_loss": tot_task_loss / tot_n,
            "det_loss": tot_det_loss / tot_n,
            "task_clean_acc": tot_task_correct_clean_preds / tot_n,
            "task_adv_acc": tot_task_correct_adv_preds / tot_n,
            "det_clean_acc": tot_det_correct_clean_preds / tot_n,
            "det_adv_acc": tot_det_correct_adv_preds / tot_n}




def select_threshold(model, X_val, y_val, eps, attack_mask=None, attack=TRAIN_ATTACK,
                     device="cpu", batch_size=4096, grid=99):
    """
    Chooses the detector threashold maximizing its balanced accuracy, as the mean
    between adversarial attack (score above threshold) and clean data correctly 
    classified (score under the threshold)ì"""
    model.eval()
    sc_c, sc_a = [], []
    for i in range(0, len(X_val), batch_size):
        x = X_val[i:i + batch_size]
        y = y_val[i:i + batch_size]
        x_adv = generate_attack(model, x, y, eps, attack, mask=attack_mask)
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
