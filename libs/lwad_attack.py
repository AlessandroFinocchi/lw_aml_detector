import contextlib
import torch
import torch.nn.functional as F

from enum import Enum
from libs.lwad_wrapper import DetectorLayer


class Attack(Enum):
    FGSM         = "fgsm"
    PGD          = "pgd"
    PGD_ADAPTIVE = "pgd_adaptive"

TRAIN_ATTACK = Attack.PGD.value            # attack to train defense
EVAL_ATTACK  = Attack.PGD_ADAPTIVE.value   # attack to evaluate defense

EPS = 0.1                   # attack intensity, in standardized measurement unit
PGD_STEPS = 20              # iterations number for PGD / adaptive PGD
PGD_ALPHA = EPS / 4         # amplitude of iteration step
PGD_EVADE_WEIGHT = 1.0      # evade term weight in adaptive PGD


# ===========================================================================
# FGSM on attackable features
# ===========================================================================
def fgsm(model, x, y, eps, mask=None):
    x_adv = x.clone().detach().requires_grad_(True)
    logits, _ = model(x_adv)
    loss = F.cross_entropy(input=logits, target=y)
    (grad,) = torch.autograd.grad(outputs=loss, inputs=x_adv)
    delta = eps * grad.sign()
    if mask is not None:
        delta = delta * mask.to(x.device)
    return (x_adv + delta).detach()


# ===========================================================================
# standard PGD limited inside the L-inf ball
# ===========================================================================
def pgd(model, x, y, eps, steps, alpha, mask=None):
    x_orig = x.clone().detach()
    # random start inside L-inf ball (on attackable features)
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    if mask is not None:
        x_adv = x_orig + (x_adv - x_orig) * mask.to(x.device)
    x_adv = x_adv.detach()

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits, _ = model(x_adv)
        loss = F.cross_entropy(input=logits, target=y)
        (grad,) = torch.autograd.grad(outputs=loss, inputs=x_adv)
        with torch.no_grad():
            delta = alpha * grad.sign()
            if mask is not None:
                delta = delta * mask.to(x.device)
            x_adv = x_adv + delta
            # projection: keep |x_adv - x_orig| <= eps
            x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
        x_adv = x_adv.detach()
    return x_adv


# ===========================================================================
# Adaptive PGD to evade both task and detector classifications
# ===========================================================================
@contextlib.contextmanager
def _detector_grad_enabled(model):
    layers = [m for m in model.modules() if isinstance(m, DetectorLayer)]
    saved = [layer.detach for layer in layers] # layer detach states backup
    for layer in layers:
        layer.detach = False
    try:
        yield
    finally:
        for layer, s in zip(layers, saved):
            layer.detach = s


def pgd_adaptive(model, x, y, eps, steps, alpha, mask=None, evade_weight=1.0):
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    if mask is not None:
        x_adv = x_orig + (x_adv - x_orig) * mask.to(x.device)
    x_adv = x_adv.detach()

    with _detector_grad_enabled(model):
        for _ in range(steps):
            x_adv.requires_grad_(True)
            logits, state = model(x_adv)
            task_loss = F.cross_entropy(input=logits, target=y)
            score = state.adv_score().mean() # mean "adversarial" probability
            objective = task_loss - evade_weight * score
            (grad,) = torch.autograd.grad(outputs=objective, inputs=x_adv)
            with torch.no_grad():
                delta = alpha * grad.sign()
                if mask is not None:
                    delta = delta * mask.to(x.device)
                x_adv = x_adv + delta
                x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = x_adv.detach()
    return x_adv


# ===========================================================================
# Attack selection
# ===========================================================================
def generate_attack(model, x, y, eps, attack, mask=None):
    attack = Attack(attack)
    if attack is Attack.FGSM:
        return fgsm(model, x, y, eps, mask=mask)
    if attack is Attack.PGD:
        return pgd(model, x, y, eps, steps=PGD_STEPS, alpha=PGD_ALPHA, mask=mask)
    if attack is Attack.PGD_ADAPTIVE:
        return pgd_adaptive(model, x, y, eps, steps=PGD_STEPS, alpha=PGD_ALPHA,
                            mask=mask, evade_weight=PGD_EVADE_WEIGHT)
    raise ValueError(f"Unknown attack: {attack!r}")