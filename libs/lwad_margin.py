"""Per-layer calibration and automatic selection of FurtherAL margins.

Two levels:

  suggest_margins  - fast calibration: measures the natural real/adv distance
                     d_i of every layer after a short warmup trained WITHOUT
                     the activation loss, and proposes margin_i = factor * d_i.

  select_margins   - automatic selection: keeps the parametrization
                     margin_i = factor * d_i (so every layer gets a margin on
                     its own scale) and searches the best factor by briefly
                     training one candidate per value and comparing the
                     validation scores. The search stays 1-D with any number
                     of layers.

usage example:
    res = select_margins(cfg, X_train, y_train, X_val, y_val,
                         attack_mask=attack_mask, device=device,
                         class_weights=class_weights)
    cfg = lc.DetectorArchConfig(act_margin=res.margins)
"""

from __future__ import annotations

import torch
import dataclasses
from typing import Optional

import libs.lwad_wrapper as lw
import libs.lwad_config as lc
import libs.lwad_trainer as lt
import libs.lwad_evaluator as le
from libs.lwad_attack import generate_attack


# ===========================================================================
# Measurement
# ===========================================================================
@torch.no_grad()
def _measure_layer_distances(model, x, x_adv):
    """For every FurtherAL layer (in network order): mean squared distance d
    between real and adv activations, plus the activation scale.
    Only FurtherAL layers own a margin, so only those are measured: on an
    architecture without FurtherAL the list comes back empty."""
    rows = []
    h_real, h_adv = x, x_adv
    for layer in model.layers:
        # hidden activation, base for no FlowState
        h_real = layer.base(h_real)
        h_adv = layer.base(h_adv)
        if isinstance(layer, lw.FurtherAL):
            d = (h_adv - h_real).pow(2).mean(dim=-1).mean().item()  # real/adv act. distance
            scale = h_real.pow(2).mean().item() ** 0.5              # layer activation magnitude
            rows.append({"layer": type(layer).__name__,
                         "d": d, "scale": scale})
    return rows


def _warmup(config, X, y, attack_mask, device, epochs, class_weights=None):
    """Trains the model briefly WITHOUT the activation loss (lambda_act=0).

    The margin is a hyperparameter of training, so it cannot be computed on an
    already trained model. Training with lambda_act=0 breaks that: the warmed-up 
    model does not depend on the margin at all, yet its activations are about
    the order of magnitue the margin will actually operate in."""
    torch.manual_seed(lc.SEED)
    built = lc.create_architecture(config, X.shape[1], device=device)
    if epochs <= 0:
        return built.model
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y),
        batch_size=config.batch_size, shuffle=True,
    )
    for _ in range(epochs):
        lt.train_epoch(built.model, loader, built.optimizer, eps=config.eps,
                       lambda_det=getattr(config, "lambda_det", 0.0),
                       lambda_act=0.0,                  # no margin involved
                       task_loss_on_adv=config.task_loss_on_adv,
                       class_weights=class_weights, attack_mask=attack_mask,
                       attack=config.train_attack,
                       threshold=getattr(config, "threshold", 0.5),
                       attack_kwargs=config.attack_kwargs(), device=device)
    return built.model


def _natural_distances(config, X, y, attack_mask, device, warmup_epochs=3,
                       class_weights=None):
    """Activation real vs adv distances, measured after `warmup_epochs` epochs 
    trained without the activation loss (warmup_epochs=0 means untrained model).

    Warmup matters: on an untrained model the attack barely hits and, more 
    importantly, the distances are not representative."""
    model = _warmup(config, X, y, attack_mask, device, warmup_epochs,
                    class_weights=class_weights)
    model.eval()
    x, yy = X.to(device), y.to(device)
    x_adv = generate_attack(model, x, yy, config.eps, config.train_attack,
                            mask=attack_mask, **config.attack_kwargs())
    rows = _measure_layer_distances(model, x, x_adv)
    base_d = tuple(max(r["d"], 1e-8) for r in rows)  # for not 0 margins
    return base_d, rows


# ===========================================================================
# Calibration (short warm-up, no margin search)
# ===========================================================================
def suggest_margins(config, X, y, attack_mask=None, device="cpu",
                    factor=2.0, warmup_epochs=3, class_weights=None,
                    verbose=True) -> Optional[tuple]:
    """Proposes a PER-LAYER margin: margin_i = factor * natural d_i.
    Returns a tuple ordered like the FurtherAL layers of the network, or None
    if the architecture has no layer with a margin (e.g. architecture 2).

    warmup_epochs: epochs trained without the activation loss before measuring
    (0 = untrained model, NOT recommended, see _natural_distances)."""
    base_d, rows = _natural_distances(config, X, y, attack_mask, device,
                                      warmup_epochs=warmup_epochs,
                                      class_weights=class_weights)
    if verbose and rows:
        state = "untrained" if warmup_epochs <= 0 else f"after {warmup_epochs} warmup epochs"
        print(f"real/adv distance {state} (eps={config.eps}):")
        for r in rows:
            print(f"   {r['layer']:12s}  d={r['d']:.5f}   "
                  f"|activations|={r['scale']:.3f}   "
                  f"d/scale={r['d'] / (r['scale'] ** 2 + 1e-12):.5f}")
    if not base_d:
        if verbose:
            print("no FurtherAL layer: margins not applicable")
        return None
    margins = tuple(factor * d for d in base_d)
    if verbose:
        print(f"\nsuggested act_margin = {tuple(round(m, 5) for m in margins)}  "
              f"({factor:g}x the natural distance of each layer)")
    return margins


# ===========================================================================
# Automatic selection (unidimensional search over the shared factor)
# ===========================================================================
@dataclasses.dataclass
class MarginSearchResult:
    margins: tuple          # one margin per FurtherAL, in network order
    factor: float           # winning multiplier
    score: float            # validation score of the winner
    base_distances: tuple   # natural d_i used as the scale of each layer
    warmup_epochs: int      # warmup epochs used to measure base_distances
    candidates: list        # details of every factor tried


def select_margins(config, X_train, y_train, X_val, y_val, attack_mask=None,
                   device="cpu", factors=(1.0, 2.0, 5.0),
                   search_epochs=3, warmup_epochs=3, max_train=None,
                   probe_size=2048, class_weights=None,
                   verbose=True) -> MarginSearchResult:
    """Automatically selects a margin for every FurtherAL layer: starting from
    a set of propose factors, search the best factor such that the best margins
            margin_i = factor * d_i
    are the ones with higher scores

    Useful parameters:
      search_epochs : training epochs per candidate
      warmup_epochs : epochs trained without the activation loss before
                      measuring d_i (0 = untrained model, not recommended:
                      the layers distances would be unrepresentative)
      max_train     : subsample the train set to speed the search up
      probe_size    : samples used to measure the natural d_i
      class_weights : the same class weights used in the real training

    All candidates start from the same init and the same shuffle order (fixed
    seed), so they differ ONLY by their margins.
    """
    probe_n = min(probe_size, len(X_train))
    base_d, _ = _natural_distances(config, X_train[:probe_n],
                                   y_train[:probe_n], attack_mask, device,
                                   warmup_epochs=warmup_epochs,
                                   class_weights=class_weights)
    if not base_d:
        raise ValueError(
            "select_margins requires FurtherAL layers: use a "
            "DetectorArchConfig with use_act_loss=True"
        )

    if max_train is not None:
        Xtr, ytr = X_train[:max_train], y_train[:max_train]
    else:
        Xtr, ytr = X_train, y_train

    if verbose:
        print(f"select_margins: natural d = "
              f"{tuple(round(d, 5) for d in base_d)}  "
              f"(measured after {warmup_epochs} warmup epochs; "
              f"{search_epochs} epochs x {len(factors)} candidates, "
              f"{len(Xtr)} train samples)")

    candidates = []
    for f in factors:
        margins = tuple(f * d for d in base_d)
        cand = dataclasses.replace(config, act_margin=margins)
        torch.manual_seed(lc.SEED)   # same init and same shuffle for all
        built = lc.create_architecture(cand, X_train.shape[1], device=device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr),
            batch_size=cand.batch_size, shuffle=True,
        )
        for _ in range(search_epochs):
            lt.train_epoch(built.model, loader, built.optimizer, eps=cand.eps,
                           lambda_det=cand.lambda_det, lambda_act=cand.lambda_act,
                           task_loss_on_adv=cand.task_loss_on_adv,
                           class_weights=class_weights, attack_mask=attack_mask,
                           attack=cand.train_attack, threshold=cand.threshold,
                           attack_kwargs=cand.attack_kwargs(), device=device)
        val = le.evaluate(built.model, X_val, y_val, eps=cand.eps,
                          attack_mask=attack_mask, attack=cand.train_attack,
                          device=device, threshold=cand.threshold,
                          attack_kwargs=cand.attack_kwargs())
        det_bal = 0.5 * (val["det_clean_acc"] + val["det_adv_acc"])
        score = 0.5 * (val["task_clean"]["acc"] + det_bal)
        candidates.append({"factor": f, "margins": margins, "score": score,
                           "task_clean_acc": val["task_clean"]["acc"],
                           "det_bal_acc": det_bal})
        if verbose:
            print(f"   factor={f:6.1f}  margins={tuple(round(m, 5) for m in margins)}  "
                  f"task={val['task_clean']['acc']:.4f}  det bal={det_bal:.4f}  "
                  f"score={score:.4f}")

    best = max(candidates, key=lambda c: c["score"])
    if verbose:
        print(f"\nchosen factor = {best['factor']:g}  ->  "
              f"act_margin = {tuple(round(m, 5) for m in best['margins'])}")
    return MarginSearchResult(margins=best["margins"], factor=best["factor"],
                              score=best["score"], base_distances=base_d,
                              warmup_epochs=warmup_epochs,
                              candidates=candidates)