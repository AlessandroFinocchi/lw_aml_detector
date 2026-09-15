"""Smoke test: does FurtherAL actually push real and adversarial activations apart?

Compares two architectures that are IDENTICAL except for the activation loss:

    A) DetectorArchConfig(use_act_loss=False)  -> plain DetectorLayer
    B) DetectorArchConfig(use_act_loss=True)   -> FurtherAL (repulsive hinge)

Same seed, same init, same batch order, same attack settings, so any
difference in activation distance comes from the loss alone.

Reported per detector-bearing layer:
    d        mean squared distance between real and adv activations
    |act|    activation magnitude (RMS)
    d/scale  distance normalized by the activation scale

The normalized column matters: a model could raise d simply by inflating all
its activations, which separates nothing in relative terms and gives the
detector no extra signal. A real improvement raises d/scale too.

Run:  python smoke_test_further.py
"""
import torch

import libs.preprocess.unsw_bw15 as pp
import libs.lwad_wrapper as lw
import libs.lwad_config as lc
import libs.lwad_trainer as lt
import libs.lwad_evaluator as le
from libs.lwad_attack import generate_attack

DATASET_PATH = "dataset/unsw-nb15/"
EPOCHS = 15
HIDDEN_DIMS = (128, 64, 64)
N_TRAIN, N_TEST, N_FEATURES = 6000, 2000, 42

# The margin must sit ABOVE the distance the model reaches on its own,
# otherwise the hinge is already at zero and FurtherAL pushes nothing.
# It is calibrated on the TRAINED baseline, not on the untrained model:
# training grows the natural distance by orders of magnitude.
MARGIN_FACTOR = 5.0

# With the default weight the repulsion loses to the task/detector losses,
# because PGD regenerates the attack against the model at every batch and
# undoes the separation. It needs a heavier weight to actually drive.
LAMBDA_ACT = 10.0


# ===========================================================================
# Measurement at every detector-bearing layer (DetectorLayer and FurtherAL,
# which is a subclass), so both architectures produce comparable rows
# ===========================================================================
@torch.no_grad()
def layer_distances(model, x, x_adv):
    rows = []
    h_real, h_adv = x, x_adv
    for layer in model.layers:
        h_real = layer.base(h_real)
        h_adv = layer.base(h_adv)
        if isinstance(layer, lw.DetectorLayer):
            d = (h_adv - h_real).pow(2).mean(dim=-1).mean().item()
            scale = h_real.pow(2).mean().item() ** 0.5
            rows.append({"layer": type(layer).__name__, "d": d, "scale": scale,
                         "rel": d / (scale ** 2 + 1e-12),
                         "margin": getattr(layer, "margin", None)})
    return rows


def train_and_measure(cfg, X_tr, y_tr, X_te, y_te, mask):
    """Trains a config from a fixed seed and measures activation distances.
    The attack is regenerated against each model, so every model is measured
    against the attack it actually faces."""
    torch.manual_seed(lc.SEED)
    built = lc.create_architecture(cfg, X_tr.shape[1])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=cfg.batch_size, shuffle=True,
    )
    for _ in range(EPOCHS):
        lt.train_epoch(built.model, loader, built.optimizer, eps=cfg.eps,
                       lambda_det=cfg.lambda_det, lambda_act=cfg.lambda_act,
                       task_loss_on_adv=cfg.task_loss_on_adv, attack_mask=mask,
                       attack=cfg.train_attack, threshold_det=cfg.threshold_det,
                       attack_kwargs=cfg.attack_kwargs())

    built.model.eval()
    x_adv = generate_attack(built.model, X_te, y_te, cfg.eps, cfg.eval_attack,
                            mask=mask, **cfg.attack_kwargs())
    rows = layer_distances(built.model, X_te, x_adv)
    val = le.evaluate(built.model, X_te, y_te, eps=cfg.eps, attack_mask=mask,
                      attack=cfg.eval_attack, threshold_det=cfg.threshold_det,
                      attack_kwargs=cfg.attack_kwargs())
    return built.model, rows, val


def main():
    X_tr, y_tr, _, _, X_te, y_te, _, _ = pp.load_unsw(DATASET_PATH, False, True)

    mask = torch.ones(N_FEATURES)


    # --- fairness check: identical structure and initial weights ----------
    cfg_plain = lc.DetectorArchConfig(hidden_dims=HIDDEN_DIMS, use_act_loss=False)
    probe = lc.DetectorArchConfig(hidden_dims=HIDDEN_DIMS, use_act_loss=True)

    torch.manual_seed(lc.SEED); m_a = cfg_plain.build_model(N_FEATURES)
    torch.manual_seed(lc.SEED); m_b = probe.build_model(N_FEATURES)
    
    sd_a, sd_b = m_a.state_dict(), m_b.state_dict()
    assert sd_a.keys() == sd_b.keys(), "the two models differ in structure"
    assert all(torch.equal(sd_a[k], sd_b[k]) for k in sd_a), \
        "the two models start from different weights, comparison unfair"
    print(f"fairness: same structure, same initial weights ({len(sd_a)} tensors)\n")

    # --- 1) baseline: detectors, no activation loss -----------------------
    _, rows_plain, val_plain = train_and_measure(cfg_plain, X_tr, y_tr, X_te, y_te, mask)

    # --- 2) margins calibrated on what the baseline actually reaches ------
    margins = tuple(MARGIN_FACTOR * r["d"] for r in rows_plain)
    distances = tuple(f"{r['d']:.5f}" for r in rows_plain)
    print(f"baseline distances : {distances}")
    print(f"margins ({MARGIN_FACTOR:g}x)      : {tuple(f'{m:.5f}' for m in margins)}")
    print(f"lambda_act         : {LAMBDA_ACT:g}\n")

    cfg_further = lc.DetectorArchConfig(hidden_dims=HIDDEN_DIMS, use_act_loss=True,
                                        lambda_act=LAMBDA_ACT, act_margin=margins)
    _, rows_further, val_further = train_and_measure(cfg_further, X_tr, y_tr, X_te, y_te, mask)

    # --- report -----------------------------------------------------------
    print(f"activation distances after {EPOCHS} epochs "
          f"({N_TEST} test samples, eps={cfg_plain.eps})\n")
    header = (f"{'layer':>5s}  {'':14s}{'d':>11s}{'|act|':>9s}{'d/scale':>10s}{'margin':>10s}")
    print(header); print("-" * len(header))
    for i, (p, f) in enumerate(zip(rows_plain, rows_further)):
        print(f"{i:>5d}  {'DetectorLayer':14s}{p['d']:11.5f}{p['scale']:9.3f}"
              f"{p['rel']:10.5f}{'-':>10s}")
        print(f"{'':5s}  {'FurtherAL':14s}{f['d']:11.5f}{f['scale']:9.3f}"
              f"{f['rel']:10.5f}{f['margin']:10.5f}")
        print(f"{'':5s}  {'-> ratio':14s}{f['d'] / (p['d'] + 1e-12):10.2f}x"
              f"{f['scale'] / (p['scale'] + 1e-12):8.2f}x"
              f"{f['rel'] / (p['rel'] + 1e-12):9.2f}x\n")

    def det_bal(v):
        return 0.5 * (v["det_clean_acc"] + v["det_adv_acc"])
    print(f"{'':22s}{'task clean acc':>16s}{'detector bal acc':>18s}")
    print(f"{'DetectorLayer':22s}{val_plain['task_clean']['acc']:16.4f}{det_bal(val_plain):18.4f}")
    print(f"{'FurtherAL':22s}{val_further['task_clean']['acc']:16.4f}{det_bal(val_further):18.4f}")

    # --- assertions -------------------------------------------------------
    print()
    for i, (p, f) in enumerate(zip(rows_plain, rows_further)):
        assert f["d"] > p["d"], (
            f"layer {i}: FurtherAL did NOT increase the distance "
            f"({f['d']:.6f} <= {p['d']:.6f}). Raise LAMBDA_ACT or MARGIN_FACTOR: "
            f"with a low weight the adaptive attack cancels the repulsion"
        )
        print(f"   layer {i}: distance increased  {p['d']:.5f} -> {f['d']:.5f}  "
              f"({f['d'] / (p['d'] + 1e-12):.2f}x)")

    mean_plain = sum(r["d"] for r in rows_plain) / len(rows_plain)
    mean_further = sum(r["d"] for r in rows_further) / len(rows_further)
    print(f"\nmean distance over layers: DetectorLayer={mean_plain:.5f}  "
          f"FurtherAL={mean_further:.5f}  ({mean_further / mean_plain:.2f}x)")
    print("\nTEST PASSED")
    return {"plain": rows_plain, "further": rows_further, "margins": margins,
            "mean_plain": mean_plain, "mean_further": mean_further}


if __name__ == "__main__":
    main()