"""Smoke test: early exit under the "max" reduce gives the full forward decision.

    max_i p_i > t  <=>  some p_i > t

so stopping at the first detector over the threshold must flag exactly the
samples the full forward flags, and leave the label of the others unchanged.
Checked one sample at a time (as inference_time runs) on clean and
adversarial data, plus the batch rule: a batch exits only when every sample
in it is flagged.

Run:  python -m libs.tests.smoke_early_exit
"""
import torch

import libs.model.lwad_config as lc
import libs.training.lwad_trainer as lt
import libs.evaluation.lwad_evaluator as le
from libs.attacks.lwad_attack import generate_attack

torch.manual_seed(0)
N, F_DIM, EPOCHS = 512, 20, 5
X = torch.randn(N, F_DIM)
y = (X[:, 0] + X[:, 1] > 0).long()
mask = torch.ones(F_DIM)


def train(cfg):
    torch.manual_seed(lc.SEED)
    built = lc.create_model(cfg, F_DIM)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=True)
    for _ in range(EPOCHS):
        lt.train_epoch(built.model, loader, built.optimizer, eps=cfg.eps,
                       lambda_det=cfg.lambda_det, lambda_act=cfg.lambda_act,
                       task_loss_on_adv=cfg.task_loss_on_adv, attack_mask=mask,
                       attack=cfg.train_attack, threshold_det=cfg.threshold_det,
                       attack_kwargs=cfg.attack_kwargs(), reduce=cfg.score_reduce)
    built.model.eval()
    return built.model


def check_equivalence(model, X_eval, t, name):
    exits = 0
    for i in range(len(X_eval)):
        x = X_eval[i:i + 1]
        lab_f, sc_f, flag_f = le.predict(model, x, threshold_det=t, reduce="max")
        lab_e, sc_e, flag_e = le.predict(model, x, threshold_det=t, reduce="max",
                                         early_exit=True)
        assert bool(flag_f) == bool(flag_e), f"{name}[{i}]: flag differs"
        if lab_e is None:
            exits += 1
            assert bool(flag_f), f"{name}[{i}]: exited but not flagged"
            # partial max is a lower bound of the full one, still over t
            assert float(sc_e) <= float(sc_f) + 1e-6
        else:
            assert torch.equal(lab_f, lab_e), f"{name}[{i}]: label differs"
            assert torch.allclose(sc_f, sc_e), f"{name}[{i}]: score differs"
    print(f"   {name:5s}: {len(X_eval)} samples, {exits} early exits, flags identical")
    return exits


def main():
    print("== 1) config: early_exit requires score_reduce='max' ==")
    try:
        lc.DetectorModelConfig(early_exit=True, score_reduce="mean")
        raise AssertionError("validation did NOT fire")
    except ValueError as e:
        print("   OK ->", str(e)[:60], "...")
    assert lc.AdvTrainingModelConfig().early_exit is False

    cfg = lc.DetectorModelConfig(hidden_dims=(64, 32, 16), wrap_at=(0, 1, 2),
                                 score_reduce="max", early_exit=True,
                                 margin_factor=None, epochs=EPOCHS)
    model = train(cfg)

    print("\n== 2) predict: early_exit rejected with reduce='mean' ==")
    try:
        le.predict(model, X[:1], reduce="mean", early_exit=True)
        raise AssertionError("validation did NOT fire")
    except ValueError as e:
        print("   OK ->", e)

    print("\n== 3) per-sample equivalence with the full forward ==")
    # threshold from the validation procedure, as run_experiment does
    t, _ = lt.select_threshold(model, X, y, eps=cfg.eps, attack_mask=mask,
                               attack=cfg.train_attack,
                               attack_kwargs=cfg.attack_kwargs(), reduce="max")
    print(f"   threshold = {t:.3f}")
    x_adv = generate_attack(model, X, y, cfg.eps, cfg.eval_attack, mask=mask,
                            **cfg.attack_kwargs())
    check_equivalence(model, X, t, "clean")
    exits_adv = check_equivalence(model, x_adv, t, "adv")
    assert exits_adv > 0, "no adversarial sample exited early, test is vacuous"

    print("\n== 4) batch exits only if every sample is flagged ==")
    _, _, flags = le.predict(model, x_adv, threshold_det=t, reduce="max")
    flagged, clean = x_adv[flags], x_adv[~flags]
    lab, _, _ = le.predict(model, flagged, threshold_det=t, reduce="max", early_exit=True)
    assert lab is None, "an all-flagged batch did not exit"
    if len(clean):
        mixed = torch.cat([flagged[:4], clean[:1]])
        lab, _, _ = le.predict(model, mixed, threshold_det=t, reduce="max",
                               early_exit=True)
        assert lab is not None and len(lab) == len(mixed), "a mixed batch exited"
    print(f"   OK: {len(flagged)}-sample flagged batch exited, mixed batch did not")

    print("\n== 5) evaluate unchanged: it never exits early ==")
    m = le.evaluate(model, X, y, eps=cfg.eps, attack_mask=mask, attack=cfg.eval_attack,
                    threshold_det=t, attack_kwargs=cfg.attack_kwargs(), reduce="max")
    print(f"   clean_acc_e2e={m['clean_acc_e2e']:.4f}  "
          f"robust_acc_e2e={m['robust_acc_e2e']:.4f}")

    print("\n== 6) latency, one sample at a time ==")
    for name, data in (("clean", X), ("adv", x_adv)):
        full = le.inference_time(model, data, threshold_det=t, reduce="max")
        early = le.inference_time(model, data, threshold_det=t, reduce="max",
                                  early_exit=True)
        print(f"   {name:5s}: full {full['infer_ms']:.4f} ms | early exit "
              f"{early['infer_ms']:.4f} ms ({early['infer_exit_rate']:.1%} exited)")

    print("\nTEST PASSED")


if __name__ == "__main__":
    main()
