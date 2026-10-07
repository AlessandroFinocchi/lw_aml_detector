"""V&V suite - Validazione: meccanismi del metodo (M1-M6).

Works on the stage 1 CSV and checkpoints, with at least 3 seeds
(--min-seeds). Comparisons between configurations use S1, one-sided. The
experiments are the ones declared in lwad_stage1.table() (s1/base, s1/mech).

    M1  FurtherAL hinge active during training (measured on a replay)
    M2  FurtherAL separates in relative terms, not by inflating the scale
    M3  CloserAL aligns without collapsing the representation
    M4  the vulnerability of an undefended model grows with depth
    M5  per-layer detector profile in the isolated regime
    M6  the early exit saves real time on GPU
    M7  FurtherAL separates in a controlled experiment (no stage 1 data)

Run:  python -m libs.tests.vv_validate_mechanisms [--only M2]
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import math
import statistics
import sys
import time
from dataclasses import replace

import scipy.stats
import torch
from sklearn.metrics import roc_auc_score

import libs.model.lwad_wrapper as lw
import libs.model.lwad_config as lc
import libs.attacks.lwad_attack as la
import libs.training.lwad_margin as lm
import libs.evaluation.lwad_evaluator as le

TESTS: list = []

ACTIVE_MIN = 0.01
COLLAPSE_ACC_TOL = 0.02
SPREAD_RATIO_MIN = 0.25
AUC_MAX_MIN = 0.75
PROFILE_STD_MAX = 0.03

M_SAMPLES = 2000                        # clean test samples of M3 and M5
TIMED_SAMPLES, TIMED_REPEATS = 500, 30  # M6: per sample, median of the repeats
TIMED_WARMUP = 64

# --- M7: DetectorLayer against FurtherAL, trained here ----------------------
M7_SEEDS = (42, 43, 44)
M7_TRAIN_ROWS, M7_EPOCHS = 16384, 15
M7_CFG = lc.DetectorModelConfig(hidden_dims=(128, 64, 64), use_act_loss=False,
                                margin_factor=None)
# The margin must sit ABOVE the relative distance the model reaches on its
# own, otherwise the hinge is already at zero and FurtherAL pushes nothing.
# It is calibrated on the TRAINED baseline, not on the untrained model:
# training grows the natural distance by orders of magnitude.
M7_MARGIN_FACTOR = 5.0
# With the default weight the repulsion loses to the task/detector losses,
# because PGD regenerates the attack against the model at every batch and
# undoes the separation. It needs a heavier weight to actually drive.
M7_LAMBDA_ACT = 10.0


def _short(name: str) -> str:
    return name.split("/", 2)[-1]


def _mean(xs) -> float:
    return statistics.fmean(xs)


def _attack_like_eval(ck, X, y):
    """The checkpoint's evaluation attack at its training eps, seed fixed."""
    cfg = ck.config
    torch.manual_seed(vc.SEED)
    return la.generate_attack(ck.model, X, y, cfg.eps, cfg.eval_attack, mask=ck.attack_mask,
                              **cfg.attack_kwargs())


# ===========================================================================
# Tests
# ===========================================================================
@vc.vv_test(TESTS, "M1", "FurtherAL attivo", vc.VALIDATE,
            "La hinge di FurtherAL deve lavorare: quota di coppie con s < m, mediata sulle "
            f"epoche, >= {ACTIVE_MIN} e > 0 nella prima epoca (misurata su un replay).")
def m1(t:vc.Report, S:vc.Session):
    for name in ("s1/base/further", "s1/base/further-advtrain"):
        seeds = S.stage1.seeds(name)
        if not vc.enough_seeds(t, S, _short(name), seeds):
            continue
        for seed in seeds:
            rp = S.replay(name, seed)
            ep = rp["epochs"]
            per_layer = [_mean(col) for col in zip(*rp["layers"])]
            t.info(f"{_short(name)} seed {seed}: quota per epoca {[round(f, 4) for f in ep]}; "
                   f"media per FurtherAL {[round(f, 4) for f in per_layer]}")
            t.info({True: "   replay identico al checkpoint dello stadio 1",
                    False: "   nota: il replay non riproduce il checkpoint salvato "
                           "(stadio 1 eseguito senza determinismo?)",
                    None: "   nessun checkpoint dello stadio 1 con cui confrontare il replay"}
                   [rp["faithful"]])
            mean = _mean(ep)
            ok = mean >= ACTIVE_MIN and ep[0] > 0
            t.check(f"{_short(name)} seed {seed}: media >= {ACTIVE_MIN} e prima epoca > 0", ok,
                    f"media {mean:.4f}, prima epoca {ep[0]:.4f}" + ("" if ok else ": FurtherAL inerte"))


M2_PAIRS = (("s1/base/further", "s1/base/detlayer"),
            ("s1/mech/further-wide3", "s1/mech/detlayer-wide3"))


@vc.vv_test(TESTS, "M2", "Separazione reale e persistente", vc.VALIDATE,
            "Nei livelli FurtherAL slav_rel deve superare quella di DetectorLayer (S1); "
            "se cresce solo slav e' inflazione di scala.")
def m2(t:vc.Report, S:vc.Session):
    st = S.stage1
    for k, (fur, det) in enumerate(M2_PAIRS):
        label = f"{_short(fur)} vs {_short(det)}"
        seeds = st.paired_seeds(fur, det)
        if not vc.enough_seeds(t, S, label, seeds):
            continue
        for layer in vc.s1_experiment(fur).config().resolved_wrap_at():
            rel = vc.s1_test(st.column(fur, "slav_rel", seeds, layer),
                             st.column(det, "slav_rel", seeds, layer), "greater")
            raw = vc.s1_test(st.column(fur, "slav", seeds, layer),
                             st.column(det, "slav", seeds, layer), "greater")
            t.info(f"{label} livello {layer}: slav_rel {_mean(st.column(fur, 'slav_rel', seeds, layer)):.4g}"
                   f" vs {_mean(st.column(det, 'slav_rel', seeds, layer)):.4g}, slav "
                   f"{_mean(st.column(fur, 'slav', seeds, layer)):.4g} vs "
                   f"{_mean(st.column(det, 'slav', seeds, layer)):.4g}")
            what = f"{label} livello {layer}: slav_rel FurtherAL > DetectorLayer (S1)"
            if rel[0] in (vc.PASS, vc.INCONCLUSIVE):
                t.outcome(what, rel[0], rel[1])
            elif raw[0] == vc.PASS:
                t.outcome(what, vc.FAIL, f"{rel[1]}; cresce solo slav ({raw[1]}): inflazione di scala")
            else:
                t.outcome(what, vc.FAIL, f"{rel[1]}: nessuna separazione significativa")
        if k == 1:
            for layer in (1, 2):
                ratio = (_mean(st.column(fur, "slav_rel", seeds, layer))
                         / _mean(st.column(det, "slav_rel", seeds, layer)))
                t.info(f"{label} livello {layer} (non avvolto): slav_rel FUR/DET = {ratio:.3f} "
                       f"(solo report)")


@torch.no_grad()
def _dispersion(model, X) -> list[float]:
    """Per hidden layer: mean over the units of the variance across samples
    of a^(l), over the mean of a^(l)^2."""
    return [(a.var(dim=0, unbiased=False).mean() / a.pow(2).mean()).item()
            for a in le._hidden_preacts(model, X)]


@vc.vv_test(TESTS, "M3", "Allineamento CloserAL non degenere", vc.VALIDATE,
            "closer contro advtrain: nei livelli CloserAL scendono slav e slav_rel (S1), "
            "senza perdita di accuratezza ne' collasso della dispersione.")
def m3(t:vc.Report, S:vc.Session):
    st = S.stage1
    closer, adv = "s1/base/closer", "s1/base/advtrain"
    seeds = st.paired_seeds(closer, adv)
    if not vc.enough_seeds(t, S, "closer vs advtrain", seeds):
        return

    # 1) both the raw and the relative deviation go down
    for layer in vc.s1_experiment(closer).config().resolved_wrap_at():
        raw = vc.s1_test(st.column(closer, "slav", seeds, layer),
                         st.column(adv, "slav", seeds, layer), "less")
        rel = vc.s1_test(st.column(closer, "slav_rel", seeds, layer),
                         st.column(adv, "slav_rel", seeds, layer), "less")
        what = f"livello {layer}: slav e slav_rel di closer < advtrain (S1)"
        detail = f"slav {raw[1]}; slav_rel {rel[1]}"
        if raw[0] == rel[0] == vc.PASS:
            t.outcome(what, vc.PASS, detail)
        elif vc.INCONCLUSIVE in (raw[0], rel[0]):
            t.outcome(what, vc.INCONCLUSIVE, detail)
        elif raw[0] == vc.PASS:
            t.outcome(what, vc.FAIL, f"{detail}: scende solo slav, riduzione di scala")
        else:
            t.outcome(what, vc.FAIL, f"{detail}: slav non scende")

    # 2) no accuracy collapse
    acc_n = _mean(st.column(closer, "task_clean_acc", seeds))
    acc_a = _mean(st.column(adv, "task_clean_acc", seeds))
    t.check(f"task_clean_acc media di closer >= advtrain - {COLLAPSE_ACC_TOL}",
            acc_n >= acc_a - COLLAPSE_ACC_TOL, f"{acc_n:.4f} vs {acc_a:.4f}")

    # 3) no representation collapse
    X, _ = S.rows("test", M_SAMPLES)
    ck_seeds = [s for s in seeds if st.checkpoint(closer, s) and st.checkpoint(adv, s)]
    if not ck_seeds:
        t.inconclusive("dispersione per livello", "checkpoint mancanti")
        return
    disp = {n: [_dispersion(S.ckpt(n, s).model, X) for s in ck_seeds] for n in (closer, adv)}
    for layer in range(len(disp[closer][0])):
        dn = _mean(d[layer] for d in disp[closer])
        da = _mean(d[layer] for d in disp[adv])
        t.check(f"livello {layer}: dispersione closer / advtrain >= {SPREAD_RATIO_MIN}",
                dn / da >= SPREAD_RATIO_MIN, f"{dn:.4f} / {da:.4f} = {dn / da:.3f}")


@vc.vv_test(TESTS, "M4", "Amplificazione con la profondita'", vc.VALIDATE,
            "Modello non difeso a 4 livelli: slav_rel dell'ultimo livello supera quella del "
            "primo e almeno 2 dei 3 guadagni sqrt(slav_rel(l) / slav_rel(l-1)) sono > 1.")
def m4(t:vc.Report, S:vc.Session):
    name = "s1/mech/undefended-deep"
    seeds = S.stage1.seeds(name)
    if not vc.enough_seeds(t, S, _short(name), seeds):
        return
    for seed in seeds:
        r = S.stage1.row(name, seed)["slav_rel"]
        gains = [math.sqrt(r[i] / r[i - 1]) for i in range(1, len(r))]
        ok = r[-1] > r[0] and sum(g > 1 for g in gains) >= 2
        t.check(f"seed {seed}: slav_rel ultimo > primo e >= 2 guadagni > 1", ok,
                f"slav_rel {r}, guadagni {[round(g, 3) for g in gains]}")


@torch.no_grad()
def _detector_aurocs(ck, X, y, x_adv) -> list[float]:
    """AUROC of every single detector, clean (0) against adversarial (1)."""
    _, sc = ck.model(X)
    _, sa = ck.model(x_adv)
    labels = [0] * len(X) + [1] * len(x_adv)
    return [roc_auc_score(labels, torch.sigmoid(torch.cat([c, a])).squeeze(-1).cpu().numpy())
            for c, a in zip(sc.detections, sa.detections)]


@vc.vv_test(TESTS, "M5", "Profilo dei detector nel regime isolato", vc.VALIDATE,
            "DET a 4 livelli con un detector per livello: il miglior AUROC medio deve essere "
            f">= {AUC_MAX_MIN} e il profilo stabile tra i seed.")
def m5(t:vc.Report, S:vc.Session):
    name = "s1/mech/detlayer-deep"
    seeds = S.stage1.ckpt_seeds(name)
    if not vc.enough_seeds(t, S, _short(name), seeds):
        return
    X, y = S.rows("test", M_SAMPLES)
    aucs = {}
    for seed in seeds:
        ck = S.ckpt(name, seed)
        aucs[seed] = _detector_aurocs(ck, X, y, _attack_like_eval(ck, X, y))
        t.info(f"seed {seed}: AUROC per detector {[round(a, 4) for a in aucs[seed]]}")
    per_layer = list(zip(*aucs.values()))
    means = [_mean(col) for col in per_layer]
    t.info(f"profilo medio {[round(m, 4) for m in means]}")
    best = max(range(len(means)), key=means.__getitem__)
    t.check(f"max sui livelli dell'AUROC medio >= {AUC_MAX_MIN}", means[best] >= AUC_MAX_MIN,
            f"livello {best}: {means[best]:.4f}")
    for layer, col in enumerate(per_layer):
        if len(col) < 2:
            t.inconclusive(f"livello {layer}: std tra seed <= {PROFILE_STD_MAX}",
                           "serve piu' di un seed")
            continue
        sd = statistics.stdev(col)
        t.check(f"livello {layer}: std tra seed dell'AUROC <= {PROFILE_STD_MAX}",
                sd <= PROFILE_STD_MAX, f"{sd:.4f}")
    rho = scipy.stats.spearmanr(range(len(means)), means)
    t.info(f"Spearman AUROC medio vs profondita': rho={rho.statistic:.3f}, p={rho.pvalue:.4f}")


@torch.no_grad()
def _median_ms(variants: dict, x) -> dict:
    """Median over TIMED_REPEATS of each variant on one sample, CUDA
    synchronized on both sides of every measure. The variants run interleaved,
    in a rotating order, so a drift of the GPU clock hits all of them."""
    names = list(variants)
    times = {n: [] for n in names}
    for rep in range(TIMED_REPEATS):
        for n in names[rep % len(names):] + names[:rep % len(names)]:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            variants[n](x)
            torch.cuda.synchronize()
            times[n].append((time.perf_counter() - t0) * 1e3)
    return {n: statistics.median(v) for n, v in times.items()}


@vc.vv_test(TESTS, "M6", "Risparmio reale dell'early exit", vc.VALIDATE,
            f"FUR a 4 detector con early exit: sui campioni che escono al primo detector "
            f"l'early exit e' piu' veloce del forward completo (Wilcoxon), "
            f"{TIMED_SAMPLES} puliti + {TIMED_SAMPLES} adversarial.")
def m6(t:vc.Report, S:vc.Session):
    name = "s1/mech/further-exit-deep"
    seeds = S.stage1.ckpt_seeds(name)
    if not seeds:
        t.inconclusive(f"{_short(name)}: checkpoint dello stadio 1", f"{name} assente")
        return
    seed = vc.SEED if vc.SEED in seeds else seeds[0]
    ck = S.ckpt(name, seed)
    model, thr = ck.model, ck.threshold_det
    if not (ck.config.early_exit and ck.config.score_reduce == "max"):
        t.inconclusive("configurazione del checkpoint", "attesi early_exit=True e score_reduce='max'")
        return
    t.info(f"checkpoint seed {seed}, soglia {thr:.3f}")

    X, y = S.rows("test", TIMED_SAMPLES)
    x_adv = _attack_like_eval(ck, X, y)
    det_idx = [i for i, layer in enumerate(model.layers) if isinstance(layer, lw.DetectorLayer)]

    @torch.no_grad()
    def exit_detector(x):
        """Order number of the detector a sample exits at (None: no exit).
        state.exit_layer is an index in model.layers."""
        _, st = model(x, exit_threshold=thr)
        return None if st.exit_layer is None else det_idx.index(st.exit_layer)

    exits = {"clean": [exit_detector(X[i:i + 1]) for i in range(len(X))],
             "adv": [exit_detector(x_adv[i:i + 1]) for i in range(len(x_adv))]}

    # blocking: the exits counted here are the ones le.inference_time counts
    rate = sum(e is not None for e in exits["adv"]) / len(x_adv)
    ref = le.inference_time(model, x_adv, n_samples=len(x_adv), threshold_det=thr,
                            reduce="max", early_exit=True)["infer_exit_rate"]
    if not t.check("quota di adversarial che escono = infer_exit_rate di le.inference_time",
                   rate == ref, f"{rate:.4f} vs {ref:.4f}" + ("" if rate == ref else ": BLOCCANTE")):
        return

    variants = {"full": lambda x: le._infer(model, x, thr, "max", early_exit=False),
                "early": lambda x: le._infer(model, x, thr, "max", early_exit=True),
                "backbone": lambda x: vc.bare_forward(model, x).argmax(-1)}
    for fn in variants.values():
        for _ in range(TIMED_WARMUP):
            fn(X[:1])
    vc.progress(f"misura di {2 * TIMED_SAMPLES} campioni x {TIMED_REPEATS} ripetizioni "
                f"x {len(variants)} varianti")
    samples = []                             # (exit detector, medians)
    for data, ex in ((X, exits["clean"]), (x_adv, exits["adv"])):
        for i in range(len(data)):
            samples.append((ex[i], _median_ms(variants, data[i:i + 1])))

    # savings per exit detector
    t.info(f"{'uscita':16s} {'n':>5s} {'full ms':>9s} {'early ms':>9s} {'backbone ms':>12s} "
           f"{'risparmio':>10s}")
    for d in list(range(len(det_idx))) + [None]:
        group = [m for e, m in samples if e == d]
        if not group:
            continue
        med = {v: statistics.median(m[v] for m in group) for v in variants}
        saving = statistics.median(1 - m["early"] / m["full"] for m in group)
        label = f"detector {d} (l{det_idx[d]})" if d is not None else "nessuna uscita"
        t.info(f"{label:16s} {len(group):5d} {med['full']:9.4f} {med['early']:9.4f} "
               f"{med['backbone']:12.4f} {saving:10.1%}")

    first = [m for e, m in samples if e == 0]
    if not first:
        t.inconclusive("early exit piu' veloce al primo detector (Wilcoxon)",
                       "nessun campione esce al primo detector")
        return
    try:
        w = scipy.stats.wilcoxon([m["early"] for m in first], [m["full"] for m in first],
                                 alternative="less")
    except ValueError as e:
        t.inconclusive("early exit piu' veloce al primo detector (Wilcoxon)", str(e))
        return
    t.check("early exit piu' veloce del forward completo al primo detector "
            "(Wilcoxon unilaterale, p < 0.05)", w.pvalue < 0.05,
            f"n={len(first)}, p={w.pvalue:.3g}")


def _m7_run(S, cfg, X, y, Xt, yt, seed: int):
    """cfg trained at seed; per detector layer the real/adv distance d, the
    activation scale and rel = d / scale^2, the quantity the FurtherAL hinge
    acts on. The attack is regenerated against each model, so every model is
    measured against the attack it actually faces."""
    model = vc.train_model(cfg, S.bundle, X, y, M7_EPOCHS, seed=seed)
    mask, kw = S.bundle.attack_mask, cfg.attack_kwargs()
    torch.manual_seed(seed)
    x_adv = la.generate_attack(model, Xt, yt, cfg.eps, cfg.eval_attack, mask=mask, **kw)
    rows = lm._measure_layer_distances(model, Xt, x_adv)
    torch.manual_seed(seed)
    val = le.evaluate(model, Xt, yt, eps=cfg.eps, attack_mask=mask, attack=cfg.eval_attack,
                      threshold_det=cfg.threshold_det, attack_kwargs=kw, device=vc.DEVICE)
    return rows, val


@vc.vv_test(TESTS, "M7", "Separazione relativa in un esperimento controllato", vc.VALIDATE,
            "Senza dati dello stadio 1: DetectorLayer e FurtherAL con stessi seed, init e ordine "
            f"dei batch, margini {M7_MARGIN_FACTOR:g}x la d/scale raggiunta da DetectorLayer, "
            f"lambda_act={M7_LAMBDA_ACT:g}. FurtherAL deve alzare d/scale in ogni livello con "
            f"detector, per seed e su {len(M7_SEEDS)} seed (S1); d grezza solo report.")
def m7(t:vc.Report, S:vc.Session):
    X, y = S.rows("train", M7_TRAIN_ROWS)
    Xt, yt = S.rows("test", M_SAMPLES)
    nf, wrap = S.bundle.n_features, M7_CFG.resolved_wrap_at()
    fur_cfg = replace(M7_CFG, use_act_loss=True, lambda_act=M7_LAMBDA_ACT)

    # the comparison isolates the activation loss only if both start alike
    torch.manual_seed(vc.SEED)
    sd_p = M7_CFG.build_model(nf).state_dict()
    torch.manual_seed(vc.SEED)
    sd_f = fur_cfg.build_model(nf).state_dict()
    if not t.check("DetectorLayer e FurtherAL: stessa struttura e stessi pesi iniziali",
                   sd_p.keys() == sd_f.keys()
                   and all(torch.equal(sd_p[k], sd_f[k]) for k in sd_p),
                   f"{len(sd_p)} vs {len(sd_f)} tensori"):
        return

    rel = {"plain": [], "further": []}      # per seed, per detector layer
    for seed in M7_SEEDS:
        vc.progress(f"seed {seed}: DetectorLayer e FurtherAL, {M7_EPOCHS} epoche "
                    f"su {M7_TRAIN_ROWS} righe")
        rows_p, val_p = _m7_run(S, M7_CFG, X, y, Xt, yt, seed)
        margins = tuple(M7_MARGIN_FACTOR * r["rel"] for r in rows_p)
        rows_f, val_f = _m7_run(S, replace(fur_cfg, act_margin=margins), X, y, Xt, yt, seed)
        rel["plain"].append([r["rel"] for r in rows_p])
        rel["further"].append([r["rel"] for r in rows_f])

        t.info(f"seed {seed}: margini {tuple(round(m, 5) for m in margins)}")
        t.info(f"{'livello':>7s}  {'':14s}{'d':>10s}{'|act|':>9s}{'d/scale':>10s}")
        for layer, p, f in zip(wrap, rows_p, rows_f):
            t.info(f"{layer:>7d}  {'DetectorLayer':14s}{p['d']:10.5f}{p['scale']:9.3f}"
                   f"{p['rel']:10.5f}")
            t.info(f"{'':7s}  {'FurtherAL':14s}{f['d']:10.5f}{f['scale']:9.3f}{f['rel']:10.5f}")
        for name, v in (("DetectorLayer", val_p), ("FurtherAL", val_f)):
            t.info(f"{name:14s} task_clean acc {v['task_clean']['acc']:.4f}, detector bal acc "
                   f"{0.5 * (v['det_clean_acc'] + v['det_adv_acc']):.4f}")

        # only d/scale is asserted: the detector starts with a LayerNorm and
        # the scale-invariant loss leaves the scale free, so d can drop
        # while the separation grows, and a larger d can be pure inflation
        for layer, p, f in zip(wrap, rows_p, rows_f):
            t.check(f"seed {seed} livello {layer}: d/scale FurtherAL > DetectorLayer",
                    f["rel"] > p["rel"],
                    f"{p['rel']:.5f} -> {f['rel']:.5f} ({f['rel'] / (p['rel'] + 1e-12):.2f}x), "
                    f"d grezza {p['d']:.5f} -> {f['d']:.5f}")

    for i, layer in enumerate(wrap):
        outcome, detail = vc.s1_test([r[i] for r in rel["further"]],
                                     [r[i] for r in rel["plain"]], "greater")
        t.outcome(f"livello {layer}: d/scale FurtherAL > DetectorLayer (S1)", outcome, detail)


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
