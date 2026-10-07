"""V&V suite - Verifica: coerenza di sistema (V4.1-V4.6).

Every test runs a mini-campaign on UNSW-NB15 through lx.run_suite (device
"cuda", margin cache cleared before every suite) and, where the spec says so,
repeats the comparison on the stage 1 results ("reale"). A stage 1
experiment missing from the CSV turns its part into a SKIP.

One mini-campaign run takes a couple of minutes: the whole section ~40 min.

Run:  python -m libs.tests.vv_verify_system [--only V4.4]
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import statistics
import sys

import libs.evaluation.lwad_evaluator as le
import libs.experiments.lwad_experiments as lx

TESTS: list = []

MARGIN_TINY_ACTIVE_MAX = 0.01   # V4.2: above it the hinge is not saturated at zero
RELOAD_TOL = 0.005              # V4.4, V4.6: a load_from run evaluates without reseeding,
                                # so the random start of its PGD differs


# ===========================================================================
# Comparison helpers
# ===========================================================================
def compare_runs(t, label, a: lx.RunResult, b: lx.RunResult,
                 exclude=vc.LATENCY_KEYS, training=True) -> None:
    """Identical metrics, threshold and checkpoints; with training also the
    epochs run and the best validation score."""
    d = vc.diff_metrics(a.metrics, b.metrics, exclude)
    t.check(f"{label}: metriche identiche", not d, vc.summarize(d))
    if training:
        t.check(f"{label}: epochs_ran identico", a.epochs_ran == b.epochs_ran,
                f"{a.epochs_ran} vs {b.epochs_ran}")
        t.check(f"{label}: best_val_score identico", a.best_val_score == b.best_val_score,
                f"{a.best_val_score!r} vs {b.best_val_score!r}")
    t.check(f"{label}: soglia identica", vc.same_value(a.threshold_det, b.threshold_det),
            f"{a.threshold_det!r} vs {b.threshold_det!r}")
    d = vc.diff_checkpoints(a.checkpoint, b.checkpoint)
    t.check(f"{label}: checkpoint identici", not d, vc.summarize(d))


def check_close(t, label: str, diff, *args, **kwargs) -> None:
    """Metrics equal within RELOAD_TOL, diff being vc.diff_metrics or
    vc.diff_rows; the detail still counts the values not bit-identical."""
    over, exact = diff(*args, **kwargs, tol=RELOAD_TOL), diff(*args, **kwargs)
    detail = (vc.summarize(over) if over
              else f"{len(exact)} valori non identici, tutti entro la tolleranza" if exact
              else None)
    t.check(f"{label} entro {RELOAD_TOL:g}", not over, detail)


def compare_stage1(t, S, name_a: str, name_b: str, exclude=(), act_margin=False) -> list[int]:
    """The same identities on the stage 1 runs, seed by seed. Returns the
    seeds compared (empty when one of the two experiments is missing)."""
    st = S.stage1
    missing = [n for n in (name_a, name_b) if not st.seeds(n)]
    if missing:
        t.skip(f"reale: {name_a} contro {name_b}",
               f"assente nello stadio 1: {', '.join(missing)}")
        return []
    seeds = st.paired_seeds(name_a, name_b)
    if not seeds:
        t.skip(f"reale: {name_a} contro {name_b}", "nessun seed in comune")
    for seed in seeds:
        d = vc.diff_rows(st.row(name_a, seed), st.row(name_b, seed), exclude)
        t.check(f"reale seed {seed}: metriche CSV identiche", not d, vc.summarize(d))
        pa, pb = st.checkpoint(name_a, seed), st.checkpoint(name_b, seed)
        if not (pa and pb):
            t.skip(f"reale seed {seed}: checkpoint identici", "file di checkpoint mancante")
            continue
        d = vc.diff_checkpoints(pa, pb)
        t.check(f"reale seed {seed}: checkpoint identici", not d, vc.summarize(d))
        if act_margin:
            ma = vc.raw_checkpoint(pa)["config"]["act_margin"]
            mb = vc.raw_checkpoint(pb)["config"]["act_margin"]
            t.check(f"reale seed {seed}: act_margin identico", ma == mb, f"{ma} vs {mb}")
    return seeds


# ===========================================================================
# Tests
# ===========================================================================
@vc.vv_test(TESTS, "V4.1", "FurtherAL con lambda_act = 0 coincide con DetectorLayer", vc.VERIFY,
            "Una activation loss di peso nullo non deve cambiare nulla: DET_S e FUR_S "
            "(lambda_act=0, margin_factor=None) identici al bit.")
def v4_1(t:vc.Report, S:vc.Session):
    det = S.reference_run("det")
    fur = S.run_suite([lx.Exp("vv/fur-lambda0", vc.FUR_S, lambda_act=0.0,
                              margin_factor=None)], "v41").results[0]
    compare_runs(t, "DET_S vs FUR_S(lambda_act=0)", det, fur)
    compare_stage1(t, S, "s1/vv/actloss-zero", "s1/base/detlayer")


@vc.vv_test(TESTS, "V4.2", "FurtherAL con kappa << 1 vicino a DetectorLayer", vc.VERIFY,
            "Solo reale: s1/vv/margin-tiny contro s1/base/detlayer; identici, oppure "
            "equivalenti entro il rumore (S2) se la hinge resta spenta.")
def v4_2(t:vc.Report, S:vc.Session):
    st = S.stage1
    tiny, ref = "s1/vv/margin-tiny", "s1/base/detlayer"
    seeds = st.paired_seeds(tiny, ref)
    if len(seeds) < vc.S1_MIN_SEEDS:
        t.inconclusive("almeno 3 seed appaiati nello stadio 1",
                       f"{tiny}: {st.seeds(tiny)}, {ref}: {st.seeds(ref)}")
        return

    # 1) identical checkpoints settle it
    diffs = {s: vc.diff_checkpoints(st.checkpoint(tiny, s), st.checkpoint(ref, s))
             if st.checkpoint(tiny, s) and st.checkpoint(ref, s) else ["file mancante"]
             for s in seeds}
    if not any(diffs.values()):
        t.outcome(f"checkpoint identici in tutti i {len(seeds)} seed", vc.PASS)
        return
    t.info("checkpoint non identici in " + ", ".join(f"seed {s}" for s, d in diffs.items() if d)
           + ": si misura l'attivita' della hinge")

    # 2) a hinge still active means the margin is not tiny enough to compare
    active = {}
    for seed in seeds:
        rp = S.replay(tiny, seed)
        active[seed] = statistics.fmean(rp["epochs"])
        t.info(f"seed {seed}: quota coppie con s < m per epoca "
               f"{[round(f, 4) for f in rp['epochs']]}, media {active[seed]:.4f} "
               f"(replay identico al checkpoint: {rp['faithful']})")
    over = {s: f for s, f in active.items() if f > MARGIN_TINY_ACTIVE_MAX}
    if over:
        t.inconclusive(f"quota coppie con s < m <= {MARGIN_TINY_ACTIVE_MAX} in ogni seed",
                       ", ".join(f"seed {s}: {f:.4f}" for s, f in over.items()))
        return

    # 3) equivalence within the seed noise
    for metric in vc.KEY_METRICS:
        ok, detail = vc.s2_equivalent(st.column(tiny, metric, seeds), st.column(ref, metric, seeds))
        t.check(f"S2 su {metric}", ok, detail)


@vc.vv_test(TESTS, "V4.3", "Riproducibilita' con cache fredda e calda", vc.VERIFY,
            "Due run identiche di FUR_S su un'architettura nuova: la seconda usa i margini "
            "in cache e deve coincidere con la prima.")
def v4_3(t:vc.Report, S:vc.Session):
    arch = dict(hidden_dims=(32, 8), wrap_at=(1,))
    res = S.run_suite([lx.Exp("vv/repro-cold", vc.FUR_S, **arch),
                       lx.Exp("vv/repro-warm", vc.FUR_S, **arch)], "v43")
    keys = len(lx._MARGIN_CACHE)
    t.check("a fine suite la cache dei margini ha una sola chiave", keys == 1, f"{keys} chiavi")
    cold, warm = res.results
    compare_runs(t, "cold vs warm", cold, warm)
    t.check("cold vs warm: act_margin identico", cold.config.act_margin == warm.config.act_margin,
            f"{cold.config.act_margin} vs {warm.config.act_margin}")
    compare_stage1(t, S, "s1/vv/repro-cold", "s1/vv/repro-warm", act_margin=True)


@vc.vv_test(TESTS, "V4.4", "Seed e ordine di esecuzione", vc.VERIFY,
            "Solo mini-campagna: il seed cambia il risultato, l'ordine delle run nella suite no "
            f"(anche per una run con load_from, che non reimposta il seed dell'attacco: "
            f"metriche entro {RELOAD_TOL:g}, relativa per slav e slav_rel).")
def v4_4(t:vc.Report, S:vc.Session):
    # 1) the seed matters
    res = S.run_suite([lx.Exp("vv/seed", vc.FUR_S)], "v44-seed", seeds=(42, 43))
    r42, r43 = sorted(res.results, key=lambda r: r.seed)
    km42, km43 = vc.key_metrics(r42), vc.key_metrics(r43)
    differ = [k for k in vc.KEY_METRICS if not vc.same_value(km42[k], km43[k])]
    t.check("FUR_S seed 42 vs 43: almeno una KEY_METRIC differisce", bool(differ),
            f"{len(differ)}/{len(vc.KEY_METRICS)} differiscono: {', '.join(differ)}")
    d = vc.diff_checkpoints(r42.checkpoint, r43.checkpoint)
    t.check("FUR_S seed 42 vs 43: checkpoint diversi", bool(d), f"{len(d)} differenze")

    # 2) the order of the runs does not
    A = lx.Exp("vv/A", vc.FUR_S)
    B = lx.Exp("vv/B", vc.FUR_S, lambda_act=2.0)
    C = lx.Exp("vv/C", vc.DET_S)
    fwd = {r.experiment: r for r in S.run_suite([A, B, C], "v44-ABC").results}
    bwd = {r.experiment: r for r in S.run_suite([C, B, A], "v44-CBA").results}
    for name in ("vv/A", "vv/B", "vv/C"):
        d = vc.diff_metrics(fwd[name].metrics, bwd[name].metrics, vc.LATENCY_KEYS)
        t.check(f"[A, B, C] vs [C, B, A]: metriche di {name[3:]} identiche", not d, vc.summarize(d))
        d = vc.diff_checkpoints(fwd[name].checkpoint, bwd[name].checkpoint)
        t.check(f"[A, B, C] vs [C, B, A]: checkpoint di {name[3:]} identici", not d, vc.summarize(d))

    # 3) nor does it for a run that only evaluates a stored checkpoint
    L = lx.Exp("vv/L", vc.DET_S, load_from=S.reference_run("det").checkpoint)
    cl = {r.experiment: r for r in S.run_suite([C, L], "v44-CL").results}
    lc_ = {r.experiment: r for r in S.run_suite([L, C], "v44-LC").results}
    check_close(t, "[C, L] vs [L, C]: metriche di L (load_from) uguali", vc.diff_metrics,
                cl["vv/L"].metrics, lc_["vv/L"].metrics, vc.LATENCY_KEYS)


@vc.vv_test(TESTS, "V4.5", "L'early exit cambia solo la latenza", vc.VERIFY,
            "FUR_S con score_reduce='max', con e senza early_exit: stesse metriche e checkpoint, "
            "solo i tassi di uscita cambiano.")
def v4_5(t:vc.Report, S:vc.Session):
    exclude = vc.LATENCY_KEYS + vc.EXIT_RATE_KEYS
    off, on = S.run_suite([lx.Exp("vv/exit-off", vc.FUR_S, score_reduce="max"),
                           lx.Exp("vv/exit-on", vc.FUR_S, score_reduce="max", early_exit=True)],
                          "v45", infer=64).results
    compare_runs(t, "exit-off vs exit-on", off, on, exclude=exclude)
    rates_off = [off.metrics[k] for k in vc.EXIT_RATE_KEYS]
    t.check("senza early exit i tassi di uscita sono 0", rates_off == [0, 0], f"{rates_off}")
    t.check("con early exit infer_exit_rate_adv > 0", on.metrics["infer_exit_rate_adv"] > 0,
            f"{on.metrics['infer_exit_rate_adv']:.2%} (clean {on.metrics['infer_exit_rate']:.2%})")

    seeds = compare_stage1(t, S, "s1/vv/exit-off", "s1/vv/exit-on", exclude=vc.EXIT_RATE_KEYS)
    st = S.stage1
    for seed in seeds:
        r_off, r_on = st.row("s1/vv/exit-off", seed), st.row("s1/vv/exit-on", seed)
        rates_off = [r_off[k] for k in vc.EXIT_RATE_KEYS]
        t.check(f"reale seed {seed}: senza early exit i tassi di uscita sono 0",
                rates_off == [0, 0], f"{rates_off}")
        t.check(f"reale seed {seed}: con early exit infer_exit_rate_adv > 0",
                (r_on["infer_exit_rate_adv"] or 0) > 0, f"{r_on['infer_exit_rate_adv']}")


@vc.vv_test(TESTS, "V4.6", "Un checkpoint ricaricato riproduce la run", vc.VERIFY,
            "Rivalutare con load_from e lo stesso seed un modello addestrato deve dare le "
            f"stesse metriche entro {RELOAD_TOL:g} (relativa per slav e slav_rel: la "
            "valutazione non reimposta il seed dell'attacco) e per i detector la stessa soglia.")
def v4_6(t:vc.Report, S:vc.Session):
    det, adv = S.reference_run("det"), S.reference_run("adv")
    rd, ra = S.run_suite([lx.Exp("vv/reload-det", vc.DET_S, load_from=det.checkpoint),
                          lx.Exp("vv/reload-adv", vc.ADV_S, load_from=adv.checkpoint)],
                         "v46").results
    for label, a, b in (("DET_S", det, rd), ("ADV_S", adv, ra)):
        check_close(t, f"{label} addestrato vs ricaricato: metriche uguali", vc.diff_metrics,
                    a.metrics, b.metrics, vc.LATENCY_KEYS)
    t.check("DET_S addestrato vs ricaricato: soglia identica",
            rd.threshold_det == det.threshold_det, f"{det.threshold_det!r} vs {rd.threshold_det!r}")

    # stage 1: the checkpoints of seed 42, timed as the campaign does
    st = S.stage1
    for name in ("s1/base/detlayer", "s1/base/closer"):
        path = st.checkpoint(name, vc.SEED)
        if not path:
            t.skip(f"reale: {name} seed {vc.SEED}", "checkpoint assente nello stadio 1")
            continue
        exp = vc.s1_experiment(name)
        reload_exp = lx.Exp(f"vv/reload-{name.split('/')[-1]}", exp.base,
                            **dict(exp.overrides, load_from=path))
        res = S.run_suite([reload_exp], f"v46-{name.split('/')[-1]}",
                          infer=le.DEFAULT_TIMED_INFER_SAMPLES).results[0]
        row, saved = vc.normalize_row(res.row()), st.row(name, vc.SEED)
        check_close(t, f"reale {name} seed {vc.SEED}: metriche uguali dopo load_from",
                    vc.diff_rows, saved, row, exclude=("epochs", "val_score"))
        if exp.config().uses_detectors:
            t.check(f"reale {name} seed {vc.SEED}: soglia identica",
                    vc.same_value(row["threshold_det"], saved["threshold_det"]),
                    f"{saved['threshold_det']!r} vs {row['threshold_det']!r}")


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
