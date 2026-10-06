"""V&V suite - Validazione: credibilita' della robustezza (A3, A1, A2, A4).

The stage 1 checkpoints are re-evaluated with their saved threshold on a
fixed subset of 2000 test samples. The seed is fixed before every
evaluation and alpha = eps/4 is recomputed for every eps.

    m_task = task_adv.acc     (adversarial accuracy of the classifier)
    m_e2e  = robust_acc_e2e   (adversarial sample classified right OR flagged)

A3 runs first: it checks the adaptive attack and picks beta*, the evade
weight that A1, A2 and A4 use against the detector models.

Run:  python -m libs.tests.vv_validate_robustness [--only A3]
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import sys
from dataclasses import dataclass
from typing import Optional

import torch

import libs.model.lwad_config as lc
import libs.attacks.lwad_attack as la
import libs.evaluation.lwad_evaluator as le

TESTS: list = []

TOL_A = 0.005
EPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
BETA_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
EPS_LARGE = (2.0, 5.0, 10.0)
UNBOUNDED_MAX = 0.05
CONV_SETTINGS = ((20, 1), (50, 1), (100, 1), (100, 5))    # (steps, restarts)
TOL_CONV = 0.01

A_SAMPLES = 2000
ITER_STEPS = 20             # A1: steps of the iterative adaptive attack
UNBOUNDED_EPS, UNBOUNDED_STEPS = 10.0, 50

NO_DET = ("undefended", "advtrain", "nearest", "nearest-clean")
WITH_DET = ("detlayer", "detlayer-advtrain", "further", "further-advtrain")


# ===========================================================================
# Evaluation of one attack on a stage 1 checkpoint
# ===========================================================================
@dataclass
class Robustness:
    m_task: float
    m_e2e: float
    det_adv_acc: Optional[float]     # None for the models without detectors


def robust_eval(ck, X, y, attack: str, eps: float, *, mask, steps: Optional[int] = None,
                alpha: Optional[float] = None, beta: Optional[float] = None,
                restarts: int = 1) -> Robustness:
    """m_task and m_e2e of one attack, with the checkpoint's saved threshold
    and score reduction. Same definitions as le.evaluate, computed per sample
    so that restarts can be combined: with restarts > 1 the attack succeeds
    on a sample as soon as one restart does (OR over the restarts)."""
    model, cfg = ck.model, ck.config
    thr = ck.threshold_det if ck.threshold_det is not None else lc.DEFAULT_THRESHOLD_DET
    steps = cfg.pgd_steps if steps is None else steps
    alpha = eps / 4 if alpha is None else alpha
    torch.manual_seed(vc.SEED)
    robust_task = robust_e2e = None
    det_adv_acc = None
    for _ in range(restarts):
        x_adv = la.generate_attack(model, X, y, eps, attack, mask=mask, steps=steps,
                                   alpha=alpha, evade_weight=beta, reduce=cfg.score_reduce)
        labels, _, flags = le.predict(model, x_adv, threshold_det=thr, reduce=cfg.score_reduce)
        task = labels == y
        e2e = task | flags if flags is not None else task
        robust_task = task if robust_task is None else robust_task & task
        robust_e2e = e2e if robust_e2e is None else robust_e2e & e2e
        if det_adv_acc is None and flags is not None:
            det_adv_acc = flags.float().mean().item()
    # computed as le.evaluate does: task_adv.acc is an integer ratio
    # (_binary_metrics), robust_acc_e2e a float32 mean
    return Robustness(int(robust_task.sum()) / len(y),
                      robust_e2e.float().mean().item(), det_adv_acc)


def _checkpoints(t, S, shorts):
    """(short name, seed, checkpoint) of every stage 1 run of the given
    models. A missing model leaves the "every model" criterion unverifiable,
    hence INCONCLUSIVE."""
    for short in shorts:
        name = f"s1/base/{short}"
        seeds = S.stage1.ckpt_seeds(name)
        if not seeds:
            t.inconclusive(f"{short}: checkpoint dello stadio 1", f"{name} assente")
            continue
        for seed in seeds:
            yield short, seed, S.ckpt(name, seed)


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.4f}"


# ===========================================================================
# A3 results, shared with A1, A2 and A4
# ===========================================================================
def a3_results(S) -> dict:
    """(short, seed) -> PGD and per-beta results of the detector models at
    their training eps, plus beta* (the beta > 0 with the lowest m_e2e)."""
    if "a3" not in S.store:
        X, y = S.rows("test", A_SAMPLES)
        out = {}
        for short in WITH_DET:
            name = f"s1/base/{short}"
            for seed in S.stage1.ckpt_seeds(name):
                ck = S.ckpt(name, seed)
                eps, mask = ck.config.eps, ck.attack_mask
                betas = {b: robust_eval(ck, X, y, "pgd_adaptive", eps, mask=mask, beta=b)
                         for b in BETA_GRID}
                out[(short, seed)] = {
                    "eps": eps,
                    "pgd": robust_eval(ck, X, y, "pgd", eps, mask=mask),
                    "betas": betas,
                    "beta_star": min((b for b in BETA_GRID if b > 0),
                                     key=lambda b: betas[b].m_e2e)}
        S.store["a3"] = out
    return S.store["a3"]


def a3_blocked(t, S) -> bool:
    """A3.1 is a blocking error: an adaptive attack that does not reduce to
    PGD at beta = 0 invalidates every result obtained with it."""
    bad = [k for k, r in a3_results(S).items() if r["betas"][0.0].m_e2e != r["pgd"].m_e2e]
    if bad:
        t.skip("esecuzione", f"bloccato da A3.1: beta=0 diverso da PGD su {bad}")
    return bool(bad)


def beta_star(S, short: str, seed: int) -> float:
    return a3_results(S)[(short, seed)]["beta_star"]


# ===========================================================================
# Tests
# ===========================================================================
@vc.vv_test(TESTS, "A3", "Attacco adattivo almeno efficace di PGD", vc.VALIDATE,
            "Sui modelli con detector, all'eps di addestramento, l'attacco adattivo con "
            "beta > 0 deve battere PGD di almeno TOL_A su m_e2e; sceglie beta*.")
def a3(t:vc.Report, S:vc.Session):
    X, y = S.rows("test", A_SAMPLES)
    res = a3_results(S)
    for short in WITH_DET:
        if not S.stage1.ckpt_seeds(f"s1/base/{short}"):
            t.inconclusive(f"{short}: checkpoint dello stadio 1", f"s1/base/{short} assente")
    for (short, seed), r in res.items():
        pgd = r["pgd"]
        t.info(f"{short} seed {seed} (eps={r['eps']:g}):")
        t.info(f"   {'attacco':12s} {'m_task':>8s} {'det_adv_acc':>12s} {'m_e2e':>8s}")
        t.info(f"   {'pgd':12s} {pgd.m_task:8.4f} {_fmt(pgd.det_adv_acc):>12s} {pgd.m_e2e:8.4f}")
        for b, rb in r["betas"].items():
            star = "  <- beta*" if b == r["beta_star"] else ""
            t.info(f"   {'beta=' + format(b, 'g'):12s} {rb.m_task:8.4f} "
                   f"{_fmt(rb.det_adv_acc):>12s} {rb.m_e2e:8.4f}{star}")

        # harness check: robust_eval measures what le.evaluate measures
        ck = S.ckpt(f"s1/base/{short}", seed)
        torch.manual_seed(vc.SEED)
        ev = le.evaluate(ck.model, X, y, eps=r["eps"], attack_mask=ck.attack_mask,
                         attack="pgd", threshold_det=ck.threshold_det,
                         attack_kwargs=dict(ck.config.attack_kwargs(), alpha=r["eps"] / 4),
                         reduce=ck.config.score_reduce)
        t.check(f"{short} seed {seed}: m_task e m_e2e coincidono con le.evaluate",
                (ev["task_adv"]["acc"], ev["robust_acc_e2e"]) == (pgd.m_task, pgd.m_e2e),
                f"le.evaluate {ev['task_adv']['acc']:.4f}/{ev['robust_acc_e2e']:.4f}")

        b0 = r["betas"][0.0].m_e2e
        if b0 != pgd.m_e2e:
            t.outcome(f"{short} seed {seed}: con beta=0 m_e2e coincide con PGD", vc.FAIL,
                      f"{b0:.4f} vs {pgd.m_e2e:.4f}: ERRORE BLOCCANTE")
            continue
        t.check(f"{short} seed {seed}: con beta=0 m_e2e coincide con PGD", True, f"{b0:.4f}")
        best = r["betas"][r["beta_star"]].m_e2e
        t.check(f"{short} seed {seed}: min su beta>0 di m_e2e < m_e2e(PGD) - TOL_A",
                best < pgd.m_e2e - TOL_A,
                f"beta*={r['beta_star']:g}: {best:.4f} vs {pgd.m_e2e - TOL_A:.4f}")


@vc.vv_test(TESTS, "A1", "Attacco iterativo almeno efficace del singolo passo", vc.VALIDATE,
            "PGD non deve fare peggio di FGSM su m_task; sui detector l'adattivo a 20 passi "
            "non deve fare peggio di quello a un passo su m_e2e.")
def a1(t:vc.Report, S:vc.Session):
    if a3_blocked(t, S):
        return
    X, y = S.rows("test", A_SAMPLES)
    for short, seed, ck in _checkpoints(t, S, NO_DET + WITH_DET):
        eps, mask = ck.config.eps, ck.attack_mask
        pgd = robust_eval(ck, X, y, "pgd", eps, mask=mask)
        fgsm = robust_eval(ck, X, y, "fgsm", eps, mask=mask)
        t.check(f"{short} seed {seed}: m_task(PGD) <= m_task(FGSM) + TOL_A",
                pgd.m_task <= fgsm.m_task + TOL_A, f"{pgd.m_task:.4f} vs {fgsm.m_task:.4f}")
        if short in WITH_DET:
            b = beta_star(S, short, seed)
            many = robust_eval(ck, X, y, "pgd_adaptive", eps, mask=mask, steps=ITER_STEPS, beta=b)
            one = robust_eval(ck, X, y, "pgd_adaptive", eps, mask=mask, steps=1, alpha=eps, beta=b)
            t.check(f"{short} seed {seed}: m_e2e(adattivo {ITER_STEPS} passi) <= "
                    f"m_e2e(adattivo 1 passo) + TOL_A",
                    many.m_e2e <= one.m_e2e + TOL_A,
                    f"beta*={b:g}: {many.m_e2e:.4f} vs {one.m_e2e:.4f}")


def _monotone_violations(ms: dict) -> list[str]:
    """Pairs eps_i < eps_j where the larger budget leaves more robustness."""
    eps = sorted(ms)
    return [f"eps {ei:g}->{ej:g}: {ms[ei]:.4f}->{ms[ej]:.4f}"
            for i, ei in enumerate(eps) for ej in eps[i + 1:] if ms[ej] > ms[ei] + TOL_A]


@vc.vv_test(TESTS, "A2", "Successo crescente con eps", vc.VALIDATE,
            "Su EPS_GRID la robustezza non deve crescere con il budget: m_task con PGD su tutti "
            "i modelli, m_e2e con l'adattivo e beta* sui detector.")
def a2(t:vc.Report, S:vc.Session):
    if a3_blocked(t, S):
        return
    X, y = S.rows("test", A_SAMPLES)
    n_pairs = len(EPS_GRID) * (len(EPS_GRID) - 1) // 2
    for short, seed, ck in _checkpoints(t, S, NO_DET + WITH_DET):
        mask = ck.attack_mask
        ms = {e: robust_eval(ck, X, y, "pgd", e, mask=mask).m_task for e in EPS_GRID}
        t.info(f"{short} seed {seed} m_task(PGD): "
               + "  ".join(f"{e:g}:{m:.4f}" for e, m in ms.items()))
        bad = _monotone_violations(ms)
        t.check(f"{short} seed {seed}: m_task(PGD) non crescente in eps ({n_pairs} coppie)",
                not bad, "; ".join(bad) or None)
        if short in WITH_DET:
            b = beta_star(S, short, seed)
            ms = {e: robust_eval(ck, X, y, "pgd_adaptive", e, mask=mask, beta=b).m_e2e
                  for e in EPS_GRID}
            t.info(f"{short} seed {seed} m_e2e(adattivo, beta*={b:g}): "
                   + "  ".join(f"{e:g}:{m:.4f}" for e, m in ms.items()))
            bad = _monotone_violations(ms)
            t.check(f"{short} seed {seed}: m_e2e(adattivo) non crescente in eps "
                    f"({n_pairs} coppie)", not bad, "; ".join(bad) or None)


@vc.vv_test(TESTS, "A4", "Budget illimitato e convergenza", vc.VALIDATE,
            "Con eps enorme l'attacco deve azzerare la robustezza; piu' passi e ripartenze "
            "non devono abbassarla oltre TOL_CONV.")
def a4(t:vc.Report, S:vc.Session):
    if a3_blocked(t, S):
        return
    X, y = S.rows("test", A_SAMPLES)
    for short, seed, ck in _checkpoints(t, S, NO_DET + WITH_DET):
        mask = ck.attack_mask
        if short in WITH_DET:
            attack, beta, key = "pgd_adaptive", beta_star(S, short, seed), "m_e2e"
        else:
            attack, beta, key = "pgd", None, "m_task"
        what = f"{key} ({attack}" + (f", beta*={beta:g})" if beta is not None else ")")

        # 1) unbounded budget
        large = {e: getattr(robust_eval(ck, X, y, attack, e, mask=mask,
                                        steps=UNBOUNDED_STEPS, beta=beta), key)
                 for e in EPS_LARGE}
        t.info(f"{short} seed {seed} {what}, {UNBOUNDED_STEPS} passi: "
               + "  ".join(f"eps {e:g}:{m:.4f}" for e, m in large.items()))
        m = large[UNBOUNDED_EPS]
        label = f"{short} seed {seed}: eps={UNBOUNDED_EPS:g} -> {key} <= {UNBOUNDED_MAX}"
        if m <= UNBOUNDED_MAX:
            t.check(label, True, f"{m:.4f}")
        else:
            m_all = getattr(robust_eval(ck, X, y, attack, UNBOUNDED_EPS, mask=torch.ones_like(mask),
                                        steps=UNBOUNDED_STEPS, beta=beta), key)
            t.check(label, m_all <= UNBOUNDED_MAX,
                    f"{m:.4f} con maschera; {m_all:.4f} con maschera tutta a uno"
                    + (": plateau dovuto alle feature non attaccabili"
                       if m_all <= UNBOUNDED_MAX else ""))

        # 2) convergence, success combined in OR over the restarts
        eps = ck.config.eps
        conv = {(st, rs): getattr(robust_eval(ck, X, y, attack, eps, mask=mask, steps=st,
                                              beta=beta, restarts=rs), key)
                for st, rs in CONV_SETTINGS}
        t.info(f"{short} seed {seed} {what}, eps={eps:g}: "
               + "  ".join(f"{st}x{rs}:{v:.4f}" for (st, rs), v in conv.items()))
        gap = conv[(20, 1)] - conv[(100, 5)]
        t.check(f"{short} seed {seed}: m(20 passi, 1 partenza) - m(100 passi, 5 partenze) "
                f"<= TOL_CONV", gap <= TOL_CONV, f"{gap:+.4f}")


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
