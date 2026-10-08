"""V&V suite - Validazione: credibilita' della robustezza (A3, A1, A2, A4).

The stage 1 checkpoints are re-evaluated with their saved threshold on a
fixed subset of 2000 test samples. The seed is fixed before every
evaluation. A3 and A1 use the campaign's attack (alpha = 2*eps/steps);
A2 and A4 give every attack of a test the same step, and enough steps to
cross the ball of the largest eps of the test (its diameter, 2 * eps):
    bounded budget   (A2, A4.2): alpha = PGD_ALPHA
    unbounded budget (A4.1):     alpha = UNBOUNDED_ALPHA
On the detector models the adaptive attack of A2 and A4 also draws its
random start within the training eps, not within the whole ball, and keeps
every sample's best iterate (attack_options, pgd_adaptive_from). In the
unbounded budget it tries every beta of BETA_GRID instead of beta* alone
(robust_any_beta).

    m_task = task_adv.acc     (adversarial accuracy of the classifier)
    m_e2e  = robust_acc_e2e   (adversarial sample classified right OR flagged)

A3 runs first: it checks the adaptive attack and picks beta*, the evade
weight that A1, A2 and A4 use against the detector models.

Run:  python -m libs.tests.vv_validate_robustness [--only A3]
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import math
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

import libs.model.lwad_config as lc
import libs.attacks.lwad_attack as la
import libs.evaluation.lwad_evaluator as le

TESTS: list = []

TOL_A = 0.005
EPS_GRID = (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)
BETA_GRID = (0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0)
EPS_LARGE = (2.0, 3.0, 5.0)
UNBOUNDED_MAX = 0.05
# (steps as a multiple of steps_to_cross(eps, PGD_ALPHA), restarts): at
# eps = 0.2 they are 40x1, 100x1, 200x1 and 200x5
CONV_SETTINGS = ((1, 1), (2.5, 1), (5, 1), (5, 5))
TOL_CONV = 0.01

A_SAMPLES = 2000
ITER_STEPS = 20             # A1: steps of the iterative adaptive attack
PGD_ALPHA = 0.01            # A2, A4.2 (bounded budget): step of every attack
UNBOUNDED_ALPHA = 0.01      # A4.1 (unbounded budget): step of every attack; at 0.05
                            # further jumps over its narrow success region
UNBOUNDED_EPS = max(EPS_LARGE)   # A4.1: the eps that must zero the robustness

NO_DET = ("undefended", "advtrain", "closer", "closer-clean")
WITH_DET = ("detlayer", "detlayer-advtrain", "further", "further-advtrain")


# ===========================================================================
# Evaluation of one attack on a stage 1 checkpoint
# ===========================================================================
@dataclass
class Robustness:
    m_task: float
    m_e2e: float
    det_adv_acc: Optional[float]     # None for the models without detectors


def pgd_adaptive_from(model, x, y, eps, radius, *, steps, alpha, mask, evade_weight, reduce,
                      thr: Optional[float] = None):
    """la.pgd_adaptive with two changes, used by A2 and A4:
    - the random start is drawn within radius <= eps instead of eps (the
      projection stays on the eps ball);
    - with thr, every sample gets its best iterate: the first point of the
      walk (start included) misclassified and not flagged at thr, the last
      one if no point is. Without it the last iterate, as la.pgd_adaptive.
    evade_weight can also be a tensor, one weight per sample.
    With radius = eps, thr = None and a float evade_weight it is
    la.pgd_adaptive bit for bit (checked by check_local_attack)."""
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-radius, radius)
    if mask is not None:
        x_adv = x_orig + (x_adv - x_orig) * mask
    x_adv = x_adv.detach()
    x_best, found = x_adv.clone(), torch.zeros(len(y), dtype=torch.bool, device=x.device)

    with la._detector_grad_enabled(model):
        for step in range(steps + 1):        # one more forward: the last iterate counts
            x_adv.requires_grad_(True)
            logits, state = model(x_adv)
            score = state.adv_score(reduce=reduce)
            if thr is not None:
                with torch.no_grad():        # success as le.predict: flagged when score > thr
                    new = ~found & (logits.argmax(-1) != y) & (score <= thr)
                    x_best[new] = x_adv.detach()[new]
                    found |= new
            if step == steps:
                break
            if torch.is_tensor(evade_weight):    # one weight per sample (robust_any_beta)
                task_loss = F.cross_entropy(input=logits, target=y, reduction="none")
                objective = (task_loss - evade_weight * score).mean()
            else:
                task_loss = F.cross_entropy(input=logits, target=y)
                objective = task_loss - evade_weight * score.mean()
            (grad,) = torch.autograd.grad(outputs=objective, inputs=x_adv)
            with torch.no_grad():
                delta = alpha * grad.sign()
                if mask is not None:
                    delta = delta * mask
                x_adv = x_adv + delta
                x_adv = x_orig + torch.clamp(x_adv - x_orig, -eps, eps)
            x_adv = x_adv.detach()
    x_adv = x_adv.detach()
    if thr is None:
        return x_adv
    x_best[~found] = x_adv[~found]
    return x_best


def robust_eval(ck, X, y, attack: str, eps: float, **kw) -> Robustness:
    """m_task and m_e2e of one attack (see attack_masks)."""
    task, e2e, flags = attack_masks(ck, X, y, attack, eps, **kw)
    # computed as le.evaluate does: task_adv.acc is an integer ratio
    # (_binary_metrics), robust_acc_e2e a float32 mean
    return Robustness(int(task.sum()) / len(y), e2e.float().mean().item(),
                      None if flags is None else flags.float().mean().item())


def robust_any_beta(ck, X, y, eps: float, *, mask, steps: int, alpha: float,
                    radius: float) -> Robustness:
    """The adaptive attack with every beta of BETA_GRID and the best iterate:
    a sample is broken as soon as one beta breaks it (OR over beta, as over
    the restarts). beta* is picked at the training eps, where the
    adversarially trained models keep m_task ~ 0.8 whatever the attack; at a
    large eps it can leave the CE no say (beta* = 50 on further-advtrain).
    The betas run in one batch, X repeated once per beta with its own evade
    weight: the model treats every sample on its own (LayerNorm, no
    BatchNorm), so a copy's gradient depends on its own term only."""
    k, n = len(BETA_GRID), len(y)
    betas = torch.tensor(BETA_GRID, dtype=X.dtype, device=X.device).repeat_interleave(n)
    task, e2e, _ = attack_masks(ck, X.repeat(k, 1), y.repeat(k), "pgd_adaptive", eps,
                                mask=mask, steps=steps, alpha=alpha, beta=betas,
                                radius=radius, best_iterate=True)
    task, e2e = task.view(k, n).all(0), e2e.view(k, n).all(0)
    return Robustness(int(task.sum()) / n, e2e.float().mean().item(), None)


def attack_masks(ck, X, y, attack: str, eps: float, *, mask, steps: Optional[int] = None,
                 alpha: Optional[float] = None, beta=None, restarts: int = 1,
                 radius: Optional[float] = None, best_iterate: bool = False):
    """Per sample robustness of one attack, with the checkpoint's saved
    threshold and score reduction: (task, e2e) booleans as le.evaluate
    defines them, plus the flags of the first restart (None without
    detectors). With restarts > 1 the attack succeeds on a sample as soon as
    one restart does (OR over the restarts).
    radius and best_iterate (pgd_adaptive only, see attack_options) switch
    to pgd_adaptive_from; by default the library attack."""
    model, cfg = ck.model, ck.config
    thr = ck.threshold_det if ck.threshold_det is not None else lc.DEFAULT_THRESHOLD_DET
    steps = cfg.pgd_steps if steps is None else steps
    alpha = 2* eps / steps if alpha is None else alpha
    torch.manual_seed(vc.SEED)
    robust_task = robust_e2e = first_flags = None
    for _ in range(restarts):
        if radius is None and not best_iterate:
            x_adv = la.generate_attack(model, X, y, eps, attack, mask=mask, steps=steps,
                                       alpha=alpha, evade_weight=beta, reduce=cfg.score_reduce)
        else:
            x_adv = pgd_adaptive_from(model, X, y, eps, eps if radius is None else radius,
                                      steps=steps, alpha=alpha, mask=mask, evade_weight=beta,
                                      reduce=cfg.score_reduce, thr=thr if best_iterate else None)
        labels, _, flags = le.predict(model, x_adv, threshold_det=thr, reduce=cfg.score_reduce)
        task = labels == y
        e2e = task | flags if flags is not None else task
        robust_task = task if robust_task is None else robust_task & task
        robust_e2e = e2e if robust_e2e is None else robust_e2e & e2e
        if first_flags is None:
            first_flags = flags
    return robust_task, robust_e2e, first_flags


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


def steps_to_cross(eps: float, alpha: float) -> int:
    """Fewest steps of alpha that cross the eps ball (2 * eps): from any
    random start every point of the ball is reachable (rounded first:
    0.07 / 0.01 = 7.000000000000001 would get one step more)."""
    return math.ceil(round(2 * eps / alpha, 6))


def attack_options(short: str, ck, eps: float) -> dict:
    """robust_eval options of A2 and A4. Without detectors none: the library
    PGD, whose loss is the success criterion itself. On the detector models:
    - radius: the random start within the training eps, the perturbations
      the detector was trained on; a wider uniform start lands where the
      detector flags with a saturated score, the evade gradient vanishes
      and the attack cannot walk back;
    - best_iterate: CE - beta * score is not the success criterion, and
      the walk loses successes it has already reached (further keeps
      pushing CE until the detector flags again)."""
    if short not in WITH_DET:
        return {}
    return dict(radius=min(eps, ck.config.eps), best_iterate=True)


def check_local_attack(t, S) -> None:
    """Harness check of A2 and A4: with radius = eps and the last iterate,
    pgd_adaptive_from is la.pgd_adaptive bit for bit, so the start and the
    choice of the iterate are the only changes."""
    for short in WITH_DET:
        seeds = S.stage1.ckpt_seeds(f"s1/base/{short}")
        if seeds:
            break
    else:
        return
    ck, X, y = S.ckpt(f"s1/base/{short}", seeds[0]), *S.rows("test", A_SAMPLES)
    eps, kw = ck.config.eps, dict(steps=10, alpha=PGD_ALPHA, mask=ck.attack_mask,
                                  evade_weight=beta_star(S, short, seeds[0]),
                                  reduce=ck.config.score_reduce)
    torch.manual_seed(vc.SEED)
    ref = la.pgd_adaptive(ck.model, X, y, eps, **kw)
    torch.manual_seed(vc.SEED)
    got = pgd_adaptive_from(ck.model, X, y, eps, eps, **kw)
    t.check(f"{short} seed {seeds[0]}: attacco locale con partenza entro eps e ultimo "
            f"iterato identico a la.pgd_adaptive",
            torch.equal(ref, got), f"{int((ref != got).sum())} elementi diversi")


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
                         attack_kwargs=dict(ck.config.attack_kwargs(), 
                                            alpha=2 * r["eps"] / ck.config.pgd_steps),
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
            "i modelli, m_e2e con l'adattivo e beta* sui detector. Ogni attacco usa "
            f"alpha={PGD_ALPHA:g} e {steps_to_cross(max(EPS_GRID), PGD_ALPHA)} passi, quanti "
            f"ne servono per attraversare la palla di eps={max(EPS_GRID):g}; sui detector la "
            "partenza casuale resta entro l'eps di addestramento e conta il miglior iterato.")
def a2(t:vc.Report, S:vc.Session):
    if a3_blocked(t, S):
        return
    check_local_attack(t, S)
    X, y = S.rows("test", A_SAMPLES)
    n_pairs = len(EPS_GRID) * (len(EPS_GRID) - 1) // 2
    steps = steps_to_cross(max(EPS_GRID), PGD_ALPHA)
    for short, seed, ck in _checkpoints(t, S, NO_DET + WITH_DET):
        mask = ck.attack_mask
        ms = {e: robust_eval(ck, X, y, "pgd", e, mask=mask, steps=steps,
                             alpha=PGD_ALPHA).m_task for e in EPS_GRID}
        t.info(f"{short} seed {seed} m_task(PGD): "
               + "  ".join(f"{e:g}:{m:.4f}" for e, m in ms.items()))
        bad = _monotone_violations(ms)
        t.check(f"{short} seed {seed}: m_task(PGD) non crescente in eps ({n_pairs} coppie)",
                not bad, "; ".join(bad) or None)
        if short in WITH_DET:
            b = beta_star(S, short, seed)
            ms = {e: robust_eval(ck, X, y, "pgd_adaptive", e, mask=mask, steps=steps,
                                 alpha=PGD_ALPHA, beta=b,
                                 **attack_options(short, ck, e)).m_e2e
                  for e in EPS_GRID}
            t.info(f"{short} seed {seed} m_e2e(adattivo, beta*={b:g}, "
                   f"partenza entro {ck.config.eps:g}, miglior iterato): "
                   + "  ".join(f"{e:g}:{m:.4f}" for e, m in ms.items()))
            bad = _monotone_violations(ms)
            t.check(f"{short} seed {seed}: m_e2e(adattivo) non crescente in eps "
                    f"({n_pairs} coppie)", not bad, "; ".join(bad) or None)


@vc.vv_test(TESTS, "A4", "Budget illimitato e convergenza", vc.VALIDATE,
            f"Con eps enorme (alpha={UNBOUNDED_ALPHA:g}, "
            f"{steps_to_cross(UNBOUNDED_EPS, UNBOUNDED_ALPHA)} passi; sui detector ogni beta "
            "di BETA_GRID, in OR) l'attacco deve azzerare la robustezza; all'eps di "
            f"addestramento, con alpha={PGD_ALPHA:g} e beta*, piu' passi e ripartenze non "
            "devono abbassarla oltre TOL_CONV. Sui detector la partenza casuale resta entro "
            "l'eps di addestramento e conta il miglior iterato.")
def a4(t:vc.Report, S:vc.Session):
    if a3_blocked(t, S):
        return
    check_local_attack(t, S)
    X, y = S.rows("test", A_SAMPLES)
    for short, seed, ck in _checkpoints(t, S, NO_DET + WITH_DET):
        mask = ck.attack_mask
        if short in WITH_DET:
            attack, beta, key = "pgd_adaptive", beta_star(S, short, seed), "m_e2e"
        else:
            attack, beta, key = "pgd", None, "m_task"
        what = f"{key} ({attack}" + (f", beta*={beta:g})" if beta is not None else ")")

        start = ("" if short not in WITH_DET
                 else f", partenza entro {ck.config.eps:g}, miglior iterato")

        # 1) unbounded budget: same step and same steps for every eps; on the
        # detector models every beta, not beta* alone
        steps = steps_to_cross(UNBOUNDED_EPS, UNBOUNDED_ALPHA)
        def unbounded(e, attack_mask):
            if short in WITH_DET:
                return robust_any_beta(ck, X, y, e, mask=attack_mask, steps=steps,
                                       alpha=UNBOUNDED_ALPHA,
                                       radius=attack_options(short, ck, e)["radius"]).m_e2e
            return robust_eval(ck, X, y, attack, e, mask=attack_mask, steps=steps,
                               alpha=UNBOUNDED_ALPHA).m_task
        large = {e: unbounded(e, mask) for e in EPS_LARGE}
        what_large = (f"{key} ({attack}, OR su beta in {list(BETA_GRID)})"
                      if short in WITH_DET else what)
        t.info(f"{short} seed {seed} {what_large}, {steps} passi, "
               f"alpha={UNBOUNDED_ALPHA:g}{start}: "
               + "  ".join(f"eps {e:g}:{m:.4f}" for e, m in large.items()))
        m = large[UNBOUNDED_EPS]
        label = f"{short} seed {seed}: eps={UNBOUNDED_EPS:g} -> {key} <= {UNBOUNDED_MAX}"
        if m <= UNBOUNDED_MAX:
            t.check(label, True, f"{m:.4f}")
        else:
            m_all = unbounded(UNBOUNDED_EPS, torch.ones_like(mask))
            t.check(label, m_all <= UNBOUNDED_MAX,
                    f"{m:.4f} con maschera; {m_all:.4f} con maschera tutta a uno"
                    + (": plateau dovuto alle feature non attaccabili"
                       if m_all <= UNBOUNDED_MAX else ""))

        # 2) convergence at alpha = PGD_ALPHA, success combined in OR over the
        # restarts; the fewest steps already cross the ball
        eps = ck.config.eps
        settings = [(round(k * steps_to_cross(eps, PGD_ALPHA)), rs) for k, rs in CONV_SETTINGS]
        conv = {(st, rs): getattr(robust_eval(ck, X, y, attack, eps, mask=mask, steps=st,
                                              alpha=PGD_ALPHA, beta=beta, restarts=rs,
                                              **attack_options(short, ck, eps)), key)
                for st, rs in settings}
        t.info(f"{short} seed {seed} {what}, eps={eps:g}, alpha={PGD_ALPHA:g}{start}: "
               + "  ".join(f"{st}x{rs}:{v:.4f}" for (st, rs), v in conv.items()))
        (lo, lo_rs), (hi, hi_rs) = settings[0], settings[-1]
        gap = conv[(lo, lo_rs)] - conv[(hi, hi_rs)]
        t.check(f"{short} seed {seed}: m({lo} passi, {lo_rs} partenza) - "
                f"m({hi} passi, {hi_rs} partenze) <= TOL_CONV", gap <= TOL_CONV, f"{gap:+.4f}")


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
