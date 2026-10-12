"""Stage 5: the worst case of the stage 4 winners.

    python scripts/experiment.py --only s5/ --dataset UNSW_BW15 --seeds 42

No retraining: every run reloads the checkpoint stage 4 saved for its winner,
on the dataset and seed of the run, and changes only the attack. The detector
threshold stays the one picked on the training attack, as for a deployed
defense that does not know what is coming. Per winner:

    search  at the training eps: fgsm; pgd over steps x step size; against the
            detectors too, pgd_adaptive over steps x step size x evade weight
    curve   other budgets, at 100 steps: pgd, and pgd_adaptive per evade weight

The search run with steps=20, alpha=None (and evade weight 1) on the winner's
own attack repeats its stage 4 evaluation. A winner's worst case is its lowest
robust_acc_e2e at the training eps, the curve the lowest per eps.

The CloserAL architectures trained with lambda_act=0 (plain adversarial
training, stage 4 trains them) face the same attacks as the CloserAL winners:
the gap between the two is what the attractive loss adds.

Runs: 12 detector winners x 41 + 3 CloserAL x 9 + 3 adversarial training x 9
= 546, but each is an evaluation: seconds instead of minutes.
"""
from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
import libs.experiments.lwad_stage2 as s2
import libs.experiments.lwad_stage4 as s4
from libs.experiments.lwad_experiments import Exp, Sweep


# ===========================================================================
# Reloading the stage 4 runs
# ===========================================================================
CHECKPOINT_DIR = os.path.join("results", "checkpoints")   # run_suite's default

_STAGE4 = {e.name: e for e in lx.expand_all(s4.table())}


def trained(name: str) -> Optional[lc.ModelConfig]:
    """The config of the stage 4 run called name, reloading its checkpoint
    instead of training. None, with a warning, if stage 4 has no such run:
    stale names must not stop the other stages from running."""
    exp = _STAGE4.get(name)
    if exp is None:
        print(f"warning: stage 5: no stage 4 run is called {name!r}, its attacks "
              f"are left out (update the winners in lwad_stage5.py)")
        return None
    return replace(exp.config(),
                   load_from=os.path.join(CHECKPOINT_DIR, lx.checkpoint_file(name)))


# ===========================================================================
# Stage 4 winners: run names as the stage 4 summary prints them
# PLACEHOLDERS (the stage 4 runs that keep the stage 3 config): set the best.
# ===========================================================================
# --- detector models trained and evaluated on pgd --------------------------
DET_MINI_PGD    = trained("s4/det/pgd/mini/[task_loss_on_adv=False]")
FUR_MINI_PGD    = trained("s4/fur/pgd/mini/[lambda_act=1,task_loss_on_adv=False]")

DET_MEDIUM_PGD  = trained("s4/det/pgd/medium/[task_loss_on_adv=False]")
FUR_MEDIUM_PGD  = trained("s4/fur/pgd/medium/[lambda_act=1,task_loss_on_adv=False]")

DET_HIGH_PGD    = trained("s4/det/pgd/high/[task_loss_on_adv=False]")
FUR_HIGH_PGD    = trained("s4/fur/pgd/high/[lambda_act=1,task_loss_on_adv=False]")

# --- detector models trained and evaluated on pgd_adaptive -----------------
DET_MINI_APGD   = trained("s4/det/apgd/mini/[task_loss_on_adv=False]")
FUR_MINI_APGD   = trained("s4/fur/apgd/mini/[lambda_act=1,task_loss_on_adv=False]")

DET_MEDIUM_APGD = trained("s4/det/apgd/medium/[task_loss_on_adv=False]")
FUR_MEDIUM_APGD = trained("s4/fur/apgd/medium/[lambda_act=1,task_loss_on_adv=False]")

DET_HIGH_APGD   = trained("s4/det/apgd/high/[task_loss_on_adv=False]")
FUR_HIGH_APGD   = trained("s4/fur/apgd/high/[lambda_act=1,task_loss_on_adv=False]")

# --- CloserAL, pgd only -----------------------------------------------------
CLO_MINI   = trained("s4/clo/mini/[lambda_act=1]")
CLO_MEDIUM = trained("s4/clo/medium/[lambda_act=1]")
CLO_HIGH   = trained("s4/clo/high/[lambda_act=1]")

# --- Adversarial training: the CloserAL architectures, lambda_act=0 --------
# not winners (nothing to pick): the baseline CloserAL is compared against
ADV_MINI   = trained("s4/advtrain/mini")
ADV_MEDIUM = trained("s4/advtrain/medium")
ADV_HIGH   = trained("s4/advtrain/high")


# ===========================================================================
# Axes
# ===========================================================================
EPS        = s2.COMMON["eps"]          # the training budget: the threat model
STEPS      = [20, 100]                 # the training attack, then near convergence
LONG_STEPS = [max(STEPS)]
ALPHAS     = [None, EPS / 4]           # None: 2*eps/steps; eps/4 reaches the
                                       # corners of the ball in a few steps
BETAS      = [1.0, 10.0, 100.0, 1000.0]  # the detector score is a sigmoid that
                                         # saturates: its gradient needs a large
                                         # weight to count against the task loss
EPS_CURVE  = [0.1, 0.3, 0.5, 1.0]      # the training eps is in the search


def attacks(name: str, model: Optional[lc.ModelConfig]) -> list[Exp]:
    """The attacks one reloaded winner faces."""
    if model is None:
        return []
    # the eval attack is set explicitly: the winner's own may be either one
    pgd = replace(model, eval_attack="pgd")
    exps = [
        Exp(f"{name}/fgsm", model, eval_attack="fgsm"),
        Sweep(f"{name}/pgd/", pgd, pgd_steps=STEPS, pgd_alpha=ALPHAS),
        Sweep(f"{name}/eps/pgd/", pgd, eps=EPS_CURVE, pgd_steps=LONG_STEPS),
    ]
    if model.uses_detectors:            # white box on the detectors as well
        adaptive = replace(model, eval_attack="pgd_adaptive")
        exps += [
            Sweep(f"{name}/adaptive/", adaptive, pgd_steps=STEPS,
                  pgd_alpha=ALPHAS, pgd_evade_weight=BETAS),
            Sweep(f"{name}/eps/adaptive/", adaptive, eps=EPS_CURVE,
                  pgd_steps=LONG_STEPS, pgd_evade_weight=BETAS),
        ]
    return exps


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- DetectorLayer winners -----------------------------------------
        *attacks("s5/det/pgd/mini",    DET_MINI_PGD),
        *attacks("s5/det/pgd/medium",  DET_MEDIUM_PGD),
        *attacks("s5/det/pgd/high",    DET_HIGH_PGD),
        *attacks("s5/det/apgd/mini",   DET_MINI_APGD),
        *attacks("s5/det/apgd/medium", DET_MEDIUM_APGD),
        *attacks("s5/det/apgd/high",   DET_HIGH_APGD),

        # --- FurtherAL winners ---------------------------------------------
        *attacks("s5/fur/pgd/mini",    FUR_MINI_PGD),
        *attacks("s5/fur/pgd/medium",  FUR_MEDIUM_PGD),
        *attacks("s5/fur/pgd/high",    FUR_HIGH_PGD),
        *attacks("s5/fur/apgd/mini",   FUR_MINI_APGD),
        *attacks("s5/fur/apgd/medium", FUR_MEDIUM_APGD),
        *attacks("s5/fur/apgd/high",   FUR_HIGH_APGD),

        # --- CloserAL winners: no detector, so no adaptive attack ----------
        *attacks("s5/clo/mini",   CLO_MINI),
        *attacks("s5/clo/medium", CLO_MEDIUM),
        *attacks("s5/clo/high",   CLO_HIGH),

        # --- Adversarial training: CloserAL without its attractive loss ----
        *attacks("s5/advtrain/mini",   ADV_MINI),
        *attacks("s5/advtrain/medium", ADV_MEDIUM),
        *attacks("s5/advtrain/high",   ADV_HIGH),
    ]


lx.EXPERIMENTS.extend(table())
