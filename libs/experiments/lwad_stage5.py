"""Stage 5: the worst case of the stage 4 winners.

    python scripts/experiment.py --only s5/ --dataset UNSW_BW15 --seeds 42

No retraining: every run reloads the checkpoint stage 4 saved for its winner,
on the dataset and seed of the run, and changes only the attack. The detector
threshold stays the one picked on the training attack, as for a deployed
defense that does not know what is coming. Per winner:

    search  fgsm over eps; 
            pgd over steps x step size;
            pgd_adaptive over steps x step size x evade weight
    curve   other budgets, at 100 steps: pgd, and pgd_adaptive per evade weight

Runs: 18 detector models x 41 + 6 models without detectors x 9 = 792, but
each is an evaluation: seconds instead of minutes.
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
# ===========================================================================
# --- detector models trained and evaluated on pgd --------------------------
DET_MINI_PGD    = trained("s4/det/pgd/mini/[task_loss_on_adv=False]")
FUR_MINI_PGD    = trained("s4/fur/pgd/mini/[lambda_act=1,task_loss_on_adv=False]")

DET_MEDIUM_PGD  = trained("s4/det/pgd/medium/[task_loss_on_adv=False]")
FUR_MEDIUM_PGD  = trained("s4/fur/pgd/medium/[lambda_act=1,task_loss_on_adv=False]")

DET_HIGH_PGD    = trained("s4/det/pgd/high/[task_loss_on_adv=False]")
FUR_HIGH_PGD    = trained("s4/fur/pgd/high/[lambda_act=1,task_loss_on_adv=False]")

# --- detector models trained and evaluated on pgd_adaptive -----------------
# lambda_det=1 is the default, so it is not in the name
DET_MINI_APGD   = trained("s4/det/apgd/mini/[task_loss_on_adv=False]")
FUR_MINI_APGD   = trained("s4/fur/apgd/mini/[lambda_act=2,task_loss_on_adv=False]")

DET_MEDIUM_APGD = trained("s4/det/apgd/medium/[task_loss_on_adv=False]")
FUR_MEDIUM_APGD = trained("s4/fur/apgd/medium/[lambda_act=2,task_loss_on_adv=False]")

DET_HIGH_APGD   = trained("s4/det/apgd/high/[task_loss_on_adv=False]")
FUR_HIGH_APGD   = trained("s4/fur/apgd/high/[lambda_act=2,task_loss_on_adv=False]")

# comparison: the pgd_adaptive winners with task_loss_on_adv=True ---------
DET_MINI_APGD_ADV   = trained("s4/det/apgd/mini/[task_loss_on_adv=True]")
FUR_MINI_APGD_ADV   = trained("s4/fur/apgd/mini/[lambda_act=2,task_loss_on_adv=True]")

DET_MEDIUM_APGD_ADV = trained("s4/det/apgd/medium/[task_loss_on_adv=True]")
FUR_MEDIUM_APGD_ADV = trained("s4/fur/apgd/medium/[lambda_act=2,task_loss_on_adv=True]")

DET_HIGH_APGD_ADV   = trained("s4/det/apgd/high/[task_loss_on_adv=True]")
FUR_HIGH_APGD_ADV   = trained("s4/fur/apgd/high/[lambda_act=2,task_loss_on_adv=True]")

# --- CloserAL, pgd only -----------------------------------------------------
CLO_MINI   = trained("s4/clo/mini/[lambda_act=1]")
CLO_MEDIUM = trained("s4/clo/medium/[lambda_act=1]")
CLO_HIGH   = trained("s4/clo/high/[lambda_act=1]")

# comparison: adversarial training, CloserAL with lambda_act=0 -------------
ADV_MINI   = trained("s4/advtrain/mini")
ADV_MEDIUM = trained("s4/advtrain/medium")
ADV_HIGH   = trained("s4/advtrain/high")


# ===========================================================================
# Axes
# ===========================================================================
EPS        = s2.COMMON["eps"]           # the training budget: the threat model
STEPS      = [20, 100]                  # the training attack, then near convergence
LONG_STEPS = [max(STEPS)]
ALPHAS     = [None, EPS / 5]            # None: 2*eps/steps; eps/4 reaches the
                                        # corners of the ball in a few steps
BETAS      = [1.0, 5.0, 10.0, 100.0]    # the detector score is a sigmoid that
                                        # saturates: its gradient needs a large
                                        # weight to count against the task loss
EPS_CURVE  = [0.1, 0.3, 0.5, 1.0]       # the training eps is in the search


def attacks(name: str, model: Optional[lc.ModelConfig]) -> list[Exp]:
    """The attacks a reloaded winner faces."""
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

        # --- controls: pgd_adaptive winners with task_loss_on_adv=True -----
        *attacks("s5/det-advtrain/apgd/mini",   DET_MINI_APGD_ADV),
        *attacks("s5/det-advtrain/apgd/medium", DET_MEDIUM_APGD_ADV),
        *attacks("s5/det-advtrain/apgd/high",   DET_HIGH_APGD_ADV),
        *attacks("s5/fur-advtrain/apgd/mini",   FUR_MINI_APGD_ADV),
        *attacks("s5/fur-advtrain/apgd/medium", FUR_MEDIUM_APGD_ADV),
        *attacks("s5/fur-advtrain/apgd/high",   FUR_HIGH_APGD_ADV),

        # --- CloserAL winners: no detector, so no adaptive attack ----------
        *attacks("s5/clo/mini",   CLO_MINI),
        *attacks("s5/clo/medium", CLO_MEDIUM),
        *attacks("s5/clo/high",   CLO_HIGH),

        # --- control: adversarial training, same architectures as CloserAL -
        *attacks("s5/advtrain/mini",   ADV_MINI),
        *attacks("s5/advtrain/medium", ADV_MEDIUM),
        *attacks("s5/advtrain/high",   ADV_HIGH),
    ]


lx.EXPERIMENTS.extend(table())