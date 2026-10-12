"""Stage 4: the loss weights of the stage 3 winners.

    python scripts/experiment.py --only s4/ --dataset UNSW_BW15 --seeds 42

Every winner keeps its stage 3 mechanism and explores only the weights its
losses actually use, on log grids:

    DetectorLayer   task_loss_on_adv
    FurtherAL       lambda_act x task_loss_on_adv, crossed: they push the same
                    activations in opposite directions (stage 1: lambda_act=10
                    helps a lot once the backbone is adversarially trained),
                    and margin_factor, the distance FurtherAL pushes to
    CloserAL        lambda_act, task_loss_on_adv being mandatory there
    detach=False    lambda_det x task_loss_on_adv: only then does the detector
                    loss reach the backbone

With detach=True lambda_det scales the gradient of the detector alone, and
Adam normalizes a constant factor away: s4/vv verifies it on one model.

If lambda_act is too low (0.1, 0.5), the repulsive loss loses against the task
loss and the samples never reach their margin, so margin_factor=2,5,8 give always
the same gradients at each step and runs are identical (with lambda_act=0 the margin
is useless). With lambda_act >= 1, exploring the margin makes sense.

Runs with the placeholder winners: 6 DetectorLayer x 2 + 6 FurtherAL x (10+12)
+ 3 CloserAL x 5 + 3 = 162; each detector winner with detach=False adds 6.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
import libs.experiments.lwad_stage3 as s3
from libs.experiments.lwad_experiments import Exp, Sweep


# ===========================================================================
# Stage 3 winners: stage 2 model + detection mechanism
# PLACEHOLDERS (the stage 2 defaults): set the mechanism stage 3 picked,
# s3.MEAN or s3.MAX and detach.
# ===========================================================================
# --- detector models trained and evaluated on pgd --------------------------
DET_MINI_PGD    = replace(s3.DET_MINI_PGD,    **s3.MEAN, detach=True)
FUR_MINI_PGD    = replace(s3.FUR_MINI_PGD,    **s3.MEAN, detach=True)

DET_MEDIUM_PGD  = replace(s3.DET_MEDIUM_PGD,  **s3.MEAN, detach=True)
FUR_MEDIUM_PGD  = replace(s3.FUR_MEDIUM_PGD,  **s3.MEAN, detach=True)

DET_HIGH_PGD    = replace(s3.DET_HIGH_PGD,    **s3.MEAN, detach=True)
FUR_HIGH_PGD    = replace(s3.FUR_HIGH_PGD,    **s3.MEAN, detach=True)

# --- detector models trained and evaluated on pgd_adaptive -----------------
DET_MINI_APGD   = replace(s3.DET_MINI_APGD,   **s3.MEAN, detach=True)
FUR_MINI_APGD   = replace(s3.FUR_MINI_APGD,   **s3.MEAN, detach=True)

DET_MEDIUM_APGD = replace(s3.DET_MEDIUM_APGD, **s3.MEAN, detach=True)
FUR_MEDIUM_APGD = replace(s3.FUR_MEDIUM_APGD, **s3.MEAN, detach=True)

DET_HIGH_APGD   = replace(s3.DET_HIGH_APGD,   **s3.MEAN, detach=True)
FUR_HIGH_APGD   = replace(s3.FUR_HIGH_APGD,   **s3.MEAN, detach=True)

# --- CloserAL, pgd only -----------------------------------------------------
CLO_MINI   = s3.CLO_MINI
CLO_MEDIUM = s3.CLO_MEDIUM
CLO_HIGH   = s3.CLO_HIGH


# ===========================================================================
# Axes
# ===========================================================================
TASK_ON_ADV   = [False, True]
LAMBDA_ACT    = [0.3, 0.5, 1.0, 2.0, 3.0]
LAMBDA_DET    = [0.1, 0.5, 2]   # the default 1 is in the main grid
MARGIN_FACTOR = [3.0, 5.0, 8.0] # the default 5 is in the main grid
MARGIN_LAMBDA_ACT = [1.0, 3.0]  # the margin matters once it is reached (read above)


def weights(name: str, model: lc.ModelConfig) -> list[Exp]:
    """The loss weights one winner actually uses."""
    if not model.uses_detectors:                                    # CloserAL
        return [Sweep(f"{name}/", model, lambda_act=LAMBDA_ACT)]
    if model.use_act_loss:                                          # FurtherAL
        exps = [Sweep(f"{name}/", model, lambda_act=LAMBDA_ACT,
                      task_loss_on_adv=TASK_ON_ADV),
                Sweep(f"{name}/margin/", model, margin_factor=MARGIN_FACTOR,
                      lambda_act=MARGIN_LAMBDA_ACT, task_loss_on_adv=TASK_ON_ADV)]
    else:                                                           # DetectorLayer
        exps = [Sweep(f"{name}/", model, task_loss_on_adv=TASK_ON_ADV)]
    if not model.detach:
        exps.append(Sweep(f"{name}/lambda_det/", model, lambda_det=LAMBDA_DET,
                          task_loss_on_adv=TASK_ON_ADV))
    return exps


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- DetectorLayer winners -----------------------------------------
        *weights("s4/det/pgd/mini",    DET_MINI_PGD),
        *weights("s4/det/pgd/medium",  DET_MEDIUM_PGD),
        *weights("s4/det/pgd/high",    DET_HIGH_PGD),
        *weights("s4/det/apgd/mini",   DET_MINI_APGD),
        *weights("s4/det/apgd/medium", DET_MEDIUM_APGD),
        *weights("s4/det/apgd/high",   DET_HIGH_APGD),

        # --- FurtherAL winners ---------------------------------------------
        *weights("s4/fur/pgd/mini",    FUR_MINI_PGD),
        *weights("s4/fur/pgd/medium",  FUR_MEDIUM_PGD),
        *weights("s4/fur/pgd/high",    FUR_HIGH_PGD),
        *weights("s4/fur/apgd/mini",   FUR_MINI_APGD),
        *weights("s4/fur/apgd/medium", FUR_MEDIUM_APGD),
        *weights("s4/fur/apgd/high",   FUR_HIGH_APGD),

        # --- CloserAL winners ----------------------------------------------
        *weights("s4/clo/mini",   CLO_MINI),
        *weights("s4/clo/medium", CLO_MEDIUM),
        *weights("s4/clo/high",   CLO_HIGH),

        # --- V&V -----------------------------------------------------------
        # Must match up to Adam's epsilon: detached, the detector loss feeds
        # only the detector, whose updates a constant factor does not change
        # (lambda_det=0 would, the detector would never train)
        Sweep("s4/vv/lambda_det-detached/", replace(DET_MINI_PGD, detach=True),
              lambda_det=[0.1, 1.0, 10.0]),
    ]


lx.EXPERIMENTS.extend(table())
