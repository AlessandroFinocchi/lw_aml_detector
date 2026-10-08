"""Stage 4: the loss weights of the stage 3 winners.

    python notebooks/experiment.py --only s4/ --dataset UNSW_BW15 --seeds 42

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

Runs with the placeholder winners: 3x2 + 3x(14+8) + 3x7 + 3 = 96; each
detector winner with detach=False adds 8.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
import libs.experiments.lwad_stage3 as s3
from libs.experiments.lwad_experiments import Exp, Sweep


# ===========================================================================
# Stage 3 winners: stage 2 model + detection mechanism
# PLACEHOLDERS (the stage 2 defaults): set the mechanism stage 3 picked.
# A detector winner may switch loss: use the s3.DETLAYER / s3.FURTHER bundle.
# ===========================================================================
DET_MINI   = replace(s3.DET_MINI,   **s3.DETLAYER, **s3.MEAN, detach=True)
DET_MEDIUM = replace(s3.DET_MEDIUM, **s3.DETLAYER, **s3.MEAN, detach=True)
DET_HIGH   = replace(s3.DET_HIGH,   **s3.DETLAYER, **s3.MEAN, detach=True)

FUR_MINI   = replace(s3.FUR_MINI,   **s3.FURTHER,  **s3.MEAN, detach=True)
FUR_MEDIUM = replace(s3.FUR_MEDIUM, **s3.FURTHER,  **s3.MEAN, detach=True)
FUR_HIGH   = replace(s3.FUR_HIGH,   **s3.FURTHER,  **s3.MEAN, detach=True)

CLO_MINI   = replace(s3.CLO_MINI,   detach_reference=False)
CLO_MEDIUM = replace(s3.CLO_MEDIUM, detach_reference=False)
CLO_HIGH   = replace(s3.CLO_HIGH,   detach_reference=False)


# ===========================================================================
# Axes
# ===========================================================================
TASK_ON_ADV   = [False, True]
LAMBDA_ACT    = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]  # 0: no activation loss
LAMBDA_DET    = [0.1, 0.5, 5.0, 10.0]                  # the default 1 is in the main grid
MARGIN_FACTOR = [2.0, 10.0]                            # the default 5 is in the main grid
MARGIN_LAMBDA_ACT = [1.0, 10.0]  # from LAMBDA_ACT: the margin matters once it is reached


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
        *weights("s4/det/mini",   DET_MINI),
        *weights("s4/det/medium", DET_MEDIUM),
        *weights("s4/det/high",   DET_HIGH),

        # --- FurtherAL winners ---------------------------------------------
        *weights("s4/fur/mini",   FUR_MINI),
        *weights("s4/fur/medium", FUR_MEDIUM),
        *weights("s4/fur/high",   FUR_HIGH),

        # --- CloserAL winners ----------------------------------------------
        *weights("s4/clo/mini",   CLO_MINI),
        *weights("s4/clo/medium", CLO_MEDIUM),
        *weights("s4/clo/high",   CLO_HIGH),

        # --- V&V -----------------------------------------------------------
        # Must match up to Adam's epsilon: detached, the detector loss feeds
        # only the detector, whose updates a constant factor does not change
        # (lambda_det=0 would, the detector would never train)
        Sweep("s4/vv/lambda_det-detached/", replace(DET_MINI, detach=True),
              lambda_det=[0.1, 1.0, 10.0]),
    ]


lx.EXPERIMENTS.extend(table())
