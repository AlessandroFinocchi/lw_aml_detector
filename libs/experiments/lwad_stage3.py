"""Stage 3: the detection mechanism of the stage 2 winners.

    python scripts/experiment.py --only s3/ --dataset UNSW_BW15 --seeds 42

Model types and training budget come from stage 2. For each attack and size, the
DetectorLayer and FurtherAL winners get every combination of

    loss    detlayer | further
    reduce  mean | max             max also exits early: same flags, less latency
    detach  True | False

Runs: pgd 4+4+8, apgd 8+8+(8+4), CloserAL 3x2 = 50.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp, Sweep
from libs.experiments.lwad_stage2 import (APGD, CLO, DET, DETLAYER, FUR,
                                          FURTHER, PGD)


# ===========================================================================
# Presets: model types, losses and attacks come from stage 2
# ===========================================================================
# Reduce modes as bundles: max and its early exit stay a single choice
MEAN = dict(score_reduce="mean")
MAX  = dict(score_reduce="max", early_exit=True)


# ===========================================================================
# Stage 2 winners, per attack the detector models were trained and evaluated on
# ===========================================================================
DET_MINI_PGD   = replace(DET, **PGD, hidden_dims=(32, 16),       detector_dims=(32, 16), wrap_at=(0,))
DET_MEDIUM_PGD = replace(DET, **PGD, hidden_dims=(128, 64),      detector_dims=(32, 16), wrap_at=(0,))
DET_HIGH_PGD   = replace(DET, **PGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 2))
FUR_MINI_PGD   = replace(FUR, **PGD, hidden_dims=(32, 16),       detector_dims=(32, 16), wrap_at=(0,))
FUR_MEDIUM_PGD = replace(FUR, **PGD, hidden_dims=(128, 64),      detector_dims=(32, 16), wrap_at=(0,))
FUR_HIGH_PGD   = replace(FUR, **PGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 2))

DET_MINI_APGD   = replace(DET, **APGD, hidden_dims=(48, 16),       detector_dims=(32, 16), wrap_at=(0, 1))
DET_MEDIUM_APGD = replace(DET, **APGD, hidden_dims=(160, 64),      detector_dims=(64, 32), wrap_at=(0, 1))
DET_HIGH_APGD   = replace(DET, **APGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 1, 2))
FUR_MINI_APGD   = replace(FUR, **APGD, hidden_dims=(48, 16),       detector_dims=(32, 16), wrap_at=(0, 1))
FUR_MEDIUM_APGD = replace(FUR, **APGD, hidden_dims=(160, 64),      detector_dims=(64, 32), wrap_at=(0, 1))
FUR_HIGH_APGD   = replace(FUR, **APGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0,))

CLO_MINI   = replace(CLO, hidden_dims=(48, 16),       wrap_at=(0,))
CLO_MEDIUM = replace(CLO, hidden_dims=(160, 64),      wrap_at=(0,))
CLO_HIGH   = replace(CLO, hidden_dims=(256, 128, 32), wrap_at=(0,))


# ===========================================================================
# Axes
# ===========================================================================
LOSSES  = {"detlayer": DETLAYER, "further": FURTHER}
REDUCES = {"mean": MEAN, "max": MAX}
DETACH  = [True, False]
DETACH_REFERENCE = [False, True]    # False = the stage 2 winner, as reference


def mechanisms(name: str, model: lc.DetectorModelConfig) -> list[Exp]:
    """Every detection mechanism on one model: loss x reduce x detach."""
    reduces = REDUCES if model.n_wrapped_layers > 1 else {"mean": MEAN}
    return [Sweep(f"{name}/{loss}-{red}/", replace(model, **LOSSES[loss], **reduces[red]),
                  detach=DETACH)
            for loss in LOSSES for red in reduces]


def winners(name: str, det: lc.DetectorModelConfig,
            fur: lc.DetectorModelConfig) -> list[Exp]:
    """The mechanisms of the DetectorLayer and FurtherAL winners of one size:
    once if they share their shape (the loss being an axis, their runs would
    coincide), on each shape otherwise."""
    if replace(det, **FURTHER) == fur:
        return mechanisms(name, det)
    return mechanisms(f"{name}/det-shape", det) + mechanisms(f"{name}/fur-shape", fur)


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- detector winners on pgd ---------------------------------------
        *winners("s3/pgd/mini",    DET_MINI_PGD,    FUR_MINI_PGD),
        *winners("s3/pgd/medium",  DET_MEDIUM_PGD,  FUR_MEDIUM_PGD),
        *winners("s3/pgd/high",    DET_HIGH_PGD,    FUR_HIGH_PGD),

        # --- detector winners on pgd_adaptive ------------------------------
        *winners("s3/apgd/mini",   DET_MINI_APGD,   FUR_MINI_APGD),
        *winners("s3/apgd/medium", DET_MEDIUM_APGD, FUR_MEDIUM_APGD),
        *winners("s3/apgd/high",   DET_HIGH_APGD,   FUR_HIGH_APGD),

        # --- CloserAL winners ----------------------------------------------
        # No detector: the mechanism is whether the clean activations are a
        # fixed anchor the adversarial ones are pulled to
        Sweep("s3/clo/mini/",   CLO_MINI,   detach_reference=DETACH_REFERENCE),
        Sweep("s3/clo/medium/", CLO_MEDIUM, detach_reference=DETACH_REFERENCE),
        Sweep("s3/clo/high/",   CLO_HIGH,   detach_reference=DETACH_REFERENCE),
    ]


lx.EXPERIMENTS.extend(table())
