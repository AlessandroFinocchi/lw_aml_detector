"""Stage 3: the detection mechanism of the stage 2 winners.

    python scripts/experiment.py --only s3/ --dataset UNSW_BW15 --seeds 42

Model types and training budget come from stage 2. Every DetectorLayer and
FurtherAL winner, on each attack and size, gets every combination of

    reduce  mean | max       max also exits early: same flags, less latency
    detach  true | false

A model with one detector has nothing to reduce (mean == max) and keeps mean.

Runs: pgd (2+2) + (2+2) + (4+4), apgd (4+4) + (4+4) + (4+2), CloserAL (3) = 41.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp, Sweep
from libs.experiments.lwad_stage2 import APGD, CLO, DET, FUR, PGD


# ===========================================================================
# Presets
# ===========================================================================
# Reduce modes as bundles: max and its early exit stay a single choice
MEAN = dict(score_reduce="mean")
MAX  = dict(score_reduce="max", early_exit=True)

# ===========================================================================
# Axes
# ===========================================================================
REDUCES = {"mean": MEAN, "max": MAX}
DETACH  = [True, False]


# ===========================================================================
# Stage 2 winners
# ===========================================================================
# --- detector models trained and evaluated on pgd --------------------------
DET_MINI_PGD    = replace(DET, **PGD,  hidden_dims=(32, 16),       detector_dims=(32, 16), wrap_at=(0,))
FUR_MINI_PGD    = replace(FUR, **PGD,  hidden_dims=(32, 16),       detector_dims=(32, 16), wrap_at=(0,))

DET_MEDIUM_PGD  = replace(DET, **PGD,  hidden_dims=(128, 64),      detector_dims=(32, 16), wrap_at=(0,))
FUR_MEDIUM_PGD  = replace(FUR, **PGD,  hidden_dims=(128, 64),      detector_dims=(32, 16), wrap_at=(0,))

DET_HIGH_PGD    = replace(DET, **PGD,  hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 2))
FUR_HIGH_PGD    = replace(FUR, **PGD,  hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 2))

# --- detector models trained and evaluated on pgd_adaptive -----------------
DET_MINI_APGD   = replace(DET, **APGD, hidden_dims=(48, 16),       detector_dims=(32, 16), wrap_at=(0, 1))
FUR_MINI_APGD   = replace(FUR, **APGD, hidden_dims=(48, 16),       detector_dims=(32, 16), wrap_at=(0, 1))

DET_MEDIUM_APGD = replace(DET, **APGD, hidden_dims=(160, 64),      detector_dims=(64, 32), wrap_at=(0, 1))
FUR_MEDIUM_APGD = replace(FUR, **APGD, hidden_dims=(160, 64),      detector_dims=(64, 32), wrap_at=(0, 1))

DET_HIGH_APGD   = replace(DET, **APGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0, 1, 2))
FUR_HIGH_APGD   = replace(FUR, **APGD, hidden_dims=(256, 128, 32), detector_dims=(64, 32), wrap_at=(0,))

# --- CloserAL, pgd only -----------------------------------------------------
CLO_MINI   = replace(CLO, hidden_dims=(48, 16),       wrap_at=(0,))
CLO_MEDIUM = replace(CLO, hidden_dims=(160, 64),      wrap_at=(0,))
CLO_HIGH   = replace(CLO, hidden_dims=(256, 128, 32), wrap_at=(0,))


def mechanisms(name: str, model: lc.DetectorModelConfig) -> list[Exp]:
    """Every detection mechanism of one model, its loss unchanged: 
        reduce x detach. 
    
    E.g. mechanisms("s3/det/apgd/mini", DET_MINI_APGD) gives
        s3/det/apgd/mini/mean/[detach=True]    s3/det/apgd/mini/mean/[detach=False]
        s3/det/apgd/mini/max/[detach=True]     s3/det/apgd/mini/max/[detach=False]
    and the max runs are left out for a model with one detector."""
    reduces = REDUCES if model.n_wrapped_layers > 1 else {"mean": MEAN}
    return [Sweep(f"{name}/{red}/", replace(model, **reduces[red]), detach=DETACH)
            for red in reduces]


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- DetectorLayer winners -----------------------------------------
        *mechanisms("s3/det/pgd/mini",    DET_MINI_PGD),
        *mechanisms("s3/det/pgd/medium",  DET_MEDIUM_PGD),
        *mechanisms("s3/det/pgd/high",    DET_HIGH_PGD),
        *mechanisms("s3/det/apgd/mini",   DET_MINI_APGD),
        *mechanisms("s3/det/apgd/medium", DET_MEDIUM_APGD),
        *mechanisms("s3/det/apgd/high",   DET_HIGH_APGD),

        # --- FurtherAL winners ---------------------------------------------
        *mechanisms("s3/fur/pgd/mini",    FUR_MINI_PGD),
        *mechanisms("s3/fur/pgd/medium",  FUR_MEDIUM_PGD),
        *mechanisms("s3/fur/pgd/high",    FUR_HIGH_PGD),
        *mechanisms("s3/fur/apgd/mini",   FUR_MINI_APGD),
        *mechanisms("s3/fur/apgd/medium", FUR_MEDIUM_APGD),
        *mechanisms("s3/fur/apgd/high",   FUR_HIGH_APGD),

        # --- CloserAL winners ----------------------------------------------
        Exp("s3/clo/mini",   CLO_MINI),
        Exp("s3/clo/medium", CLO_MEDIUM),
        Exp("s3/clo/high",   CLO_HIGH),
    ]


lx.EXPERIMENTS.extend(table())
