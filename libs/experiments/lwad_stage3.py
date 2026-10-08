"""Stage 3: the detection mechanism of the stage 2 winners.

    python notebooks/experiment.py --only s3/ --dataset UNSW_BW15 --seeds 42

Attack and training budget are those of stages 1-2. Every detector winner
(mini, medium, high of DetectorLayer and FurtherAL) gets every combination of

    loss    detlayer | further     use_act_loss, crossed with the family too:
                                   the loss is compared at equal shape
    reduce  mean | max             max also exits early: same flags, less latency
    detach  True | False

A full factorial rather than one axis at a time: with a single seed each main
effect is then a mean over the other axes (4 runs against 4, not 1 against 1),
and the interactions come out of the same runs. A model with one detector has
nothing to reduce (mean == max) and keeps mean only. CloserAL has no detector:
its one mechanism is detach_reference.

Runs: 6 detector winners x 8 + 3 CloserAL x 2 = 54.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp, Sweep


# ===========================================================================
# Presets
# ===========================================================================
COMMON = dict(epochs=10, eps=0.2, train_attack="pgd", eval_attack="pgd")

# Mechanisms as bundles of overrides: a choice spanning two fields (FurtherAL
# and its calibrated margin, max and its early exit) stays a single choice
DETLAYER = dict(use_act_loss=False, margin_factor=None)    # nothing to calibrate
FURTHER  = dict(use_act_loss=True, margin_factor=lc.DEFAULT_MARGIN_FACTOR)
MEAN     = dict(score_reduce="mean")
MAX      = dict(score_reduce="max", early_exit=True)

DET = lc.DetectorModelConfig(**COMMON, **DETLAYER)         # DetectorLayer
FUR = lc.DetectorModelConfig(**COMMON, **FURTHER)          # FurtherAL
CLO = lc.AdvTrainingModelConfig(**COMMON)                  # CloserAL


# ===========================================================================
# Stage 2 winners
# PLACEHOLDERS (the smallest stage 2 candidates): swap in the real winners.
# Only the shape is set here, everything else comes from the preset.
# ===========================================================================
DET_MINI   = replace(DET, hidden_dims=(32, 16),  detector_dims=(32, 16), wrap_at=(0, 1))
DET_MEDIUM = replace(DET, hidden_dims=(64, 32),  detector_dims=(32, 16), wrap_at=(0, 1))
DET_HIGH   = replace(DET, hidden_dims=(128, 64), detector_dims=(32, 16), wrap_at=(0, 1))

FUR_MINI   = replace(FUR, hidden_dims=(32, 16),  detector_dims=(64, 32), wrap_at=(0, 1))
FUR_MEDIUM = replace(FUR, hidden_dims=(64, 32),  detector_dims=(64, 32), wrap_at=(0, 1))
FUR_HIGH   = replace(FUR, hidden_dims=(128, 64), detector_dims=(64, 32), wrap_at=(0, 1))

CLO_MINI   = replace(CLO, hidden_dims=(48, 16),  wrap_at=(0, 1))
CLO_MEDIUM = replace(CLO, hidden_dims=(80, 32),  wrap_at=(0, 1))
CLO_HIGH   = replace(CLO, hidden_dims=(128, 64), wrap_at=(0, 1))


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


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- DetectorLayer winners -----------------------------------------
        *mechanisms("s3/det/mini",   DET_MINI),
        *mechanisms("s3/det/medium", DET_MEDIUM),
        *mechanisms("s3/det/high",   DET_HIGH),

        # --- FurtherAL winners ---------------------------------------------
        *mechanisms("s3/fur/mini",   FUR_MINI),
        *mechanisms("s3/fur/medium", FUR_MEDIUM),
        *mechanisms("s3/fur/high",   FUR_HIGH),

        # --- CloserAL winners ----------------------------------------------
        # No detector: the mechanism is whether the clean activations are a
        # fixed anchor the adversarial ones are pulled to
        Sweep("s3/clo/mini/",   CLO_MINI,   detach_reference=DETACH_REFERENCE),
        Sweep("s3/clo/medium/", CLO_MEDIUM, detach_reference=DETACH_REFERENCE),
        Sweep("s3/clo/high/",   CLO_HIGH,   detach_reference=DETACH_REFERENCE),
    ]


lx.EXPERIMENTS.extend(table())
