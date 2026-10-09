"""Stage 2: the network architectures.

    python scripts/experiment.py --only s2/ --dataset UNSW_BW15 --seeds 42

Runs: (26 DetectorLayer + 26 FurtherAL) x 2 attacks + 17 CloserAL = 121.
"""
from __future__ import annotations

from dataclasses import replace

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp, Sweep


# ===========================================================================
# Presets, shared by the later stages
# ===========================================================================
COMMON = dict(epochs=10, eps=0.2)

# Attacks: a model is trained and evaluated on the same one
PGD      = dict(train_attack="pgd", eval_attack="pgd")
ADAPTIVE = dict(train_attack="pgd_adaptive", eval_attack="pgd_adaptive")  # detectors only

# Activation losses of the detector models
DETLAYER = dict(use_act_loss=False, margin_factor=None)    # nothing to calibrate
FURTHER  = dict(use_act_loss=True, margin_factor=lc.DEFAULT_MARGIN_FACTOR)

DET = lc.DetectorModelConfig(**COMMON, **PGD, **DETLAYER)  # DetectorLayer
FUR = lc.DetectorModelConfig(**COMMON, **PGD, **FURTHER)   # FurtherAL
CLO = lc.AdvTrainingModelConfig(**COMMON, **PGD)           # CloserAL


# ===========================================================================
# Shape grids
# ===========================================================================
DET_2_LAYERS = dict(hidden_dims=[(32, 16), (64, 32), (128, 64)],  # 18 runs
                    detector_dims=[(32, 16), (64, 32)],
                    wrap_at=[(0,), (1,), (0, 1)])
DET_3_LAYERS = dict(hidden_dims=[(128, 64, 16), (256, 128, 32)],  #  8 runs
                    detector_dims=[(64, 32)],
                    wrap_at=[(0,), (2,), (0, 2), (0, 1, 2)])

# Wider backbones having no detector (see scripts/params_calc.py)
CLO_2_LAYERS = dict(hidden_dims=[(48, 16), (80, 32), (160, 64)],  #  9 runs
                    wrap_at=[(0,), (1,), (0, 1)])
CLO_3_LAYERS = dict(hidden_dims=[(192, 64, 16), (256, 128, 32)],  #  8 runs
                    wrap_at=[(0,), (2,), (0, 2), (0, 1, 2)])


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- DetectorLayer -------------------------------------------------
        Sweep("s2/det/pgd/2_layers/",      DET,                      **DET_2_LAYERS),
        Sweep("s2/det/pgd/3_layers/",      DET,                      **DET_3_LAYERS),
        Sweep("s2/det/adaptive/2_layers/", replace(DET, **ADAPTIVE), **DET_2_LAYERS),
        Sweep("s2/det/adaptive/3_layers/", replace(DET, **ADAPTIVE), **DET_3_LAYERS),

        # --- FurtherAL -----------------------------------------------------
        Sweep("s2/fur/pgd/2_layers/",      FUR,                      **DET_2_LAYERS),
        Sweep("s2/fur/pgd/3_layers/",      FUR,                      **DET_3_LAYERS),
        Sweep("s2/fur/adaptive/2_layers/", replace(FUR, **ADAPTIVE), **DET_2_LAYERS),
        Sweep("s2/fur/adaptive/3_layers/", replace(FUR, **ADAPTIVE), **DET_3_LAYERS),

        # --- CloserAL ------------------------------------------------------
        Sweep("s2/clo/pgd/2_layers/",      CLO,                      **CLO_2_LAYERS),
        Sweep("s2/clo/pgd/3_layers/",      CLO,                      **CLO_3_LAYERS),
    ]


lx.EXPERIMENTS.extend(table())
