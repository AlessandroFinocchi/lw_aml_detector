"""Stage 1: baselines, verification and validation.

    python notebooks/experiment.py --only s2/ --verbose 1
"""
from __future__ import annotations

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp, Paired, Sweep


# ===========================================================================
# Presets
# ===========================================================================
COMMON = dict(epochs=10, eps=0.2, train_attack="pgd", eval_attack="pgd",
              hidden_dims=(64, 32), wrap_at=(0,1))

DET = lc.DetectorModelConfig(**COMMON, use_act_loss=False,      # DetectorLayer
                             margin_factor=None)
FUR = lc.DetectorModelConfig(**COMMON, use_act_loss=True)       # FurtherAL
ADV = lc.AdvTrainingModelConfig(**COMMON)                       # CloserAL


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- detector model ------------------------------------------------
        Sweep("s2/det/2_layers/",         DET,
              hidden_dims=[(32,16), (64, 32), (128, 64)],
              detector_dims=[(32,16), (64,32)],
              wrap_at=[(0,), (1,), (0,1)]),
        Sweep("s2/det/3_layers/",         DET,
              hidden_dims=[(128, 64, 16), (256, 128, 32), (512, 256, 96)],
              detector_dims=[(64,32), (128,64)],
              wrap_at=[(0,), (2,), (0,2), (0,1,2)]),

        # --- detector model with repulsive activation loss -----------------
        Sweep("s2/fur/2_layers/",         FUR,
              hidden_dims=[(32,16), (64, 32), (128, 64)], 
              detector_dims=[(32,16), (64,32)],
              wrap_at=[(0,), (1,), (0,1)],),
        Sweep("s2/fur/3_layers/",         FUR,
              hidden_dims=[(128, 64, 16), (256, 128, 32), (512, 256, 96)], 
              detector_dims=[(64,32), (128,64)],
              wrap_at=[(0,), (2,), (0,2), (0,1,2)]),

        # --- adversarial training model ------------------------------------
        Sweep("s2/adv/2_layers/",         ADV,
              hidden_dims=[(48,16), (80, 32), (128, 64)], 
              wrap_at=[(0,), (1,), (0,1)],),
        Sweep("s2/adv/3_layers/",         ADV,
              hidden_dims=[(192, 64, 16), (256, 128, 32), (768, 320, 96)], 
              wrap_at=[(0,), (2,), (0,2), (0,1,2)]),
    ]


lx.EXPERIMENTS.extend(table())
