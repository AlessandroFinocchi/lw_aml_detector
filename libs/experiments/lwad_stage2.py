"""Stage 1: baselines, verification and validation.

    python notebooks/experiment.py --only s2/ --verbose 1
"""
from __future__ import annotations

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp


# ===========================================================================
# Presets
# ===========================================================================
COMMON = dict(epochs=10, eps=0.2, train_attack="pgd", eval_attack="pgd",
              hidden_dims=(64, 32), wrap_at=(0,1))

DET = lc.DetectorModelConfig(**COMMON, use_act_loss=False)      # DetectorLayer
FUR = lc.DetectorModelConfig(**COMMON, use_act_loss=True,       # FurtherAL
                            margin_factor=5.0, lambda_act=10.0)
ALI = lc.AdvTrainingModelConfig(**COMMON)                       # CloserAL


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- detector model ------------------------------------------------
        Exp("s2/det/1",        DET, hidden_dims=(64, 16), wrap_at=(1,)),

        # --- detector model with repulsive activation loss -----------------

        # --- adversarial training model ------------------------------------
    ]


lx.EXPERIMENTS.extend(table())
