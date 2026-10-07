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
COMMON = dict(epochs=10, eps=0.2, train_attack="pgd", eval_attack="pgd")
              

DET = lc.DetectorModelConfig(**COMMON, use_act_loss=False,      # DetectorLayer
                             margin_factor=None)
FUR = lc.DetectorModelConfig(**COMMON, use_act_loss=True)       # FurtherAL
CLO = lc.AdvTrainingModelConfig(**COMMON)                       # CloserAL


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        Sweep("s3/clo/2_layers/", CLO,
              hidden_dims=[(32,16), (64, 32)],
              wrap_at=[(0,), (0,1)],
              detach_reference=[True,False],
        ),
        Sweep("s3/clo/3_layers/", CLO,
              hidden_dims=[(128, 64, 16)],
              wrap_at=[(0,),(0,2)],
              detach_reference=[True,False],
        ),
    ]


lx.EXPERIMENTS.extend(table())
