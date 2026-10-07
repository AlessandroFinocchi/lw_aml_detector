"""Stage 1: baselines, verification and validation.

    python notebooks/experiment.py --only s1/ --verbose 1
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
FUR = lc.DetectorModelConfig(**COMMON, use_act_loss=True)       # FurtherAL
CLO = lc.AdvTrainingModelConfig(**COMMON)                       # CloserAL


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- baselines: one reference plus two per model type --------------
        # No defense
        Exp("s1/base/undefended",        CLO, lambda_act=0.0,
                                              task_loss_on_adv=False),

        # Detector Model
        Exp("s1/base/detlayer",          DET),
        Exp("s1/base/detlayer-advtrain", DET, task_loss_on_adv=True),

        # Detector Model with Repulsive Activation Loss
        # s1/base/further must differ from s1/base/detlayer
        Exp("s1/base/further",           FUR),
        Exp("s1/base/further-advtrain",  FUR, task_loss_on_adv=True),

        # Adversarial Training Model
        # The second run answers whether pulling clean and adversarial
        # activations together improves performances.
        Exp("s1/base/advtrain",         CLO, lambda_act=0.0),
        Exp("s1/base/closer",           CLO),
        Exp("s1/base/closer-clean",     CLO, task_loss_on_adv=False),

        # --- V&V -----------------------------------------------------------
        # (1) Must be equal to s1/base/detlayer, metric by metric. A FurtherAL
        # whose loss is weighted 0 contributes a term that is identically zero.
        Exp("s1/vv/actloss-zero",   FUR, lambda_act=0.0, margin_factor=None),

        # (2) Should go near to s1/base/detlayer too. A margin far below the
        # natural distance leaves the repulsive loss saturated at zero. 
        # Equality is not guaranteed to the last digit
        Exp("s1/vv/margin-tiny",    FUR, margin_factor=0.01),

        # (3) Architectures never used, so first run is cold, and the two 
        # should be identical if seeding is correctly managed.
        Exp("s1/vv/repro-cold",     FUR, hidden_dims=(64, 16), wrap_at=(1,)),
        Exp("s1/vv/repro-warm",     FUR, hidden_dims=(64, 16), wrap_at=(1,)),

        # (4) Early exit with detector models.
        Exp("s1/vv/exit-off",       FUR, score_reduce="max"),
        Exp("s1/vv/exit-on",        FUR, score_reduce="max", early_exit=True),

        # (5) Extras for tests
        Exp("s1/mech/detlayer-wide3",    DET, hidden_dims=(128, 64, 32), wrap_at=(0,)),
        Exp("s1/mech/further-wide3",     FUR, hidden_dims=(128, 64, 32), wrap_at=(0,)),
        Exp("s1/mech/undefended-deep",   CLO, lambda_act=0.0, task_loss_on_adv=False,
                                              hidden_dims=(256, 128, 64, 32)),
        Exp("s1/mech/detlayer-deep",     DET, hidden_dims=(256, 128, 64, 32), wrap_at=(0, 1, 2, 3)),
        Exp("s1/mech/further-exit-deep", FUR, hidden_dims=(256, 128, 64, 32), wrap_at=(0, 1, 2, 3),
                                              score_reduce="max", early_exit=True),
    ]


lx.EXPERIMENTS.extend(table())
