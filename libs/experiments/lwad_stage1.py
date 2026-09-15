"""Stage 1: baselines, verification and validation.

    python notebooks/experiment.py --only s1/ --verbose 2
"""
from __future__ import annotations

import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx
from libs.experiments.lwad_experiments import Exp


# ===========================================================================
# Presets
# ===========================================================================
COMMON = dict(epochs=10, eps=0.1, train_attack="pgd", eval_attack="pgd")

DET = lc.DetectorArchConfig(**COMMON, use_act_loss=False)    # DetectorLayer
FUR = lc.DetectorArchConfig(**COMMON, use_act_loss=True,     # FurtherAL
                            margin_factor=5.0, lambda_act=10.0)
ALI = lc.AlignmentArchConfig(**COMMON)                       # NearestAL


# ===========================================================================
# The experiments
# ===========================================================================
def table() -> list[Exp]:
    return [
        # --- baselines: one reference plus two per architecture ------------
        # No defense
        Exp("s1/base/undefended",        ALI, lambda_act=0.0,
                                              task_loss_on_adv=False),

        # --- Detector Architecture -----------------------------------------
        Exp("s1/base/detlayer",          DET),
        Exp("s1/base/detlayer-advtrain", DET, task_loss_on_adv=True),

        # --- Detector Architecture with Repulsive Activation Loss ----------
        # s1/base/further must differ from s1/base/detlayer
        Exp("s1/base/further",           FUR),
        Exp("s1/base/further-advtrain",  FUR, task_loss_on_adv=True),

        # --- Adversarial Training Architecture -----------------------------
        # The second run answers whether pulling clean and adversarial
        # activations together improves performances.
        Exp("s1/base/nearest",           ALI),
        Exp("s1/base/nearest-clean",     ALI, task_loss_on_adv=False),

        # --- V&V -----------------------------------------------------------
        # (1) Must be equal to s1/base/detlayer, metric by metric. A FurtherAL whose
        # loss is weighted zero contributes a term that is identically zero.
        Exp("s1/vv/actloss-zero",   FUR, lambda_act=0.0, margin_factor=None),

        # (2) Should go near to s1/base/detlayer too. A margin far below the
        # natural distance leaves the repulsive loss saturated at zero. Equality
        # is not guaranteed to the last digit
        Exp("s1/vv/margin-tiny",    FUR, margin_factor=0.01),

        # (3) Shape never used, so first run is cold, and the two should be 
        # identical if seeding is correctly managed.
        Exp("s1/vv/repro-cold",     FUR, hidden_dims=(64, 16), wrap_at=(0,1)),
        Exp("s1/vv/repro-warm",     FUR, hidden_dims=(64, 16), wrap_at=(0,1)),
    ]


lx.EXPERIMENTS.extend(table())
