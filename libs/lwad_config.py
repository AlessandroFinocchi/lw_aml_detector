# ===========================================================================
# cfg = DetectorArchConfig(...)   # architecture 1: detector-based (+ FurtherAL)
# cfg = AlignmentArchConfig(...)  # architecture 2: adv training (NearestAL)
# built = create_architecture(cfg, n_features, device)
# ===========================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import libs.lwad_wrapper as lw
import libs.lwad_attack as la


# ===========================================================================
# Default config
# ===========================================================================

# --- training config -------------------------------------------------------
DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3                   # backbone learning rate
DEFAULT_LR_DET = 3e-3               # detector learning rate
DEFAULT_LAMBDA_DET = 1.0            # detector loss weight
DEFAULT_LAMBDA_ACT = 1.0            # activation loss weight (FurtherAL / NearestAL)
DEFAULT_ACT_MARGIN = 9e-4           # FurtherAL contrastive loss margin
DEFAULT_TASK_LOSS_ON_ADV = False    # true = backbone adversarial training
DEFAULT_THRESHOLD = 0.5             # starting threshold
SEED = 42
DEFAULT_CHECKPOINT = "lwad_model.pt"

# --- early stopping config --------------------------------------------------
DEFAULT_PATIENCE = 5                # epochs without improvements before stopping
DEFAULT_MIN_DELTA = 1e-4            # minimum improvement for patience reset


# ===========================================================================
# Config objects: each one represent an experiment
# ===========================================================================
@dataclass
class ArchitectureConfig:
    """ Common class: hyperparameters shared between architectures """

    # --- training -----------------------------------------------------------
    epochs:           int = DEFAULT_EPOCHS
    batch_size:       int = DEFAULT_BATCH_SIZE
    lr:               float = DEFAULT_LR # backbone learning rate
    task_loss_on_adv: bool = DEFAULT_TASK_LOSS_ON_ADV
    checkpoint:       str = DEFAULT_CHECKPOINT
    patience:         int = DEFAULT_PATIENCE
    min_delta:        float = DEFAULT_MIN_DELTA

    # --- attacco ------------------------------------------------------------
    eps:              float = la.DEFAULT_EPS
    pgd_steps:        int = la.DEFAULT_PGD_STEPS
    pgd_alpha:        Optional[float] = None # None -> eps / 4
    pgd_evade_weight: float = la.DEFAULT_PGD_EVADE_WEIGHT
    train_attack:     str = la.DEFAULT_TRAIN_ATTACK
    eval_attack:      str = la.DEFAULT_EVAL_ATTACK

    # --- modello ------------------------------------------------------------
    hidden:           int = 128
    n_classes:        int = 2

    # --- activation loss ----------------------------------------------------
    lambda_act:       float = DEFAULT_LAMBDA_ACT
    detach_reference: Optional[bool] = None # None: each activation loss uses its
                                            #       own default (True for FurtherAL
                                            #       and False for NearestAL)

    # abstract class
    def __new__(cls, *args, **kwargs):
        if cls is ArchitectureConfig:
            raise TypeError("ArchitectureConfig is abstract, use a specialized class")
        return super().__new__(cls)

    # pgd_alpha can also be passed as an argument in new
    def __post_init__(self):
        if self.pgd_alpha is None:
            self.pgd_alpha = self.eps / 4 # amplitude of iteration step

    # --- what every config has to build -------------------------------------
    def build_model(self, n_features: int) -> lw.LWADSequential:
        raise NotImplementedError("defined on concrete configs")

    def build_optimizer(self, model: lw.LWADSequential) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def attack_kwargs(self) -> dict:
        """Attack parameter of generate_attack method."""
        return {"steps": self.pgd_steps, 
                "alpha": self.pgd_alpha,
                "evade_weight": self.pgd_evade_weight}

    @property
    def uses_detectors(self) -> bool:
        return False


@dataclass
class DetectorArchConfig(ArchitectureConfig):
    """Architecture 1: detector on arbitrary layer, optional with contrastive 
    loss using FurtherAL (use_act_loss=False -> DetectorLayer)."""

    lr_det:          float = DEFAULT_LR_DET     # detector learning rate
    lambda_det:      float = DEFAULT_LAMBDA_DET # detector loss weight
    threshold:       float = DEFAULT_THRESHOLD  # detector initial threshold
    detach:          bool = True                # detector loss doesn't affect backbone
    detector_hidden: int = 64                   # detector hidden units
    use_act_loss:    bool = True                # true -> FurtherAL, false -> DetectorLayer
    act_margin:      Union[float, Tuple[float, ...]] = DEFAULT_ACT_MARGIN # contrastive loss margin
                                                                          # single float for all layers
                                                                          # tuple with values for each FurtherAL

    @property
    def uses_detectors(self) -> bool:
        return True

    def margin_for_layer(self, idx: int, total: int) -> float:
        """Returns FurtherAL layer margin(s): with a scalar is the same for every layer,
        with a tuple there must be exactly one value per layer (following the layers order)"""
        if isinstance(self.act_margin, (tuple, list)):
            if len(self.act_margin) != total:
                raise ValueError(
                    f"act_margin: expected {total} values (one for every FurtherAL "
                    f"network layer, received {len(self.act_margin)}"
                )
            return float(self.act_margin[idx])
        return float(self.act_margin)

    def _det_layer(self, base: nn.Module, out_dim: int,
                   idx: int = 0, total: int = 1) -> lw.DetectorLayer:
        detector = lw.default_detector(out_dim, self.detector_hidden)
        if self.use_act_loss:
            return lw.FurtherAL(base, detector=detector,
                                margin=self.margin_for_layer(idx, total),
                                detach=self.detach,
                                detach_reference=self.detach_reference)
        return lw.DetectorLayer(base, detector=detector, detach=self.detach)

    def build_model(self, n_features: int) -> lw.LWADSequential:
        h = self.hidden
        return lw.LWADSequential(
            nn.LayerNorm(n_features),
            self._det_layer(nn.Linear(n_features, h*2), h*2, idx=0, total=2), nn.ReLU(),
            nn.Linear(h*2, h), nn.ReLU(),
            self._det_layer(nn.Linear(h, 64), 64, idx=1, total=2), nn.ReLU(),
            nn.Linear(64, self.n_classes),
        )

    def build_optimizer(self, model: lw.LWADSequential) -> torch.optim.Optimizer:
        return torch.optim.Adam(params=[
            {"params": list(model.backbone_parameters()), "lr": self.lr},
            {"params": list(model.detector_parameters()), "lr": self.lr_det},
        ])


@dataclass
class AlignmentArchConfig(ArchitectureConfig):
    """Architecture 2: adversarial training via NearestAL, no detector. Only 
    PassThrough e NearestAL. Task loss on adversarial sample is active by default"""

    task_loss_on_adv: bool = True # base default override

    def build_model(self, n_features: int) -> lw.LWADSequential:
        h = self.hidden
        def al(base: nn.Module) -> lw.NearestAL:
            return lw.NearestAL(base, detach_reference=self.detach_reference)
        return lw.LWADSequential(
            nn.Linear(n_features, h), nn.ReLU(),
            al(nn.Linear(h, h)), nn.ReLU(),
            al(nn.Linear(h, 64)), nn.ReLU(),
            nn.Linear(64, self.n_classes),
        )


# ===========================================================================
# Factory
# ===========================================================================
@dataclass
class BuiltArchitecture:
    model: lw.LWADSequential
    optimizer: torch.optim.Optimizer
    config: ArchitectureConfig


def create_architecture(config: ArchitectureConfig, n_features: int,
                        device: str = "cpu") -> BuiltArchitecture:
    """Builds model and optimizer depending on the given config.
    Architecture coherency is checked within DetectorSequential."""
    model = config.build_model(n_features).to(device)
    optimizer = config.build_optimizer(model)
    return BuiltArchitecture(model=model, optimizer=optimizer, config=config)