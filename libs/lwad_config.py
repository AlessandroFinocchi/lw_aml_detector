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

# --- general ----------------------------------------------------------------
SEED = 42
DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3                   # backbone learning rate
DEFAULT_CHECKPOINT = "lwad_model.pt"

# --- detector ---------------------------------------------------------------
DEFAULT_LR_DET = 3e-3               # detector learning rate
DEFAULT_LAMBDA_DET = 1.0            # detector loss weight
DEFAULT_THRESHOLD_DET = 0.5         # detector threshold
DEFAULT_TASK_LOSS_ON_ADV = False    # true = backbone adversarial training

# --- activations ------------------------------------------------------------
DEFAULT_LAMBDA_ACT = 1.0            # activation loss weight (FurtherAL / NearestAL)
DEFAULT_ACT_MARGIN = 9e-4           # contrastive loss margin (FurtherAL)

# --- early stopping config --------------------------------------------------
DEFAULT_PATIENCE = 2                # epochs without improvements before stopping
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

    # --- attack -------------------------------------------------------------
    eps:              float = la.DEFAULT_EPS
    pgd_steps:        int = la.DEFAULT_PGD_STEPS
    pgd_alpha:        Optional[float] = None # None -> eps / 4
    pgd_evade_weight: float = la.DEFAULT_PGD_EVADE_WEIGHT
    train_attack:     str = la.DEFAULT_TRAIN_ATTACK
    eval_attack:      str = la.DEFAULT_EVAL_ATTACK

    # --- model --------------------------------------------------------------
    hidden:           int = 128                         # for default_hidden_dims()
    hidden_dims:      Optional[Tuple[int, ...]] = None  # explicit hidden widths
    wrap_at:          Optional[Tuple[int, ...]] = None  # which layers are wrapped
                                                        # (depending on arch.)
                                                        # None -> default_wrap_at()
    input_norm:       bool = True       
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

    def __post_init__(self):
        # pgd_alpha can also be passed as an argument in new
        if self.pgd_alpha is None:
            self.pgd_alpha = self.eps / 4 # amplitude of iteration step
        # accept lists too, but store tuples: must stay checkpoint-serializable
        if self.hidden_dims is not None:
            self.hidden_dims = tuple(int(w) for w in self.hidden_dims)
        if self.wrap_at is not None:
            self.wrap_at = tuple(int(i) for i in self.wrap_at)

    # --- network shape ------------------------------------------------------
    def default_hidden_dims(self) -> Tuple[int, ...]:
        """Widths derived from the `hidden` arg."""
        raise NotImplementedError("defined on concrete configs")

    def default_wrap_at(self) -> Tuple[int, ...]:
        """Which hidden layers carry the special layer."""
        raise NotImplementedError("defined on concrete configs")

    def resolved_hidden_dims(self) -> Tuple[int, ...]:
        dims = (tuple(self.hidden_dims) if self.hidden_dims is not None
                else self.default_hidden_dims())
        if not dims:
            raise ValueError("hidden_dims: at least one hidden layer is required")
        if any(w <= 0 for w in dims):
            raise ValueError(f"hidden_dims: widths must be positive, got {dims}")
        return dims

    def resolved_wrap_at(self) -> Tuple[int, ...]:
        """Corrected, sorted, de-duplicated layer indices."""
        n = len(self.resolved_hidden_dims())
        raw = self.wrap_at if self.wrap_at is not None else self.default_wrap_at()
        out = set()
        for i in raw:
            if not -n <= i < n:
                raise ValueError(
                    f"wrap_at: index {i} out of range for {n} hidden layers"
                )
            out.add(i % n)
        return tuple(sorted(out))

    @property
    def n_wrapped_layers(self) -> int:
        return len(self.resolved_wrap_at())

    def _wrap_layer(self, base: nn.Module, out_dim: int,
                    idx: int, total: int) -> nn.Module:
        return base

    # --- what every config has to build -------------------------------------
    def build_model(self, n_features: int) -> lw.LWADSequential:
        """Linear stack driven by resolved_hidden_dims() / resolved_wrap_at()."""
        dims, wrap = self.resolved_hidden_dims(), self.resolved_wrap_at()
        mods: list[nn.Module] = [nn.LayerNorm(n_features)] if self.input_norm else []
        prev, idx = n_features, 0
        for i, width in enumerate(dims):
            layer: nn.Module = nn.Linear(prev, width)
            if i in wrap:
                layer = self._wrap_layer(layer, width, idx, len(wrap))
                idx += 1
            mods += [layer, nn.ReLU()]
            prev = width
        mods.append(nn.Linear(prev, self.n_classes))
        return lw.LWADSequential(*mods)

    def build_optimizer(self, model: lw.LWADSequential) -> torch.optim.Optimizer:
        return torch.optim.Adam(model.parameters(), lr=self.lr)

    def attack_kwargs(self) -> dict:
        """Attack parameter of generate_attack method."""
        return {"steps": self.pgd_steps,
                "alpha": self.pgd_alpha,
                "evade_weight": self.pgd_evade_weight,
                "reduce": self.score_reduce}

    @property
    def uses_detectors(self) -> bool:
        return False

    @property
    def score_reduce(self) -> str:
        """How FlowState.adv_score merges the detectors classifications.
        Only needed in architectures with detectors (this check is made
        into the Exp class during validation)
        """
        return la.DEFAULT_SCORE_REDUCE


@dataclass
class DetectorArchConfig(ArchitectureConfig):
    """Architecture 1: detector on arbitrary layer, optional with contrastive 
    loss using FurtherAL (use_act_loss=False -> DetectorLayer)."""

    lr_det:          float = DEFAULT_LR_DET           # det learning rate
    lambda_det:      float = DEFAULT_LAMBDA_DET       # det loss weight
    threshold_det:   float = DEFAULT_THRESHOLD_DET    # det initial threshold
    detach:          bool = True                      # det loss doesn't affect backbone
    detector_hidden: int = 64                         # det hidden units
    detector_dims:   Optional[Tuple[int, ...]] = None # explicit detector head widths
    detector_norm:   bool = True                      # LayerNorm at the detector input
    score_reduce:    str = la.DEFAULT_SCORE_REDUCE    # "mean" | "max": how detector 
                                                      # classifications are merged
    use_act_loss:    bool = True                      # true -> FurtherAL, false -> DetectorLayer
    act_margin:      Union[float, Tuple[float, ...]] = DEFAULT_ACT_MARGIN # contrastive loss margin
                                                                          # single float for all layers
                                                                          # tuple with values for each FurtherAL

    def __post_init__(self):
        super().__post_init__()
        if self.detector_dims is not None:
            self.detector_dims = tuple(int(w) for w in self.detector_dims)
        if isinstance(self.act_margin, list):
            self.act_margin = tuple(float(m) for m in self.act_margin)
        if self.score_reduce not in lw.FlowState.REDUCE_MODES:
            raise ValueError(
                f"score_reduce: unknown value {self.score_reduce!r}, "
                f"expected one of {lw.FlowState.REDUCE_MODES}"
            )

    @property
    def uses_detectors(self) -> bool:
        return True

    def default_hidden_dims(self) -> Tuple[int, ...]:
        return (self.hidden * 2, self.hidden, 64)

    def default_wrap_at(self) -> Tuple[int, ...]:
        return (0, 2)

    def resolved_detector_dims(self) -> Tuple[int, ...]:
        if self.detector_dims is not None:
            return tuple(self.detector_dims)
        return (self.detector_hidden * 2, self.detector_hidden)

    def margin_for_layer(self, idx: int, total: int) -> float:
        """Returns FurtherAL layer margin(s): with a scalar is the same for every layer,
        with a tuple there must be exactly one value per layer (following the layers order)"""
        if isinstance(self.act_margin, (tuple, list)):
            if len(self.act_margin) != total:
                raise ValueError(
                    f"act_margin: expected {total} values (one for every FurtherAL "
                    f"network layer), received {len(self.act_margin)}"
                )
            return float(self.act_margin[idx])
        return float(self.act_margin)

    def _det_layer(self, base: nn.Module, out_dim: int,
                   idx: int = 0, total: int = 1) -> lw.DetectorLayer:
        detector = lw.build_detector(out_dim, self.resolved_detector_dims(),
                                     layer_norm=self.detector_norm)
        if self.use_act_loss:
            return lw.FurtherAL(base, detector=detector,
                                margin=self.margin_for_layer(idx, total),
                                detach=self.detach,
                                detach_reference=self.detach_reference)
        return lw.DetectorLayer(base, detector=detector, detach=self.detach)

    def _wrap_layer(self, base: nn.Module, out_dim: int,
                    idx: int, total: int) -> nn.Module:
        return self._det_layer(base, out_dim, idx=idx, total=total)

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

    def default_hidden_dims(self) -> Tuple[int, ...]:
        return (self.hidden, self.hidden, 64)

    def default_wrap_at(self) -> Tuple[int, ...]:
        return (1, 2)

    def _wrap_layer(self, base: nn.Module, out_dim: int,
                    idx: int, total: int) -> nn.Module:
        return lw.NearestAL(base, detach_reference=self.detach_reference)


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
