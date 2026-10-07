from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, Optional


# ===========================================================================
# FlowState: flows through the network accumulating detections and losses
# ===========================================================================
class FlowState:

    def __init__(self, is_adv=None, labels=None):
        #self.labels = labels
        self.is_adv = is_adv
        self.detections: list[torch.Tensor] = []
        self.det_loss: Optional[torch.Tensor] = None # detector loss (BCE)
        self.act_loss: Optional[torch.Tensor] = None # activation loss (Further/Closer)
        self.exit_layer: Optional[int] = None        # layer the forward stopped at (early exit)

    # --- detection (for DetectorLayer) -------------------------------------
    def add_detection(self, logit: torch.Tensor) -> None:
        self.detections.append(logit)
        if self.is_adv is not None:
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1), self.is_adv.float()
            )
            self.det_loss = loss if self.det_loss is None else self.det_loss + loss

    # --- activation loss (for FurtherAL and CloserAL) ----------------------
    def add_act_loss(self, loss: torch.Tensor) -> None:
        self.act_loss = loss if self.act_loss is None else self.act_loss + loss

    def split_pairs(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """"Divides activations in pairs (real, adv). Relies on
                batch = cat([x, x_adv])
        """
        real = y[self.is_adv == 0]
        adv = y[self.is_adv == 1]
        if len(real) != len(adv):
            raise ValueError(
                f"split_pairs: {len(real)} real sample vs {len(adv)} adv, "
                "batch has to contain every sample in both versions"
            )
        return real, adv

    REDUCE_MODES = ("mean", "max")

    def adv_score(self, reduce: str = "mean") -> Optional[torch.Tensor]:
        """reduce can be:
                "mean": averages all detectors scores
                "max" : alerts if at least one detector result is adversarial
        Returns None if the network doesn't contain any detector."""
        if reduce not in self.REDUCE_MODES:
            raise ValueError(f"adv_score: unknown reduce {reduce!r}, "
                             f"expected one of {self.REDUCE_MODES}")
        if not self.detections:
            return None

        probs = torch.sigmoid(torch.cat(self.detections, dim=-1))
        return probs.max(dim=-1).values if reduce == "max" else probs.mean(dim=-1)


# ===========================================================================
# Detector build utilities
# ===========================================================================
def build_detector(in_dim: int, dims: Iterable[int] = (128, 64),
                   layer_norm: bool = True) -> nn.Module:
    """Builds a detector for real/adv classification. The detector module
    architecture must stay the following for the state_dict keys
    
    activations --> [LayerNorm] --> Linear  --> ReLU 
     in_dim         optional       (dims[1])         
                                --> Linear  --> ReLU 
                                   (dims[2])                  
                                --> ... 
                                --> Linear  --> ReLU 
                                   (dims[n])                 
                                --> Linear --> logit
                                    (1)
    Args:
        in_dim: the input dimension
        dims: the dimensions of internal Linear layers
        layer_norm: if true starts with a LayerNorm
    """
    mods: list[nn.Module] = [nn.LayerNorm(in_dim)] if layer_norm else []
    prev = in_dim
    for h in dims:
        mods += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    mods.append(nn.Linear(prev, 1))
    return nn.Sequential(*mods)

def default_detector(in_dim: int, hidden: int = 64) -> nn.Module:
    return build_detector(in_dim, (hidden*2, hidden))


# ===========================================================================
# Layers class diagram
#
#                     PassThrough
#                    /           \
#             DetectorLayer   ActivationLoss
#                    \          <abstract>
#                     \         /        \
#                      FurtherAL          CloserAL
#
# Forward pass is defined once in PassThrough; subclasses contribute
# overriding the collect() hook and combining with super().collect().
# Thus, FurtherAL(DetectorLayer, ActivationLoss) executes automatically 
# both the detection and the contrastive loss.
# ===========================================================================
class PassThrough(nn.Module):
    """(x, state) -> (base(x), state). State propagates unchanged;
    Subclasses add their contributions via collect hook"""

    def __init__(self, base: nn.Module, **kwargs):
        super().__init__(**kwargs)
        self.base = base

    def forward(self, x, state: FlowState):
        y = self.base(x)
        self.collect(x, y, state)
        return y, state

    def collect(self, x, y, state: FlowState) -> None:
        pass


class DetectorLayer(PassThrough):
    """Adds real/adv classification detector on layer activations.

    detach can be:
            True: detector loss updates only detector parameters
            False: detector loss updates also the backbone, 
                   making activations easier recognizable
    """

    def __init__(self, base: nn.Module, detector: nn.Module,
                 detach: bool = True, **kwargs):
        super().__init__(base, **kwargs)
        self.detector = detector
        self.detach = detach

    def collect(self, x, y, state: FlowState) -> None:
        feats = y.detach() if self.detach else y
        state.add_detection(self.detector(feats))  # real/adv classification
        super().collect(x, y, state)


class ActivationLoss(PassThrough):
    """
    Abstract class, subclasses define only distance_to_loss(d)

    The distance is relative for both losses: with an absolute one the
    cheapest way to satisfy either loss is to rescale every activation.

    - enabled: enables loss without changing model type
    - detach_reference: if true, real activations are treated as a fixed
                        anchor and grad only moves adv activations.
                        None -> DETACH_REFERENCE_DEFAULT of the subclass.
    """

    # overridden per loss type
    DETACH_REFERENCE_DEFAULT: bool = False

    # keeps the relative distance finite when the real activations vanish
    SCALE_EPS: float = 1e-8

    def __init__(self, base: nn.Module, enabled: bool = True,
                 detach_reference: Optional[bool] = None, **kwargs):
        super().__init__(base, **kwargs)
        self.enabled = enabled
        self.detach_reference = (self.DETACH_REFERENCE_DEFAULT
                                 if detach_reference is None else detach_reference)

    def collect(self, x, y, state: FlowState) -> None:
        # in evaluation / attack generation is_adv is not passed, no loss
        if self.enabled and state.is_adv is not None:
            real, adv = state.split_pairs(y)
            ref = real.detach() if self.detach_reference else real
            state.add_act_loss(self.distance_to_loss(self.distance(adv, ref)))
        super().collect(x, y, state)

    def distance(self, adv: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Per-pair mean squared gap over the batch mean squared activation.
        Normalized by the batch, not per pair: a single sample with 
        near-zero activations would otherwise dominate the loss."""
        return (adv - ref).pow(2).mean(dim=-1) / (ref.pow(2).mean() + self.SCALE_EPS)

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses define distance_to_loss")


class FurtherAL(DetectorLayer, ActivationLoss):
    """
    Model 1: detector + contrastive loss. Pushes away adversarial activations 
    from clean ones in order to make them more recognizable by the detector.

    When FurtherAL invokes collect, the Method Resolution Order (MRO) is

            DetectorLayer -> ActivationLoss -> PassThrough

    Contrastive Loss: ReLU(margin - distance).
    Pushes away until the distance exceeds the margin, then gets zero.
    Maximizing distance without any constraints would make it diverge to infinity.

    The distance is relative (ActivationLoss.distance). With an absolute
    distance the cheapest way to reach the margin is to inflate every
    activation (the gap grows with the scale), which separates nothing.

    DETACH_REFERENCE_DEFAULT = False. Detaching the real branch does not make
    it an anchor: real and adv share the weights, so the gradient on W of a
    linear layer becomes (W delta)(x + delta)^T instead of (W delta) delta^T,
    and the extra (W delta) x^T term drags the real activations along with
    the adv ones, inflating the scale far faster than the gap. Safe because
    ReLU(margin - d) is bounded by margin and switches off once the layers
    are far enough apart.
    """

    def __init__(self, base: nn.Module, detector: nn.Module,
                 margin: float = 1.0, **kwargs):
        super().__init__(base, detector=detector, **kwargs)
        self.margin = margin

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        return F.relu(self.margin - d).mean()


class CloserAL(ActivationLoss):
    """
    Model 2 (adversarial training): attractive loss, brings adversarial
    activations closer to the real ones. Incompatible with detectors, this
    constraint is verified within LWADSequential.

    The distance is relative (ActivationLoss.distance): an absolute one is
    lowered for free by shrinking every activation, which collapses the
    scale instead of aligning the two versions.

    DETACH_REFERENCE_DEFAULT = False, as for FurtherAL. Detaching here makes 
    the loss diverge.Intuitively, adversarial training wants a representation 
    where clean and adversarial versions meet, so both branches must be free
    to move.
    """

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        return d.mean()


# ===========================================================================
# Contenitore
# ===========================================================================
class LWADSequential(nn.Module):
    """Like nn.Sequential, but propagates (h(x), state).
       nn.Modules are automatically wrapped in PassThrough.

       Upon building, architecture coherency is verified: CloserAL can't
       cohexist with Detector-based layer within the same network."""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([
            m if isinstance(m, PassThrough) else PassThrough(m)
            for m in modules
        ])
        self._validate()

    def _validate(self) -> None:
        has_det = any(isinstance(m, DetectorLayer) for m in self.layers)
        has_closer = any(isinstance(m, CloserAL) for m in self.layers)
        if has_det and has_closer:
            raise ValueError(
                "Incoherent architecture: CloserAL can't "
                "cohexist with Detector-based layer within the same network."
            )

    @property
    def has_detectors(self) -> bool:
        return any(isinstance(m, DetectorLayer) for m in self.modules())

    def forward(self, x, labels=None, is_adv=None, exit_threshold=None):
        """exit_threshold (inference only): the forward stops as soon as every
        sample of the batch has been flagged by some detector reached so far,
        returning None logits and the exit layer in state.exit_layer.
        Exact under the "max" reduce only: max_i p_i > t <=> some p_i > t."""
        state = FlowState(labels=labels, is_adv=is_adv)
        flagged = None  # per sample: some detector so far was over the threshold
        for idx, layer in enumerate(self.layers):
            x, state = layer(x, state)
            if exit_threshold is not None and isinstance(layer, DetectorLayer):
                over = torch.sigmoid(state.detections[-1]) > exit_threshold
                flagged = over if flagged is None else flagged | over
                if bool(flagged.all()):
                    state.exit_layer = idx
                    return None, state
        return x, state

    def detector_parameters(self) -> Iterable[nn.Parameter]:
        for m in self.modules():
            if isinstance(m, DetectorLayer):
                yield from m.detector.parameters()

    # id(p) is the unique identifier of object p
    # actually is its memory address
    # in this way all detector parameters are excluded

    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        det_ids = {id(p) for p in self.detector_parameters()}
        return (p for p in self.parameters() if id(p) not in det_ids)