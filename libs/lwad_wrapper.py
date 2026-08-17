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
        self.act_loss: Optional[torch.Tensor] = None # activation loss (Further/Nearest)

    # --- detection (for DetectorLayer) -------------------------------------
    def add_detection(self, logit: torch.Tensor) -> None:
        self.detections.append(logit)
        if self.is_adv is not None:
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1), self.is_adv.float()
            )
            self.det_loss = loss if self.det_loss is None else self.det_loss + loss

    # --- activation loss (for FurtherAL and NearestAL) ---------------------
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

    def adv_score(self, reduce: str = "mean") -> Optional[torch.Tensor]:
        """reduce can be:
                "mean": averages all detectors scores
                "max" : alerts if at least one detector result is adversarial
        Returns None if the network doesn't contain any detector."""
        if not self.detections:
            return None

        probs = torch.sigmoid(torch.cat(self.detections, dim=-1))
        return probs.max(dim=-1).values if reduce == "max" else probs.mean(dim=-1)


def default_detector(in_dim: int, hidden: int = 64) -> nn.Module:
    """LayerNorm stabilizes the activation scales"""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden*2), nn.ReLU(),
        nn.Linear(hidden*2, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


# ===========================================================================
# Layer class diagram
#
#                     PassThrough
#                    /           \
#             DetectorLayer   ActivationLoss
#                    \          <abstract>
#                     \         /        \
#                      FurtherAL          NearestAL
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

    - enabled: enables loss without changing architecture
    - detach_reference: if true real activations are treated as a fixed
                        anchor and grad only moves adv activations.
                        None -> DETACH_REFERENCE_DEFAULT of the subclass.
    """

    # Overridden per loss type
    DETACH_REFERENCE_DEFAULT: bool = True

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
            # mean square distance, comparable between layers with different width
            d = (adv - ref).pow(2).mean(dim=-1)
            state.add_act_loss(self.distance_to_loss(d))
        super().collect(x, y, state)

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("subclasses define distance_to_loss")


class FurtherAL(DetectorLayer, ActivationLoss):
    """
    Architecture 1: detector + contrastive loss. Pushes away adversarial activations 
    from clean ones in order to make them more recognizable by the detector.

    When FurtherAL invokes collect, the Method Resolution Order (MRO) is

            DetectorLayer -> ActivationLoss -> PassThrough

    Contrastiva Loss: ReLU(margin - distance). 
    Pushes away until the distance exceeds the margin, then gets zero.
    Maximizing distance without any constraints would make it diverge to infinity.

    Keeps DETACH_REFERENCE_DEFAULT = True: clean activations are the
    anchor, adversarial ones are pushed away from, so that the clean
    representation is not dragged around to satisfy the repulsion. Safe here
    because ReLU(margin - d) is bounded by margin and switches off once the
    layers are far enough apart (distance = margin)
    """

    DETACH_REFERENCE_DEFAULT: bool = True

    def __init__(self, base: nn.Module, detector: nn.Module,
                 margin: float = 1.0, **kwargs):
        super().__init__(base, detector=detector, **kwargs)
        self.margin = margin

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        return F.relu(self.margin - d).mean()


class NearestAL(ActivationLoss):
    """
    Architecture 2 (adversarial training): attractive loss, brings adversarial
    activations closer to the real ones. Incompatible with detectors, this 
    constraint is verified within DetectorSequential.

    DETACH_REFERENCE_DEFAULT = False, unlike FurtherAL. 
    Detaching here makes the loss diverge.
    Intuitively, adversarial training wants a representation where 
    clean and adversarial versions MEET, so both branches must be free
    to move. Anchoring the clean branch only makes sense for repulsion.
    """

    DETACH_REFERENCE_DEFAULT: bool = False

    def distance_to_loss(self, d: torch.Tensor) -> torch.Tensor:
        return d.mean()


# ===========================================================================
# Contenitore
# ===========================================================================
class DetectorSequential(nn.Module):
    """Like nn.Sequential, but propagates (h(x), state).
       nn.Modules are automatically wrapped in PassThrough.

       Upon building, architecture coherency is verified: NearestAL can't
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
        has_nearest = any(isinstance(m, NearestAL) for m in self.layers)
        if has_det and has_nearest:
            raise ValueError(
                "Incoherent architecture: NearestAL can't "
                "cohexist with Detector-based layer within the same network."
            )

    @property
    def has_detectors(self) -> bool:
        return any(isinstance(m, DetectorLayer) for m in self.modules())

    def forward(self, x, labels=None, is_adv=None):
        state = FlowState(labels=labels, is_adv=is_adv)
        for layer in self.layers:
            x, state = layer(x, state)
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