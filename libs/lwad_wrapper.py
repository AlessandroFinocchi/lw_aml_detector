
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Iterable, Optional


class FlowState:

    def __init__(self, is_adv=None, labels=None):
        #self.labels = labels
        self.is_adv = is_adv
        self.detections: list[torch.Tensor] = []
        self.det_loss: Optional[torch.Tensor] = None

    def add_detection(self, logit: torch.Tensor) -> None:
        self.detections.append(logit)
        if self.is_adv is not None:
            loss = F.binary_cross_entropy_with_logits(
                logit.squeeze(-1), self.is_adv.float()
            )
            self.det_loss = loss if self.det_loss is None else self.det_loss + loss

    def adv_score(self, reduce: str = "mean") -> Optional[torch.Tensor]:
        """reduce can be:
                "mean": mean of all detection scores
                "max" : alerts if at least 1 adversarial detection   
        """
        if not self.detections:
            return None

        probs = torch.sigmoid(torch.cat(self.detections, dim=-1))
        return probs.max(dim=-1).values if reduce == "max" else probs.mean(dim=-1)


def default_detector(in_dim: int, hidden: int = 64) -> nn.Module:
    """LayerNorm stabilizes the activation scales"""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class PassThrough(nn.Module):
    """State propagates unchanged (x, state) -> (h(x), state)"""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x, state: FlowState):
        return self.base(x), state


class DetectorLayer(nn.Module):
    """detach can be:
            True: detector loss updates detector parameters
            False: detector loss update backbone and detector parameters,
                   making activations more easily recognizable
    """

    def __init__(self, base: nn.Module, detector: nn.Module, detach: bool = True):
        super().__init__()
        self.base = base
        self.detector = detector
        self.detach = detach

    def forward(self, x, state: FlowState):
        y = self.base(x)                          # base output
        feats = y.detach() if self.detach else y
        state.add_detection(self.detector(feats)) # real/adv detector classification
        return y, state


class DetectorSequential(nn.Module):
    """Like nn.Sequential, propagating (h(x), state)
       Normal modules get automatically wrapped in PassThrough"""

    def __init__(self, *modules: nn.Module):
        super().__init__()
        self.layers = nn.ModuleList([
            m if isinstance(m, (DetectorLayer, PassThrough)) else PassThrough(m)
            for m in modules
        ])

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