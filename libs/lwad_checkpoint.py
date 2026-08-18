"""Checkpoint saving and loading.

A checkpoint stores everything needed to rebuild a trained model without
knowing in advance which architecture produced it:

    state_dict     : model weights
    architecture   : config class name (DetectorArchConfig / AlignmentArchConfig)
    config         : all config fields, so build_model() can be replayed
    feature_names  : column order the model was trained on
    attack_mask    : which features are attackable (1.0) or categorical (0.0)
    threshold      : selected detector threshold (None for architecture 2)

Typical use:

    from libs.lwad_checkpoint import load_checkpoint
    ck = load_checkpoint("lwad_model.pt", device=device)
    labels, score, is_adv = predict(ck.model, x, threshold=ck.threshold)
"""
from __future__ import annotations

import dataclasses
import warnings
from typing import Optional

import torch

import libs.lwad_config as lc
import libs.lwad_wrapper as lw


def _config_classes() -> dict:
    """Returns a dict: name -> class: crosses recursively ArchitectureConfig 
    subclasses for easily adding new architectures"""
    found, stack = {}, [lc.ArchitectureConfig]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            found[sub.__name__] = sub
            stack.append(sub)
    return found


# ===========================================================================
# Store
# ===========================================================================
def save_checkpoint(path, model, config, feature_names,
                    attack_mask=None, threshold=None) -> None:
    """Stores weights + everything needed to rebuild the architecture.
    The checkpoint can be reloaded with weights_only=True."""
    torch.save({"state_dict": model.state_dict(),
                "architecture": type(config).__name__,
                "config": dict(config.__dict__),
                "feature_names": list(feature_names),
                "attack_mask": attack_mask,
                "threshold": threshold
                }, path)


# ===========================================================================
# Load
# ===========================================================================
@dataclasses.dataclass
class LoadedCheckpoint:
    model: lw.LWADSequential
    config: lc.ArchitectureConfig
    feature_names: list
    attack_mask: Optional[torch.Tensor]
    threshold: Optional[float]

    @property
    def uses_detectors(self) -> bool:
        return self.config.uses_detectors


def _rebuild_config(name: str, saved: dict) -> lc.ArchitectureConfig:
    """Rebuilds the config object, tolerating updates"""
    registry = _config_classes()
    if name not in registry:
        raise ValueError(
            f"unknown architecture {name!r} in checkpoint; "
            f"available: {sorted(registry)}"
        )
    cls = registry[name]
    fields = {f.name for f in dataclasses.fields(cls)}

    unknown = set(saved) - fields          # fields removed from the config
    missing = fields - set(saved)          # fields added after saving
    if unknown:
        warnings.warn(f"checkpoint fields ignored (no longer in {name}): "
                      f"{sorted(unknown)}")
    if missing:
        warnings.warn(f"fields missing from checkpoint, defaults used: "
                      f"{sorted(missing)}")
    return cls(**{k: v for k, v in saved.items() if k in fields})


def load_checkpoint(path, device="cpu", eval_mode=True) -> LoadedCheckpoint:
    """Rebuilds model + config from a checkpoint written by save_checkpoint.
    The architecture is replayed through config.build_model, so weights and
    module names always line up."""
    ck = torch.load(path, map_location=device)

    for key in ("state_dict", "architecture", "config", "feature_names"):
        if key not in ck:
            raise KeyError(
                f"checkpoint missing {key!r}: it was probably written by an "
                "older version, re-train or save it with save_checkpoint"
            )

    config = _rebuild_config(ck["architecture"], ck["config"])
    feature_names = list(ck["feature_names"])

    model = config.build_model(len(feature_names))
    model.load_state_dict(ck["state_dict"])   # strict: fails on any mismatch
    model.to(device)
    if eval_mode:
        model.eval()

    attack_mask = ck.get("attack_mask")
    if attack_mask is not None:
        attack_mask = attack_mask.to(device)

    return LoadedCheckpoint(model=model, config=config,
                            feature_names=feature_names,
                            attack_mask=attack_mask,
                            threshold=ck.get("threshold"))


# ===========================================================================
# Input alignment
# ===========================================================================
def align_features(df, feature_names):
    """Reorders (and checks) a DataFrame's columns to the order the model was
    trained on."""
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"missing features: {missing}")
    extra = [c for c in df.columns if c not in feature_names]
    if extra:
        warnings.warn(f"extra columns dropped: {extra}")
    return df[list(feature_names)]
