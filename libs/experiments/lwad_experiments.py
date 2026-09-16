"""Declarative experiment suites.

Edit the EXPERIMENTS table below: one line per experiment, giving a name, a
preset, and ONLY the parameters that differ from it. Then

    import libs.lwad_experiments as lx
    import libs.lwad_stage1            # appends stage 1 exp to lx.EXPERIMENTS
    res = lx.run_suite(lx.EXPERIMENTS, lx.DATASETS)
"""
from __future__ import annotations

import csv
import dataclasses
import itertools
import os
import re
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

import torch

import libs.preprocess.preprocess as pp
import libs.model.lwad_config as lc
import libs.attacks.lwad_attack as la
import libs.training.lwad_margin as lm
import libs.training.lwad_trainer as lt
import libs.evaluation.lwad_evaluator as le
import libs.model.lwad_checkpoint as lcp


# ===========================================================================
# 1) PRESETS: every experiment is a delta of these experiments
# ===========================================================================
DET = lc.DetectorModelConfig(epochs=10, eps=0.1, train_attack="pgd", eval_attack="pgd")
ALI = lc.AdvTrainingModelConfig(epochs=10, eps=0.1, train_attack="pgd", eval_attack="pgd")

#DATASETS = list(pp.KaggleDataset)
DATASETS = list([pp.KaggleDataset.UNSW_BW15])

# ===========================================================================
# 2) EXPERIMENTS: name, preset, and only what changes
#
#    Exp    : a single run
#    Sweep  : cartesian product of the given axes
#    Paired : axes advanced together (same length), not crossed
# ===========================================================================
def _table():
    return [
        # --- detector model ------------------------------------------------
        #Exp("det/base",           DET),
        #Exp("det/no-actloss",     DET, use_act_loss=False),
        #Exp("det/score-max",      DET, score_reduce="max"),
        #Exp("det/attached",       DET, detach=False),
        #Exp("det/no-input-norm",  DET, input_norm=False),
        #Exp("det/wide",           DET, hidden_dims=(512, 256, 96), wrap_at=(0, 2)),
        #Exp("det/3-det",          DET, hidden_dims=(256, 128, 64), wrap_at=(0, 1, 2),
        #                               act_margin=(9e-4, 9e-4, 9e-4)),
        #Exp("det/fat-head",       DET, detector_dims=(256, 128, 64)),
        #Exp("det/linear-head",    DET, detector_dims=()),

        # --- adv training model --------------------------------------------
        #Exp("ali/base",           ALI),
        #Exp("ali/clean-task",     ALI, task_loss_on_adv=False),
        #Exp("ali/wide",           ALI, hidden_dims=(768, 320, 96), wrap_at=(1, 2)),

        # --- sweeps --------------------------------------------------------
        #Sweep("det/eps",     DET, eps=[0.05, 0.1, 0.2]),
        #Sweep("det/lr",      DET, lr=[1e-3, 3e-4], lr_det=[3e-3, 1e-3]),
        #Sweep("det/lambda",  DET, lambda_det=[0.5, 1.0, 2.0], lambda_act=[0.0, 1.0]),
        #Sweep("det/atk",     DET, train_attack=["fgsm", "pgd"],
        #                          eval_attack=["pgd", "pgd_adaptive"]),
        #Paired("det/steps",  DET, pgd_steps=[10, 20],
        #                          pgd_alpha=[0.05, 0.025]),
        #Sweep("ali/eps",     ALI, eps=[0.05, 0.1, 0.2]),
    ]


# ===========================================================================
# Declaration layer
# ===========================================================================
def _fmt(v: Any) -> str:
    """Compact rendering of a value for names and tables."""
    if isinstance(v, float):
        return f"{v:g}"
    if isinstance(v, (tuple, list)):
        inner = ",".join(_fmt(x) for x in v)
        return f"({inner},)" if len(v) == 1 else f"({inner})"
    return str(v)


class Exp:
    """One experiment: a preset plus the overrides that make it different."""

    def __init__(self, name: str, base: lc.ModelConfig, **overrides):
        self.name = name
        self.base = base
        self.overrides = overrides  # the params that differ from base

    # --- identity -----------------------------------------------------------
    # 
    # Example for Exp('det/wide', DetectorModelConfig, hidden_dims=(512,256,96), wrap_at=(0,2))
    # * .model      -> 'det'
    # * .describe() -> 'hidden_dims=(512,256,96), wrap_at=(0,2)'
    # * repr()      -> Exp('det/wide', DetectorModelConfig, hidden_dims=(512,256,96), wrap_at=(0,2))

    @property
    def model(self) -> str:
        return type(self.base).__name__.replace("ModelConfig", "")[:3].lower()

    def describe(self) -> str:
        return ", ".join(f"{k}={_fmt(v)}" for k, v in self.overrides.items())

    def __repr__(self) -> str:
        d = self.describe()
        return f"Exp({self.name!r}, {type(self.base).__name__}{', ' + d if d else ''})"

    # --- materialization ----------------------------------------------------
    def config(self, **extra) -> lc.ModelConfig:
        """The config for this run. Careful: pgd_alpha gets materialized at
        post init time (based on eps) if not passed, and the replace method 
        calls the __init__ method, thus also the __post_init__.
        """
        ov = dict(self.overrides, **extra)
        if "eps" in ov and "pgd_alpha" not in ov:
            ov["pgd_alpha"] = None
        return dataclasses.replace(self.base, **ov)

    def expand(self) -> list["Exp"]:
        return [self]

    def validate(self) -> None:
        """For validating every run before experiments start"""
        cls = type(self.base)

        # Check for unknown fields
        known_fields = {f.name for f in dataclasses.fields(cls)}
        unknown_fields = sorted(set(self.overrides) - known_fields)
        if unknown_fields:
            raise ValueError(
                f"{self.name}: unknown field(s) {unknown_fields} for {cls.__name__}. "
                f"Valid fields: {sorted(known_fields)}"
            )
        cfg = self.config()

        # Check for coherent and correct attacks
        for which in ("train_attack", "eval_attack"):
            try:
                la.Attack(getattr(cfg, which))
            except ValueError:
                raise ValueError(
                    f"{self.name}: unknown {which}={getattr(cfg, which)!r}, "
                    f"expected one of {[a.value for a in la.Attack]}"
                ) from None
            # pgd_adaptive needs a detector to evade
            if getattr(cfg, which) == la.Attack.PGD_ADAPTIVE.value and not cfg.uses_detectors:
                raise ValueError(
                    f"{self.name}: {which}='pgd_adaptive' requires a detector-based "
                    f"model, but {cls.__name__} has none"
                )
        # Check for correct wrap indexes
        cfg.resolved_wrap_at()

        # Verify the checkpoint to load exists
        if cfg.load_from and not os.path.exists(cfg.load_from):
            print(f"warning: {self.name}: load_from={cfg.load_from!r} does not "
                  f"exist yet, the run will fail unless it is created first")

        # Check for correct passed margins, and warns if they are passed an not used
        if cfg.uses_detectors:
            if cfg.margin_factor is None:
                cfg.margin_for_layer(0, cfg.n_wrapped_layers)
                if isinstance(cfg.act_margin, tuple) and not cfg.use_act_loss:
                    print(f"warning: {self.name}: act_margin is a tuple but "
                          f"use_act_loss=False, the margins are unused")
            else:
                # act_margin is going to be overwritten by _resolve_margins, so
                # its declared shape is not checked here
                if cfg.margin_factor <= 0:
                    raise ValueError(f"{self.name}: margin_factor must be "
                                     f"positive, got {cfg.margin_factor}")
                if not cfg.use_act_loss:
                    print(f"warning: {self.name}: margin_factor is set but "
                          f"use_act_loss=False, no margin is ever used")


class Sweep(Exp):
    """Cartesian product of the given axes: each axis is a list of values."""

    def __init__(self, name: str, base: lc.ModelConfig, **axes):
        super().__init__(name, base, **axes)
        for k, v in axes.items():
            if not isinstance(v, (list, tuple)) or not len(v):
                raise ValueError(f"{name}: axis {k!r} must be a non-empty list")

    def _combinations(self) -> Iterable[tuple]:
        """Produces cartesian product of the overrides values (argument names are lost)"""
        return itertools.product(*self.overrides.values())

    def expand(self) -> list[Exp]:
        keys = list(self.overrides) # list of arguments, without their values
        out = []
        for combo in self._combinations():
            combo_overrides = dict(zip(keys, combo)) # re-associates arguments to their values
            tag = ",".join(f"{k}={_fmt(v)}" for k, v in combo_overrides.items())
            out.append(Exp(f"{self.name}[{tag}]", self.base, **combo_overrides))
        return out

    def validate(self) -> None:
        for e in self.expand():
            e.validate()


class Paired(Sweep):
    """Axes advanced together instead of crossed: all must have the same length."""

    def __init__(self, name: str, base: lc.ModelConfig, **axes):
        super().__init__(name, base, **axes)
        lengths = {k: len(v) for k, v in axes.items()}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"{name}: Paired axes must have equal lengths, got {lengths}")

    def _combinations(self) -> Iterable[tuple]:
        return zip(*self.overrides.values())


def expand_all(experiments: Sequence[Exp]) -> list[Exp]:
    """Flattens Sweep/Paired into plain Exp and rejects duplicate names."""
    out: list[Exp] = []

    # Flattning
    for e in experiments:
        out.extend(e.expand())

    # Check for duplicates
    seen: dict[str, int] = {}
    for e in out:
        seen[e.name] = seen.get(e.name, 0) + 1
    dupes = sorted(n for n, c in seen.items() if c > 1)
    if dupes:
        raise ValueError(f"duplicate experiment names: {dupes}")
    
    return out


# ===========================================================================
# Data layer
# ===========================================================================
@dataclass
class DatasetBundle:
    """One dataset, loaded once and reused by every experiment in the suite."""

    name: str
    X_train: torch.Tensor; y_train: torch.Tensor
    X_val:   torch.Tensor; y_val:   torch.Tensor
    X_test:  torch.Tensor; y_test:  torch.Tensor
    feature_names: list
    attack_mask: Optional[torch.Tensor]
    class_weights: torch.Tensor
    categorical: list = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @classmethod
    def load(cls, kd, device: str = "cpu", download: bool = False,
             verbose: bool = False,
             test_max_rows: Optional[int] = pp.DEFAULT_TEST_MAX_ROWS) -> "DatasetBundle":
        (X_train, y_train, 
         X_val, y_val,
         X_test, y_test, 
         feature_names, attack_mask) = pp.load_dataset(
            kd, download, verbose, test_max_rows=test_max_rows)

        X_train, y_train = X_train.to(device), y_train.to(device)
        X_val, y_val = X_val.to(device), y_val.to(device)
        X_test, y_test = X_test.to(device), y_test.to(device)
        if attack_mask is not None:
            attack_mask = attack_mask.to(device)

        # class weights: the datasets carry more attack than benign instances
        counts = torch.bincount(y_train, minlength=2).float()
        class_weights = (len(y_train) / (2 * counts)).to(device)

        return cls(name=kd.value, X_train=X_train, y_train=y_train,
                   X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test,
                   feature_names=feature_names, attack_mask=attack_mask,
                   class_weights=class_weights,
                   categorical=[f for f in pp.get_categorical_cols(kd)
                                if f in feature_names])

    def summary(self) -> str:
        n_attack = int((self.y_train == 1).sum())
        return (f"{self.name}:\n"
                f"\t{len(self.X_train)} train - {len(self.X_val)} val - "
                f"{len(self.X_test)} test \n"
                f"\t{self.n_features} features \n"
                f"\tcategorical features excluded from attack: {self.categorical}\n"
                f"\ttraining set class 1 (attack instances) = "
                f"{n_attack / len(self.y_train):.1%}")

    def free(self) -> None:
        """Drops the tensors so the next dataset does not stack on top of them."""
        for attr in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                     "attack_mask", "class_weights"):
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ===========================================================================
# Run utilities
# ===========================================================================
def validation_score(metrics: dict, mode: str = lc.DEFAULT_SCORE_MODE) -> float:
    """The single number every run is ranked on, identical for both model types.

    The old per-model formulas (detector balanced accuracy vs adversarial task 
    accuracy) were not comparable, so they could not order detector models and
    adversarial training models in the same table. The end-to-end metrics
    can: see lwad_evaluator for their definition.

    mode: "joint" (default) | "clean" | "robust" - see lc.ScoreMode.
    """
    return le.combined_score(metrics, mode)


def _epoch_report(epoch: int, stats: dict, val: dict, score: float) -> str:
    msg = (f"epoch {epoch:3d}:\n"
           f"\ttask loss={stats['task_loss']:.4f}\n")
    if stats["det_loss"] is not None:
        msg += f"\tdet loss={stats['det_loss']:.4f}\n"
    if stats["act_loss"] is not None:
        msg += f"\tact loss={stats['act_loss']:.4f}\n"
    msg += (f"\ttask clean samples acc={stats['task_clean_acc']:.4f}\n"
            f"\ttask adv samples acc={stats['task_adv_acc']:.4f}\n")
    if stats["det_clean_acc"] is not None:
        msg += (f"\tdet clean samples acc={stats['det_clean_acc']:.4f}\n"
                f"\tdet adv samples acc={stats['det_adv_acc']:.4f}\n")
    msg += (f"\t[val] task acc={val['task_clean']['acc']:.4f}  "
            f"clean e2e={val['clean_acc_e2e']:.4f}  "
            f"robust e2e={val['robust_acc_e2e']:.4f}  "
            f"score={score:.4f}")
    return msg


def train_with_early_stopping(model, optimizer, cfg, data: DatasetBundle,
                              device: str = "cpu", verbose: int = 1
                              ) -> tuple[Optional[dict], float, int]:
    """Training loop with early stopping, validated against the training attack.

    Returns (best state_dict on cpu, best validation score, epochs actually run).
    """
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data.X_train, data.y_train),
        batch_size=cfg.batch_size, shuffle=True,
    )
    lambda_det = cfg.lambda_det if cfg.uses_detectors else 0.0
    threshold_det = cfg.threshold_det if cfg.uses_detectors else lc.DEFAULT_THRESHOLD_DET
    attack_kwargs = cfg.attack_kwargs()

    best_val_score, best_state, patience_left = -1.0, None, cfg.patience
    epochs_ran = 0
    for epoch in range(cfg.epochs):
        epochs_ran = epoch + 1
        stats = lt.train_epoch(model, loader, optimizer,
                               eps=cfg.eps, lambda_det=lambda_det,
                               lambda_act=cfg.lambda_act,
                               task_loss_on_adv=cfg.task_loss_on_adv,
                               class_weights=data.class_weights,
                               attack_mask=data.attack_mask, attack=cfg.train_attack,
                               threshold_det=threshold_det, attack_kwargs=attack_kwargs,
                               device=device, reduce=cfg.score_reduce)

        val = le.evaluate(model, data.X_val, data.y_val, eps=cfg.eps,
                          attack_mask=data.attack_mask, attack=cfg.train_attack,
                          device=device, threshold_det=threshold_det,
                          attack_kwargs=attack_kwargs, reduce=cfg.score_reduce)
        val_score = validation_score(val, cfg.score_mode)

        if verbose >= 2:
            print(_epoch_report(epoch, stats, val, val_score))

        if val_score > best_val_score + cfg.min_delta:
            best_val_score = val_score
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0 and epochs_ran >= cfg.min_epochs:
                if verbose >= 2:
                    print(f"\tearly stopping: no improvement for "
                          f"{cfg.patience} epochs")
                break

    return best_state, best_val_score, epochs_ran


# ===========================================================================
# Results
# ===========================================================================
CSV_COLUMNS = [
    "dataset", "experiment", "model", "params", "seed", "epochs", "val_score",
    "threshold_det", "train_attack", "eval_attack",
    "task_clean_acc", "task_clean_prec", "task_clean_rec",
    "task_adv_acc", "task_adv_prec", "task_adv_rec",
    "det_clean_acc", "det_adv_acc", "det_precision", "det_recall",
    "score_clean", "score_adv",
    "clean_acc_e2e", "robust_acc_e2e",
    "duration_s", "checkpoint", "error",
]

# decimals of columns for summary_table and pivot that must have ≠4 decimals 
AGG_DECIMALS = {"epochs": 0, "duration_s": 0}


@dataclass
class RunResult:
    experiment: str
    dataset: str
    model: str
    seed: int
    params: str
    config: Optional[lc.ModelConfig] = None
    epochs_ran: int = 0
    best_val_score: float = float("nan")
    threshold_det: Optional[float] = None
    metrics: Optional[dict] = None
    duration_s: float = 0.0
    checkpoint: Optional[str] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.metrics is not None

    def row(self) -> dict:
        """Defines a csv row with model type specific values, others are ignored"""
        r = {c: "" for c in CSV_COLUMNS}
        r.update(dataset=self.dataset, experiment=self.experiment, model=self.model,
                 params=self.params, seed=self.seed, epochs=self.epochs_ran,
                 duration_s=round(self.duration_s, 1),
                 checkpoint=self.checkpoint or "", error=self.error or "")
        if self.config is not None:
            r.update(train_attack=self.config.train_attack,
                     eval_attack=self.config.eval_attack)
        if self.threshold_det is not None:
            r["threshold_det"] = round(self.threshold_det, 4)
        if self.best_val_score == self.best_val_score:
            r["val_score"] = round(self.best_val_score, 4)
        m = self.metrics
        if m:
            tc, ta = m["task_clean"], m["task_adv"]
            r.update(task_clean_acc=round(tc["acc"], 4),
                     task_clean_prec=round(tc["precision"], 4),
                     task_clean_rec=round(tc["recall"], 4),
                     task_adv_acc=round(ta["acc"], 4),
                     task_adv_prec=round(ta["precision"], 4),
                     task_adv_rec=round(ta["recall"], 4),
                     clean_acc_e2e=round(m["clean_acc_e2e"], 4),
                     robust_acc_e2e=round(m["robust_acc_e2e"], 4))
            if m["detector"] is not None:
                r.update(det_clean_acc=round(m["det_clean_acc"], 4),
                         det_adv_acc=round(m["det_adv_acc"], 4),
                         det_precision=round(m["detector"]["precision"], 4),
                         det_recall=round(m["detector"]["recall"], 4),
                         score_clean=round(m["score_clean"], 4),
                         score_adv=round(m["score_adv"], 4))
        return r


def _render_table(rows: list[list[str]], headers: list[str]) -> str:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows
              else len(headers[i]) for i in range(len(headers))]
    def line(cells):
        return "  ".join(c.ljust(w) if i == 0 else c.rjust(w)
                         for i, (c, w) in enumerate(zip(cells, widths)))
    return "\n".join([line(headers), line(["-" * w for w in widths])] +
                     [line(r) for r in rows])


def _agg_cell(values: list, decimals: int = 4) -> str:
    """Mean of the values, with ±std as soon as there is more than one seed.

    Non numeric entries (the empty string a row carries for a metric that does
    not apply to that model type) are dropped, so a column that never applies
    shows "-" rather than a misleading 0.
    """
    nums = [v for v in values if isinstance(v, (int, float))]
    if not nums:
        return "-"
    mean = sum(nums) / len(nums)
    if len(nums) == 1:
        return f"{mean:.{decimals}f}"
    sd = (sum((v - mean) ** 2 for v in nums) / (len(nums) - 1)) ** 0.5
    return f"{mean:.{decimals}f}\u00b1{sd:.{decimals}f}"


def _group_by_run(rows: list[dict]) -> dict:
    """(experiment, dataset) -> its rows, one per seed."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["experiment"], r["dataset"]), []).append(r)
    return groups


@dataclass
class SuiteResult:
    results: list[RunResult] = field(default_factory=list)

    def rows(self) -> list[dict]:
        return [r.row() for r in self.results]

    def failures(self) -> list[RunResult]:
        return [r for r in self.results if not r.ok]

    # --- reporting ----------------------------------------------------------
    def summary_table(self, sort_by: str = "val_score", aggregate: bool = True) -> str:
        """Run results, one line per (experiment, dataset) pair. Details:
           * with more than one seed the cells carry mean±std across seeds, and
             column n is how many seeds contributed
           * the order is by (dataset, sort_by)
           * the minus is for not-applicable metrics, ERR for failed runs
           * aggregate=False restores the old one-line-per-run view
        """
        cols = ["experiment", "dataset", "model", "n", "epochs", "task_clean_acc",
                "task_adv_acc", "det_clean_acc", "det_adv_acc",
                "clean_acc_e2e", "robust_acc_e2e", "threshold_det", "val_score",
                "duration_s"]
        n_bad = len(self.failures())
        head = (f"== summary ({len(self.results)} run"
                f"{f', {n_bad} failed' if n_bad else ''}) ==")

        if not aggregate:
            plain = [c for c in cols if c != "n"]
            rows = sorted(self.rows(),
                          key=lambda r: (r["dataset"],
                                         -(r[sort_by] if r[sort_by] != "" else -1e9)))
            body = [[str(r[c]) if r[c] != "" else ("ERR" if r["error"] else "-")
                     for c in plain] for r in rows]
            return f"{head}\n{_render_table(body, plain)}"

        lines = []
        for (exp, ds), rs in _group_by_run(self.rows()).items():
            ok = [r for r in rs if not r["error"]]
            n = str(len(rs)) if len(ok) == len(rs) else f"{len(ok)}/{len(rs)}"
            if not ok:
                lines.append((ds, -1e9,
                              [exp, ds, rs[0]["model"], n] + ["ERR"] * (len(cols) - 4)))
                continue
            cells = [_agg_cell([r[c] for r in ok], AGG_DECIMALS.get(c, 4))
                     for c in cols[4:]]
            keys = [r[sort_by] for r in ok if isinstance(r[sort_by], (int, float))]
            lines.append((ds, sum(keys) / len(keys) if keys else -1e9,
                          [exp, ds, ok[0]["model"], n] + cells))
        body = [line for _, _, line in sorted(lines, key=lambda t: (t[0], -t[1]))]
        return f"{head}\n{_render_table(body, cols)}"

    def pivot(self, metric: str = "val_score") -> str:
        """
        For a given metric, prints a table comparing the experiments for every dataset.
        Cells are averaged over seeds.
        """
        rows = self.rows()
        datasets, experiments = [], []
        for row in rows:
            if row["dataset"] not in datasets:
                datasets.append(row["dataset"])
            if row["experiment"] not in experiments:
                experiments.append(row["experiment"])
        cell: dict[tuple[str, str], str] = {}
        for (exp, ds), rs in _group_by_run(rows).items():
            ok = [r for r in rs if not r["error"]]
            cell[(ds, exp)] = ("ERR" if not ok else
                               _agg_cell([r.get(metric, "") for r in ok],
                                         AGG_DECIMALS.get(metric, 4)))
        body = [[e] + [cell.get((d, e), "") for d in datasets] for e in experiments]
        return (f"== {metric}: esperimento x dataset ==\n"
                + _render_table(body, ["experiment"] + datasets))


def _write_rows(path: str, rows: list[dict], append: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    exists = append and os.path.exists(path)
    with open(path, "a" if append else "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow(r)
        fh.flush()


def _done_runs(path: str) -> set:
    """(experiment, dataset, seed) triples already recorded without error."""
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if not r.get("error"):
                done.add((r["experiment"], r["dataset"], str(r["seed"])))
    return done


# ===========================================================================
# Margin calibration with cache
# ===========================================================================
MARGIN_PROBE_SIZE = 16384
_MARGIN_CACHE: dict[tuple, tuple] = {}


def clear_margin_cache() -> None:
    """Drops the stored natural distances."""
    _MARGIN_CACHE.clear()


def _margin_cache_key(cfg: lc.DetectorModelConfig, dataset: str) -> tuple:
    """What the measured natural distances d_i depend on.

    Deliberately out of the key:
    * pgd_steps and pgd_alpha
    * lr / lr_det / batch_size (fixed during experiments)
    * warmup_epochs (fixed during experiments)

    """
    return (dataset, # margins change depending on the task
            cfg.input_norm, cfg.resolved_hidden_dims(), cfg.resolved_wrap_at(), # backbone info
            cfg.detector_norm, cfg.resolved_detector_dims(), # detector info
            cfg.detach, cfg.lambda_det, # if the detector affects the backbone                                                     
            cfg.task_loss_on_adv, # if training is also adversarial training
            cfg.eps, cfg.train_attack,) # attack configuration


def _resolve_margins(cfg: lc.ModelConfig, data: DatasetBundle, *,
                     device: str = "cpu", verbose: int = 1
                     ) -> lc.ModelConfig:
    """Replaces act_margin with margin_factor * d_i, d_i measured per layer.

    Must run before torch.manual_seed(seed): the warmup inside suggest_margins
    reseeds the global rng to lc.SEED (lwad_margin._warmup), so running it
    afterwards would hand every seed of a multi-seed run the same initial
    weights - and only on a cache MISS, which would make the result depend on
    the order the runs happen to execute in.
    """
    if not cfg.uses_detectors or cfg.margin_factor is None:
        return cfg
    if not cfg.use_act_loss:
        return cfg          # DetectorLayer owns no margin: a warmup would buy nothing

    key = _margin_cache_key(cfg, data.name)
    base_d = _MARGIN_CACHE.get(key)
    cached = base_d is not None
    if base_d is None:
        n = min(MARGIN_PROBE_SIZE, len(data.X_train))
        base_d = lm.suggest_margins(cfg, data.X_train[:n], data.y_train[:n],
                                    attack_mask=data.attack_mask, device=device,
                                    factor=1.0,
                                    warmup_epochs=cfg.margin_warmup_epochs,
                                    class_weights=data.class_weights,
                                    verbose=verbose >= 2)
        if not base_d:
            raise ValueError(
                "margin_factor is set but the network has no layer carrying a "
                f"margin (wrap_at={cfg.resolved_wrap_at()}, "
                f"use_act_loss={cfg.use_act_loss})"
            )
        _MARGIN_CACHE[key] = base_d

    margins = tuple(cfg.margin_factor * d for d in base_d)
    if verbose >= 1:
        print(f"act_margin = {tuple(round(m, 6) for m in margins)}  "
              f"({cfg.margin_factor:g}x natural d="
              f"{tuple(round(d, 6) for d in base_d)}, "
              f"{'cached' if cached else 'measured'})")
    return dataclasses.replace(cfg, act_margin=margins)


# ===========================================================================
# Reusing an already trained model
# ===========================================================================
def _load_pretrained(cfg: lc.ModelConfig, data: DatasetBundle, *,
                     device: str = "cpu", verbose: int = 1):
    """Rebuilds a trained model from cfg.load_from instead of training one.

    This is what makes a worst-case search affordable: the defense is fixed,
    only the attack changes, so retraining an identical model for every attack
    would be pure waste (a training costs ~20x an evaluation).

    Returns (model, threshold_det).
    """
    if not os.path.exists(cfg.load_from):
        raise FileNotFoundError(f"load_from: no checkpoint at {cfg.load_from!r}")
    ck = lcp.load_checkpoint(cfg.load_from, device=device)

    if type(ck.config) is not type(cfg):
        raise ValueError(
            f"load_from: the checkpoint holds a {type(ck.config).__name__} but "
            f"the experiment declares a {type(cfg).__name__}"
        )
    if list(ck.feature_names) != list(data.feature_names):
        raise ValueError(
            f"load_from: the checkpoint was trained on {len(ck.feature_names)} "
            f"features, {data.name} has {data.n_features} and they do not match: "
            f"wrong dataset for this checkpoint"
        )

    threshold_det = ck.threshold_det
    if cfg.uses_detectors:
        # check score reduction
        if threshold_det is None or ck.config.score_reduce != cfg.score_reduce:
            threshold_det, val_bal = lt.select_threshold(
                ck.model, data.X_val, data.y_val, eps=cfg.eps,
                attack_mask=data.attack_mask, attack=cfg.train_attack,
                device=device, attack_kwargs=cfg.attack_kwargs(),
                reduce=cfg.score_reduce)
            if verbose >= 1:
                print(f"threshold re-selected for score_reduce="
                      f"{cfg.score_reduce!r}: {threshold_det:.3f} "
                      f"(balanced acc on validation set = {val_bal:.4f})")
    if verbose >= 2:
        print(f"loaded {cfg.load_from}, training skipped")
    return ck.model, threshold_det


# ===========================================================================
# Execution
# ===========================================================================
def _safe_filename(name: str) -> str:
    """
    Converts an experiment name into a valid file name
    """
    for ch in "/=[], ":
        name = name.replace(ch, "-")
    return re.sub(r"-+", "-", name).strip("-") # collapse any run of dashes


def run_experiment(exp: Exp, data: DatasetBundle, *, device: str = "cpu",
                   seed: int = lc.SEED, verbose: int = 1,
                   checkpoint_dir: Optional[str] = None) -> RunResult:
    """Trains, picks the detector threshold and evaluates one config on one
    dataset. With cfg.load_from set, training and threshold selection are
    skipped and the stored model is evaluated as it is."""

    # --- setting up ----------------------------------------------------------
    started = time.time()
    ckpt = (os.path.join(checkpoint_dir,
                         f"{_safe_filename(exp.name)}__{_safe_filename(data.name)}__seed{seed}.pt")
            if checkpoint_dir else None)
    cfg = exp.config(**({"checkpoint": ckpt} if ckpt else {}))
    res = RunResult(experiment=exp.name, dataset=data.name, model=exp.model,
                    seed=seed, params=exp.describe(), config=cfg, checkpoint=ckpt)

    if verbose >= 2:
        print(f"{data.summary()}\n"
              f"model type = {type(cfg).__name__}\n"
              f"train attack = {cfg.train_attack}  |  eval attack = {cfg.eval_attack}\n")

    if cfg.load_from:
        # --- reuse an already trained defense --------------------------------
        model, threshold_det = _load_pretrained(cfg, data, device=device,
                                                verbose=verbose)
        if threshold_det is None:
            threshold_det = lc.DEFAULT_THRESHOLD_DET
        res.threshold_det = threshold_det if cfg.uses_detectors else None
        res.checkpoint = cfg.load_from
    else:
        # margins first (as stated in _resolve_margins)
        cfg = _resolve_margins(cfg, data, device=device, verbose=verbose)
        res.config = cfg

        torch.manual_seed(seed)
        built = lc.create_model(cfg, data.n_features, device=device)
        model, optimizer = built.model, built.optimizer

        # --- training --------------------------------------------------------
        if verbose >= 2:
            print("\n== training ==")
        best_state, best_val, epochs_ran = train_with_early_stopping(
            model, optimizer, cfg, data, device=device, verbose=verbose)
        if best_state is None:
            raise ValueError(f"no best state available, an error has occured")
        model.load_state_dict(best_state)
        res.best_val_score, res.epochs_ran = best_val, epochs_ran

        # --- detector threshold, chosen on the training attack ---------------
        threshold_det = cfg.threshold_det if cfg.uses_detectors else lc.DEFAULT_THRESHOLD_DET
        if cfg.uses_detectors:
            threshold_det, val_bal = lt.select_threshold(
                model, data.X_val, data.y_val, eps=cfg.eps, attack_mask=data.attack_mask,
                attack=cfg.train_attack, device=device,
                attack_kwargs=cfg.attack_kwargs(), reduce=cfg.score_reduce)
            if verbose >= 2:
                print(f"\ndetector threshold choice: {threshold_det:.3f} "
                      f"(balanced acc on validation set = {val_bal:.4f})")
            res.threshold_det = threshold_det

    # --- test set evaluation -------------------------------------------------
    if verbose >= 2:
        print(f"\n== evaluation on test set (eval attack = {cfg.eval_attack}) ==")
    res.metrics = le.evaluate(model, data.X_test, data.y_test, eps=cfg.eps,
                              attack_mask=data.attack_mask, attack=cfg.eval_attack,
                              device=device, threshold_det=threshold_det,
                              attack_kwargs=cfg.attack_kwargs(),
                              reduce=cfg.score_reduce)
    if verbose >= 2:
        print(_metrics_report(res.metrics))

    # ---- checkpoint ---------------------------------------------------------
    # stores weights + model type + config, so load_checkpoint() can rebuild it
    if ckpt and not cfg.load_from:      # no overwriting the model just reused
        os.makedirs(checkpoint_dir, exist_ok=True)
        lcp.save_checkpoint(ckpt, model, cfg, data.feature_names,
                            attack_mask=data.attack_mask,
                            threshold_det=res.threshold_det)

    res.duration_s = time.time() - started
    return res


def _metrics_report(m: dict) -> str:
    tc, ta = m["task_clean"], m["task_adv"]
    out = ["TASK (positive = attack)",
           f"  clean       : acc={tc['acc']:.4f}  prec={tc['precision']:.4f}  rec={tc['recall']:.4f}",
           f"  adversarial : acc={ta['acc']:.4f}  prec={ta['precision']:.4f}  rec={ta['recall']:.4f}"]
    if m["detector"] is not None:
        det = m["detector"]
        out += ["DETECTOR (positive = adversarial)",
                f"  clean accuracy      : {m['det_clean_acc']:.4f}",
                f"  adversarial accuracy: {m['det_adv_acc']:.4f}",
                f"  precision / recall  : {det['precision']:.4f} / {det['recall']:.4f}",
                f"  end to end (comparable across model types)",
            f"  clean  : {m['clean_acc_e2e']:.4f}   (right AND not flagged)",
            f"  robust : {m['robust_acc_e2e']:.4f}   (right OR flagged)"]
    return "\n".join(out)


def run_suite(experiments: Sequence[Exp] = None, datasets: Sequence = None, *,
              device: Optional[str] = None, out_dir: str = "results",
              summary_file: str = "summary.csv",
              seeds: Sequence[int] = (lc.SEED,), verbose: int = 1,
              only: Optional[str] = None, dry_run: bool = False,
              resume: bool = False, download: bool = False,
              save_checkpoints: bool = True,
              test_max_rows: Optional[int] = pp.DEFAULT_TEST_MAX_ROWS,
              on_error: str = "record") -> SuiteResult:
    """Runs every experiment on every dataset.

    Datasets are the outer loop: preprocessing dominates the non-training cost,
    so each one is loaded and moved to the device exactly once for the whole
    table. Results are appended to <out_dir>/<summary_file> after every run, so
    an interrupted suite keeps what it already produced (and resume=True skips
    it).

    Args:
        datasets: accepts KaggleDataset members or DatasetBundle objects.
        verbose: 0 silent, 
                 1 one banner and the final metrics per run,
                 2 the full per-epoch output of the original script.
        on_error: "record" keeps going and marks the run failed, "raise" stops.
        only: run only the experiments whose name starts with this prefix
        test_max_rows: caps and balances the test split
    """
    experiments = EXPERIMENTS if experiments is None else experiments
    datasets = DATASETS if datasets is None else datasets
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    csv_path = os.path.join(out_dir, summary_file)
    ckpt_dir = os.path.join(out_dir, "checkpoints") if save_checkpoints else None
    done = _done_runs(csv_path) if resume else set()

    if on_error not in ("record", "raise"):
        raise ValueError(f"on_error: expected 'record' or 'raise', got {on_error!r}")

    # --- get all experiments -------------------------------------------------
    runs = expand_all(experiments)
    if only:
        runs = [e for e in runs if e.name.startswith(only)]
        if not runs:
            raise ValueError(f"no experiment matches only={only!r}")
    for e in runs:
        e.validate()
    print("All configurations are valid")

    # distinct names can still collapse to the same slug, which would make two
    # runs write the same checkpoint file
    claimed_by: dict[str, str] = {}          # filename -> experiment that took it
    for e in runs:
        fname = _safe_filename(e.name)
        clash = claimed_by.setdefault(fname, e.name)
        if clash != e.name:
            raise ValueError(f"experiments {clash!r} and {e.name!r} would both "
                            f"write the checkpoint {fname!r}, rename one")

    total = len(runs) * len(datasets) * len(seeds)
    if dry_run or verbose >= 1:
        print(f"== suite: {len(runs)} experiments x {len(datasets)} datasets "
              f"x {len(seeds)} seeds = {total} runs ==")
        print(f"   device={device}  out_dir={out_dir}  "
              f"test_max_rows={test_max_rows}  "
              f"total max epochs={sum(e.config().epochs for e in runs) * len(datasets) * len(seeds)}")
    if dry_run:
        for e in runs:
            print(f"   {e.model}  {e.name:44s} {e.describe()}")
        return SuiteResult()

    # --- execute all experiments ---------------------------------------------
    suite, n = SuiteResult(), 0
    for kd in datasets:
        # a ready-made bundle is used as is (and never freed): handy for tests
        preloaded = isinstance(kd, DatasetBundle)
        data = kd if preloaded else None
        for exp in runs:
            for seed in seeds:
                n += 1
                key = (exp.name, kd.name if preloaded else getattr(kd, "value", str(kd)),
                       str(seed))
                if key in done:
                    if verbose >= 1:
                        print(f"### [{n:3d}/{total}] {exp.name} | {key[1]} | "
                              f"seed {seed} | already present, skipped ###")
                    continue
                if data is None:       # loaded lazily: a fully resumed dataset costs nothing
                    data = DatasetBundle.load(kd, device=device, download=download,
                                              verbose=verbose >= 2,
                                              test_max_rows=test_max_rows)
                if verbose >= 1:
                    print(f"\n### [{n:3d}/{total}] {exp.name} | {data.name} | "
                          f"seed {seed}{' | ' + exp.describe() if exp.overrides else ''} ###")
                try:
                    res = run_experiment(exp, data, device=device, seed=seed,
                                         verbose=verbose, checkpoint_dir=ckpt_dir)
                    if verbose == 1:
                        print(_metrics_report(res.metrics))
                except Exception as exc: # one bad run doesn't kill the suite
                    if on_error == "raise":
                        raise
                    res = RunResult(experiment=exp.name, dataset=data.name,
                                    model=exp.model, seed=seed, params=exp.describe(),
                                    error=f"{type(exc).__name__}: {exc}")
                    print(f"!! run failed: {res.error}")
                    if verbose >= 2:
                        traceback.print_exc()
                suite.results.append(res)
                _write_rows(csv_path, [res.row()], append=True)   # crash-safe
        if data is not None and not preloaded:
            data.free()

    if verbose >= 1:
        print(f"\n{suite.summary_table()}\n\n{suite.pivot('val_score')}")
        print(f"\nCSV: {csv_path}")
    return suite


EXPERIMENTS = _table()
