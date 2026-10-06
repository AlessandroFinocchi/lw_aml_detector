"""Shared infrastructure of the V&V suite (libs/tests/vv_*.py).

Every section module registers its tests in a TESTS list; 
vv_suite runs them all:

    python -m libs.tests.vv_suite                    # whole suite
    python -m libs.tests.vv_suite --only V1,V2.4,A3  # by id prefix
    python -m libs.tests.vv_verify_model             # one section

Every test prints its id, a short description and the outcome of each check
it makes; the run ends with the final report .

Determinism is configured here, at import time and before any CUDA
operation: CUBLAS_WORKSPACE_CONFIG is read when cuBLAS creates its first
handle, and without it the exact equalities (V2.4, V4) can fail for reasons
unrelated to the method.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import ast
import contextlib
import csv
import glob
import io
import math
import shutil
import statistics
import tempfile
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Optional

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False

import scipy.stats

import libs.preprocess.preprocess as pp
import libs.model.lwad_wrapper as lw
import libs.model.lwad_config as lc
import libs.model.lwad_checkpoint as lcp
import libs.training.lwad_trainer as lt
import libs.experiments.lwad_experiments as lx
import libs.experiments.lwad_stage1 as s1


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ===========================================================================
# Common definitions (spec: "Definizioni comuni")
# ===========================================================================
DEVICE = "cuda"
SEED = lc.SEED
DATASET = pp.KaggleDataset.UNSW_BW15

LINF_TOL = 1e-5             # tolerance of the L-inf constraint
SUBSET_ROWS = 4096          # fixed train subset of V1 and V2 (generator seed 0)
SMALL_EPOCHS = 3            # lt.train_epoch epochs of the trained small models

# excluded from "identical metrics": wall clock, never reproducible
LATENCY_KEYS = ("infer_ms", "infer_ms_std", "infer_ms_adv", "infer_ms_adv_std")
EXIT_RATE_KEYS = ("infer_exit_rate", "infer_exit_rate_adv")
KEY_METRICS = ("task_clean_acc", "task_adv_acc", "det_clean_acc", "det_adv_acc",
               "clean_acc_e2e", "robust_acc_e2e", "val_score")

S1_P_VALUE = 0.05           # S1: paired t-test threshold
S1_MIN_SEEDS = 3            # S1: fewer paired seeds -> INCONCLUSIVE
S2_ABS_TOL = 1e-3           # S2: |mean(A - B)| <= 2 sigma_ref + S2_ABS_TOL

MINI_INFER_SAMPLES = 64     # timed samples of the mini-campaign runs

# --- small configs (V1, V2) -------------------------------------------------
DET = lc.DetectorModelConfig(hidden_dims=(32, 16, 16), wrap_at=(0, 1, 2),
                             detector_dims=(16,), use_act_loss=False,
                             margin_factor=None, eps=0.2, pgd_steps=5,
                             batch_size=64)
FUR = replace(DET, use_act_loss=True, act_margin=1.0)
ADV = lc.AdvTrainingModelConfig(hidden_dims=(32, 16, 16), wrap_at=(0, 2),
                                eps=0.2, pgd_steps=5, batch_size=64)
SMALL = {"det": DET, "fur": FUR, "adv": ADV}

# --- mini-campaign configs (V4) ---------------------------------------------
COMMON_S = dict(epochs=3, eps=0.2, pgd_steps=5, batch_size=128,
                hidden_dims=(32, 16), wrap_at=(0, 1),
                train_attack="pgd", eval_attack="pgd")
DET_S = lc.DetectorModelConfig(**COMMON_S, use_act_loss=False, margin_factor=None)
FUR_S = lc.DetectorModelConfig(**COMMON_S, use_act_loss=True, margin_warmup_epochs=1)
ADV_S = lc.AdvTrainingModelConfig(**COMMON_S)

# --- stage 1 experiments the spec relies on ---------------------------------
# All declared in lwad_stage1.table(): the validation looks their names up in
# the stage 1 CSV (with at least 3 seeds).
S1_REQUIRED = [e.name for e in s1.table()]


def s1_experiment(name: str) -> lx.Exp:
    """The Exp a stage 1 run was produced by."""
    for e in s1.table():
        if e.name == name:
            return e
    raise KeyError(f"{name!r} is not a stage 1 experiment")


# ===========================================================================
# Outcomes and per-test report
# ===========================================================================
PASS, FAIL, INCONCLUSIVE, SKIP, ERROR = "PASS", "FAIL", "INCONCLUSIVE", "SKIP", "ERROR"
VERIFY, VALIDATE = "verifica", "validazione"


def worst(outcomes) -> str:
    """Test outcome out of its checks: a SKIP (part not runnable, e.g. stage 1
    data missing) counts only when nothing else ran."""
    for o in (ERROR, FAIL, INCONCLUSIVE, PASS):
        if o in outcomes:
            return o
    return SKIP


def _short(e: Exception, n: int = 70) -> str:
    s = str(e).replace("\n", " ")
    return s if len(s) <= n else s[:n] + "..."


def progress(text: str) -> None:
    print(f"   ... {text}", flush=True)


@contextlib.contextmanager
def quiet():
    """Silences the campaign code (banners, margin prints, config warnings)."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@dataclass
class TestSpec:
    tid: str
    title: str
    kind: str       # VERIFY | VALIDATE
    desc: str
    fn: Callable


def vv_test(registry: list, tid: str, title: str, kind: str, desc: str):
    """Registers fn(report, session) as test tid of a section."""
    def deco(fn):
        registry.append(TestSpec(tid, title, kind, desc, fn))
        return fn
    return deco


class Report:
    """Collects and prints the checks of one test."""

    def __init__(self, spec: TestSpec):
        self.spec = spec
        self.checks: list[tuple[str, str, str]] = []    # (outcome, label, detail)
        self.duration = 0.0

    def outcome(self, label: str, outcome: str, detail: Optional[str] = None) -> str:
        self.checks.append((outcome, label, detail or ""))
        line = f"   {'[' + outcome + ']':<15s}{label}"
        print(line + (f" -> {detail}" if detail else ""), flush=True)
        return outcome

    def check(self, label: str, ok, detail: Optional[str] = None) -> bool:
        self.outcome(label, PASS if ok else FAIL, detail)
        return bool(ok)

    def skip(self, label: str, reason: str) -> None:
        self.outcome(label, SKIP, reason)

    def inconclusive(self, label: str, reason: str) -> None:
        self.outcome(label, INCONCLUSIVE, reason)

    def info(self, text: str) -> None:
        print(f"      {text}", flush=True)

    def raises(self, label: str, exc: type, fn: Callable) -> bool:
        """Check that fn() raises exc."""
        try:
            fn()
        except exc as e:
            return self.check(label, True, f"{exc.__name__}: {_short(e)}")
        except Exception as e:
            return self.check(label, False, f"{type(e).__name__} invece di "
                                            f"{exc.__name__}: {_short(e)}")
        return self.check(label, False, f"nessuna eccezione, attesa {exc.__name__}")

    @property
    def verdict(self) -> str:
        return worst([o for o, _, _ in self.checks])

    def counts(self) -> str:
        c = Counter(o for o, _, _ in self.checks)
        return ", ".join(f"{c[o]} {o}" for o in (PASS, FAIL, INCONCLUSIVE, SKIP, ERROR)
                         if c[o])


# ===========================================================================
# Comparisons
# ===========================================================================
def same_value(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def diff_metrics(a, b, exclude=(), path: str = "") -> list[str]:
    """Recursive comparison of two metric dicts: the paths that differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k in exclude:
                continue
            if k not in a or k not in b:
                out.append(f"{path}{k} presente in uno solo")
            else:
                out += diff_metrics(a[k], b[k], exclude, f"{path}{k}.")
        return out
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return [f"{path.rstrip('.')}: lunghezza {len(a)} != {len(b)}"]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_metrics(x, y, exclude, f"{path}{i}.")
        return out
    return [] if same_value(a, b) else [f"{path.rstrip('.')}: {a!r} != {b!r}"]


def summarize(diffs: list[str], n: int = 3) -> Optional[str]:
    if not diffs:
        return None
    more = f" (+{len(diffs) - n})" if len(diffs) > n else ""
    return f"{len(diffs)} differenze: " + "; ".join(diffs[:n]) + more


def raw_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def diff_checkpoints(pa: str, pb: str) -> list[str]:
    """Identical checkpoints: same state_dict keys, identical tensors, same
    threshold_det."""
    a, b = raw_checkpoint(pa), raw_checkpoint(pb)
    sa, sb = a["state_dict"], b["state_dict"]
    out = []
    if set(sa) != set(sb):
        out.append(f"chiavi di state_dict diverse ({len(sa)} vs {len(sb)})")
    else:
        out += [f"{k} diverso" for k in sa if not torch.equal(sa[k], sb[k])]
    if not same_value(a.get("threshold_det"), b.get("threshold_det")):
        out.append(f"threshold_det {a.get('threshold_det')!r} != {b.get('threshold_det')!r}")
    return out


# CSV columns that identify a run instead of measuring it
ROW_IGNORED = ("dataset", "experiment", "model", "params", "seed", "duration_s",
               "checkpoint", "error") + LATENCY_KEYS


def diff_rows(a: dict, b: dict, exclude=()) -> list[str]:
    """Identical metrics between two CSV rows (stage 1 or RunResult.row())."""
    skip = set(ROW_IGNORED) | set(exclude)
    return [f"{k}: {a.get(k)!r} != {b.get(k)!r}"
            for k in lx.CSV_COLUMNS if k not in skip and not same_value(a.get(k), b.get(k))]


def _parse_cell(v: str):
    if v == "":
        return None
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def normalize_row(row: dict) -> dict:
    """RunResult.row() in the form a stage 1 CSV row is read back."""
    return {k: _parse_cell(str(v)) if isinstance(v, str) else v for k, v in row.items()}


def key_metrics(res: lx.RunResult) -> dict:
    m = res.metrics
    return {"task_clean_acc": m["task_clean"]["acc"],
            "task_adv_acc": m["task_adv"]["acc"],
            "det_clean_acc": m["det_clean_acc"],
            "det_adv_acc": m["det_adv_acc"],
            "clean_acc_e2e": m["clean_acc_e2e"],
            "robust_acc_e2e": m["robust_acc_e2e"],
            "val_score": res.best_val_score}


def bare_forward(model: lw.LWADSequential, x: torch.Tensor) -> torch.Tensor:
    """Composition of the layer.base modules alone: the backbone without
    detectors and activation losses."""
    for layer in model.layers:
        x = layer.base(x)
    return x


# ===========================================================================
# Statistics
# ===========================================================================
def s1_test(a: list, b: list, alternative: str = "two-sided") -> tuple[str, str]:
    """S1: paired t-test over seeds, p < 0.05 with at least 3 seeds.
    alternative="greater" tests mean(a - b) > 0."""
    n = len(a)
    if n < S1_MIN_SEEDS:
        return INCONCLUSIVE, f"{n} seed appaiati, ne servono >= {S1_MIN_SEEDS}"
    res = scipy.stats.ttest_rel(a, b, alternative=alternative)
    p = float(res.pvalue)
    if math.isnan(p):
        return INCONCLUSIVE, "p non definito (differenze tutte nulle)"
    return (PASS if p < S1_P_VALUE else FAIL), f"t={res.statistic:.3f}, p={p:.4f}, n={n}"


def s2_equivalent(a: list, ref: list) -> tuple[Optional[bool], str]:
    """S2: |mean(A - B)| <= 2 sigma_ref + 1e-3, sigma_ref the std over the seeds
    of the reference. None when sigma_ref is undefined."""
    if len(ref) < 2:
        return None, "sigma_ref non definita con meno di 2 seed"
    delta = statistics.fmean(x - r for x, r in zip(a, ref))
    bound = 2 * statistics.stdev(ref) + S2_ABS_TOL
    return abs(delta) <= bound, f"|media(A-B)|={abs(delta):.4f}, limite {bound:.4f}"


# ===========================================================================
# Stage 1 results
# ===========================================================================
class Stage1:
    """Stage 1 rows (one per experiment and seed, errors dropped) and their
    checkpoints. A run recorded twice keeps its last row."""

    def __init__(self, paths: list[str], dataset: str):
        self.paths = paths
        self._rows: dict[tuple[str, int], dict] = {}
        for path in paths:
            with open(path, newline="") as fh:
                for r in csv.DictReader(fh):
                    if r.get("dataset") != dataset or r.get("error"):
                        continue
                    self._rows[(r["experiment"], int(r["seed"]))] = {
                        k: _parse_cell(v) for k, v in r.items()}

    def seeds(self, name: str) -> list[int]:
        return sorted(s for e, s in self._rows if e == name)

    def row(self, name: str, seed: int) -> Optional[dict]:
        return self._rows.get((name, seed))

    def checkpoint(self, name: str, seed: int) -> Optional[str]:
        r = self.row(name, seed)
        if not r or not r.get("checkpoint"):
            return None
        p = r["checkpoint"]
        p = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
        return p if os.path.exists(p) else None

    def ckpt_seeds(self, name: str) -> list[int]:
        return [s for s in self.seeds(name) if self.checkpoint(name, s)]

    def paired_seeds(self, a: str, b: str) -> list[int]:
        return sorted(set(self.seeds(a)) & set(self.seeds(b)))

    def column(self, name: str, key: str, seeds, layer: Optional[int] = None) -> list:
        vals = [self.row(name, s)[key] for s in seeds]
        return vals if layer is None else [v[layer] for v in vals]


# ===========================================================================
# FurtherAL hinge activity during training
# ===========================================================================
@contextlib.contextmanager
def hinge_tracker():
    """Counts the pairs whose FurtherAL hinge is active (s < m, s the
    real/adv distance of the pair and m the layer margin), epoch by epoch.

    Patches three functions while the block runs:
      lx.train_with_early_stopping  opens the recording window, which leaves
                                    the margin warmup out (also lt.train_epoch)
      lt.train_epoch                opens a new epoch
      FurtherAL.distance_to_loss    counts the active pairs of each layer
    """
    rec = {"on": False, "epochs": []}
    orig = (lx.train_with_early_stopping, lt.train_epoch, lw.FurtherAL.distance_to_loss)

    def train_with_early_stopping(*args, **kwargs):
        rec["on"] = True
        try:
            return orig[0](*args, **kwargs)
        finally:
            rec["on"] = False

    def train_epoch(*args, **kwargs):
        if rec["on"]:
            rec["epochs"].append({})
        return orig[1](*args, **kwargs)

    def distance_to_loss(self, d):
        if rec["on"] and rec["epochs"]:
            slot = rec["epochs"][-1].setdefault(id(self), [0, 0])
            slot[0] = slot[0] + (d.detach() < self.margin).sum()  # stays on GPU
            slot[1] += d.numel()
        return orig[2](self, d)

    lx.train_with_early_stopping, lt.train_epoch = train_with_early_stopping, train_epoch
    lw.FurtherAL.distance_to_loss = distance_to_loss
    try:
        yield rec
    finally:
        lx.train_with_early_stopping, lt.train_epoch = orig[0], orig[1]
        lw.FurtherAL.distance_to_loss = orig[2]


def hinge_fractions(rec: dict) -> tuple[list[float], list[list[float]]]:
    """Share of active pairs per epoch (pooled over the FurtherAL layers) and
    per epoch and layer, in network order."""
    per_epoch, per_layer = [], []
    for ep in rec["epochs"]:
        counts = [(int(a), n) for a, n in ep.values()]
        tot = sum(n for _, n in counts)
        per_epoch.append(sum(a for a, _ in counts) / tot if tot else float("nan"))
        per_layer.append([a / n for a, n in counts])
    return per_epoch, per_layer


# ===========================================================================
# Session: data, models and runs shared by the tests
# ===========================================================================
def train_model(cfg: lc.ModelConfig, bundle: lx.DatasetBundle, X: torch.Tensor,
                y: torch.Tensor, epochs: int, seed: int = SEED) -> lw.LWADSequential:
    """cfg built at seed and trained for `epochs` lt.train_epoch epochs on
    (X, y), with the class weights and the attack mask of the bundle, in eval
    mode. The seed fixes both the initial weights and the batch order."""
    torch.manual_seed(seed)
    built = lc.create_model(cfg, bundle.n_features, device=DEVICE)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y),
                                         batch_size=cfg.batch_size, shuffle=True)
    for _ in range(epochs):
        lt.train_epoch(built.model, loader, built.optimizer, eps=cfg.eps,
                       lambda_det=getattr(cfg, "lambda_det", 0.0),
                       lambda_act=cfg.lambda_act,
                       task_loss_on_adv=cfg.task_loss_on_adv,
                       class_weights=bundle.class_weights,
                       attack_mask=bundle.attack_mask,
                       attack=cfg.train_attack,
                       threshold_det=getattr(cfg, "threshold_det", lc.DEFAULT_THRESHOLD_DET),
                       attack_kwargs=cfg.attack_kwargs(), device=DEVICE,
                       reduce=cfg.score_reduce)
    built.model.eval()
    return built.model


class Session:
    """Everything the tests share, built lazily: a test selected alone with
    --only pays only for what it uses."""

    def __init__(self, args):
        self.args = args
        self.tmp = tempfile.mkdtemp(prefix="lwad_vv_")
        self.store: dict = {}
        self._bundle: Optional[lx.DatasetBundle] = None
        self._stage1: Optional[Stage1] = None

    @property
    def bundle(self) -> lx.DatasetBundle:
        if self._bundle is None:
            progress("caricamento di UNSW-NB15 (una volta per sessione)")
            with quiet():
                self._bundle = lx.DatasetBundle.load(
                    DATASET, DEVICE, test_max_rows=self.args.test_max_rows or None)
        return self._bundle

    @property
    def stage1(self) -> Stage1:
        if self._stage1 is None:
            self._stage1 = Stage1(self.args.s1_csv, DATASET.value)
        return self._stage1

    @property
    def min_seeds(self) -> int:
        return self.args.min_seeds

    def rows(self, split: str, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """n rows of a split, picked with a generator at seed 0: the same
        subset in every test (and its first rows for a smaller n)."""
        key = ("rows", split, n)
        if key not in self.store:
            X, y = getattr(self.bundle, f"X_{split}"), getattr(self.bundle, f"y_{split}")
            idx = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:n]
            idx = idx.to(X.device)
            self.store[key] = (X[idx], y[idx])
        return self.store[key]

    def small_model(self, kind: str, trained: bool) -> lw.LWADSequential:
        """det / fur / adv small model, untrained or trained for SMALL_EPOCHS
        epochs on the fixed train subset, in eval mode."""
        key = ("small", kind, trained)
        if key not in self.store:
            if trained:
                progress(f"addestramento del modello piccolo {kind} "
                         f"({SMALL_EPOCHS} epoche su {SUBSET_ROWS} righe)")
            X, y = self.rows("train", SUBSET_ROWS)
            self.store[key] = train_model(SMALL[kind], self.bundle, X, y,
                                          SMALL_EPOCHS if trained else 0)
        return self.store[key]

    def run_suite(self, exps: list, tag: str, seeds=(SEED,),
                  infer: int = MINI_INFER_SAMPLES) -> lx.SuiteResult:
        """One lx.run_suite on the session bundle, with the margin cache
        cleared first. Each call writes into its own directory, so two suites
        never share checkpoint files."""
        progress(f"suite [{', '.join(e.name for e in exps)}] seed {list(seeds)}")
        lx.clear_margin_cache()
        with quiet():
            return lx.run_suite(exps, [self.bundle], device=DEVICE,
                                out_dir=os.path.join(self.tmp, tag), seeds=seeds,
                                verbose=0, timed_infer_samples=infer,
                                on_error="raise")

    def reference_run(self, kind: str) -> lx.RunResult:
        """DET_S ("det") or ADV_S ("adv") trained once at seed 42 and shared by
        the V4 tests that only need a trained run and its checkpoint."""
        key = ("reference", kind)
        if key not in self.store:
            cfg = {"det": DET_S, "adv": ADV_S}[kind]
            res = self.run_suite([lx.Exp(f"vv/ref-{kind}", cfg)], f"ref-{kind}")
            self.store[key] = res.results[0]
        return self.store[key]

    def ckpt(self, name: str, seed: int) -> lcp.LoadedCheckpoint:
        """Stage 1 checkpoint, loaded once on the GPU."""
        key = ("ckpt", name, seed)
        if key not in self.store:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")    # fields added after saving
                self.store[key] = lcp.load_checkpoint(self.stage1.checkpoint(name, seed),
                                                      device=DEVICE)
        return self.store[key]

    def replay(self, name: str, seed: int) -> dict:
        """Retrains a stage 1 run with hinge_tracker on. The hinge activity is
        a training quantity the CSV does not store, so it is measured on a
        replay; "faithful" says whether the replay reproduced the saved
        checkpoint (None when there is none to compare with)."""
        key = ("replay", name, seed)
        if key not in self.store:
            progress(f"replay di {name} seed {seed}: riaddestramento strumentato")
            lx.clear_margin_cache()
            with hinge_tracker() as rec, quiet():
                res = lx.run_experiment(s1_experiment(name), self.bundle, device=DEVICE,
                                        seed=seed, verbose=0,
                                        checkpoint_dir=os.path.join(self.tmp, "replay"),
                                        timed_infer_samples=0)
            per_epoch, per_layer = hinge_fractions(rec)
            saved = self.stage1.checkpoint(name, seed)
            self.store[key] = {"epochs": per_epoch, "layers": per_layer, "result": res,
                               "faithful": (not diff_checkpoints(res.checkpoint, saved))
                                           if saved else None}
        return self.store[key]

    def close(self) -> None:
        if self.args.keep_tmp:
            print(f"\nfile temporanei conservati in {self.tmp}")
        else:
            shutil.rmtree(self.tmp, ignore_errors=True)


def enough_seeds(t: Report, S: Session, label: str, seeds: list) -> bool:
    """Validation on stage 1 needs at least S.min_seeds seeds (3 by spec)."""
    if len(seeds) >= S.min_seeds:
        return True
    t.inconclusive(f"{label}: seed dello stadio 1",
                   f"{len(seeds)} disponibili {seeds}, ne servono >= {S.min_seeds}")
    return False


# ===========================================================================
# Runner
# ===========================================================================
def _fmt_duration(s: float) -> str:
    return f"{s:.1f}s" if s < 60 else f"{int(s // 60)}m{int(s % 60):02d}s"


def _print_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    for r in [headers, ["-" * w for w in widths]] + rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def print_stage1_inventory(S: Session) -> None:
    st = S.stage1
    print("\n== Dati dello stadio 1 ==")
    if not st.paths:
        print("   nessun CSV trovato (atteso results/summary_s1*.csv): "
              "le parti reali risulteranno SKIP o INCONCLUSIVE")
    for p in st.paths:
        print(f"   CSV: {os.path.relpath(p, REPO_ROOT)}")
    for name in S1_REQUIRED:
        seeds = st.seeds(name)
        status = (f"seed {seeds}, checkpoint {len(st.ckpt_seeds(name))}/{len(seeds)}"
                  if seeds else "ASSENTE")
        print(f"   {name:28s} {status}")
    missing = [n for n in S1_REQUIRED if not st.seeds(n)]
    if missing:
        print(f"   {len(missing)} esperimenti assenti: definizioni in lwad_stage1.table()")


def run_test(spec: TestSpec, S: Session) -> Report:
    print(f"\n== {spec.tid}) {spec.title} [{spec.kind}] ==")
    print(f"   {spec.desc}")
    rep = Report(spec)
    started = time.time()
    try:
        spec.fn(rep, S)
    except Exception as e:
        rep.outcome("eccezione non gestita", ERROR, f"{type(e).__name__}: {_short(e, 120)}")
        for line in traceback.format_exc().rstrip().splitlines():
            print(f"      {line}")
    rep.duration = time.time() - started
    if not rep.checks:
        rep.skip("nessun controllo eseguito", "")
    print(f"   => {spec.tid}: {rep.verdict} ({rep.counts()}, {_fmt_duration(rep.duration)})")
    return rep


def print_summary(reports: list[Report], elapsed: float) -> None:
    print("\n== RESOCONTO FINALE ==")
    _print_table([[r.spec.tid, r.spec.kind, r.verdict, r.counts(),
                   _fmt_duration(r.duration), r.spec.title] for r in reports],
                 ["id", "tipo", "esito", "controlli", "durata", "test"])
    print()
    for kind in (VERIFY, VALIDATE):
        c = Counter(r.verdict for r in reports if r.spec.kind == kind)
        if c:
            print(f"   {kind:12s}: " + ", ".join(f"{c[o]} {o}" for o in
                  (PASS, FAIL, INCONCLUSIVE, SKIP, ERROR) if c[o]))
    print(f"   durata totale: {_fmt_duration(elapsed)}")

    for title, wanted in (("controlli non superati", (FAIL, ERROR)),
                          ("controlli inconclusivi", (INCONCLUSIVE,))):
        lines = [f"   {r.spec.tid:5s} [{o}] {label}" + (f" -> {detail}" if detail else "")
                 for r in reports for o, label, detail in r.checks if o in wanted]
        if lines:
            print(f"\n{title}:")
            print("\n".join(lines))

    verdicts = Counter(r.verdict for r in reports)
    bad = verdicts[FAIL] + verdicts[ERROR]
    if bad:
        print(f"\nESITO: {bad} TEST NON SUPERATI")
    elif verdicts[INCONCLUSIVE]:
        print(f"\nESITO: NESSUN FALLIMENTO, {verdicts[INCONCLUSIVE]} TEST INCONCLUSIVI")
    else:
        print("\nTUTTI I TEST SUPERATI")


def main(tests: list[TestSpec], argv=None) -> int:
    parser = argparse.ArgumentParser(description="Suite V&V LWAD")
    parser.add_argument("--only", default=None,
                        help="id separati da virgola, per prefisso: V1,V2.4,A,M3")
    parser.add_argument("--s1-csv", action="append", default=None,
                        help="CSV dello stadio 1 (ripetibile); "
                             "default results/summary_s1*.csv")
    parser.add_argument("--test-max-rows", type=int, default=pp.DEFAULT_TEST_MAX_ROWS,
                        help="cap del test split, lo stesso della campagna")
    parser.add_argument("--min-seeds", type=int, default=S1_MIN_SEEDS,
                        help="seed dello stadio 1 richiesti dalla validazione "
                             "(spec: 3; S1 ne richiede comunque 3)")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="conserva CSV e checkpoint delle mini-campagne")
    args = parser.parse_args(argv)
    if args.s1_csv is None:
        args.s1_csv = sorted(glob.glob(os.path.join(REPO_ROOT, "results", "summary_s1*.csv")))

    prefixes = [p.strip() for p in (args.only or "").split(",") if p.strip()]
    selected = [s for s in tests if not prefixes or any(s.tid.startswith(p) for p in prefixes)]
    if not selected:
        print(f"nessun test corrisponde a --only={args.only!r}")
        return 2

    cuda = torch.cuda.is_available()
    print("== Suite V&V LWAD ==")
    print(f"   torch {torch.__version__}, device {DEVICE}"
          + (f" ({torch.cuda.get_device_name(0)})" if cuda else " NON disponibile"))
    print(f"   determinismo: CUBLAS_WORKSPACE_CONFIG={os.environ['CUBLAS_WORKSPACE_CONFIG']}, "
          f"use_deterministic_algorithms={torch.are_deterministic_algorithms_enabled()}, "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark}")
    print(f"   test selezionati ({len(selected)}): {', '.join(s.tid for s in selected)}")

    started = time.time()
    if not cuda:
        print("\n   GPU non disponibile: tutti i test vengono saltati, senza ripiegare sulla CPU")
        reports = []
        for spec in selected:
            rep = Report(spec)
            rep.checks.append((SKIP, "GPU non disponibile", ""))
            reports.append(rep)
        print_summary(reports, time.time() - started)
        return 0

    S = Session(args)
    try:
        if any(s.tid.startswith(("V4", "A", "M")) for s in selected):
            print_stage1_inventory(S)
        reports = [run_test(spec, S) for spec in selected]
    finally:
        S.close()
    print_summary(reports, time.time() - started)
    return 1 if any(r.verdict in (FAIL, ERROR) for r in reports) else 0
