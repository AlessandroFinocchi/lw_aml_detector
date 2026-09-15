"""Runs the experiment suite declared in libs/lwad_experiments.py.

To change WHAT is run, edit the EXPERIMENTS table in that module: this file only
decides how the suite is executed.

    make exp                      # nohup + res.log, as before
    python notebooks/experiment.py --dry-run
    python notebooks/experiment.py --only det/ --resume
"""
import argparse

import libs.preprocess.preprocess as pp
import libs.lwad_config as lc
import libs.lwad_experiments as lx

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", default=None,
                    help="run only the experiments whose name starts with this prefix")
parser.add_argument("--dataset", default=None, choices=[d.name for d in pp.KaggleDataset],
                    help="restrict the suite to a single dataset")
parser.add_argument("--dry-run", action="store_true",
                    help="list the runs without training anything")
parser.add_argument("--resume", action="store_true",
                    help="skip the runs already recorded in summary.csv")
parser.add_argument("--download", action="store_true", help="download the raw datasets")
parser.add_argument("--no-checkpoints", action="store_true")
parser.add_argument("--out-dir", default="results")
parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
parser.add_argument("--seeds", default=str(lc.SEED),
                    help="comma separated seeds, e.g. 42,43,44: every experiment "
                         "is repeated once per seed, which is what tells a real "
                         "difference from run to run noise")
parser.add_argument("--test-max-rows", type=int, default=pp.DEFAULT_TEST_MAX_ROWS,
                    help="cap on the test split, balancing the attack categories (0 disables it)")
args = parser.parse_args()

datasets = ([pp.KaggleDataset[args.dataset]] if args.dataset else lx.DATASETS)

seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
if not seeds:
    parser.error(f"--seeds: no seed parsed out of {args.seeds!r}")

lx.run_suite(lx.EXPERIMENTS, datasets,
             out_dir=args.out_dir, only=args.only, dry_run=args.dry_run,
             resume=args.resume, download=args.download, seeds=seeds,
             test_max_rows=args.test_max_rows or None,
             save_checkpoints=not args.no_checkpoints, verbose=args.verbose)
