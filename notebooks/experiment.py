"""Runs the experiment suite declared in libs/lwad_experiments.py.

To change WHAT is run, edit the EXPERIMENTS table in that module: this file only
decides how the suite is executed.

    make exp                      # nohup + res.log, as before
    python notebooks/experiment.py --dry-run
    python notebooks/experiment.py --only det/ --resume
    python notebooks/experiment.py --only s2/ --summary-file summary_s2.csv
"""
import argparse

import libs.preprocess.preprocess as pp
import libs.model.lwad_config as lc
import libs.experiments.lwad_experiments as lx

# appending stage experiments to lx.EXPERIMENTS
import libs.experiments.lwad_stage1            
import libs.experiments.lwad_stage2
import libs.experiments.lwad_stage3
import libs.experiments.lwad_stage4
import libs.experiments.lwad_stage5

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--only", default=None,
                    help="run only the experiments whose name starts with this prefix")
parser.add_argument("--dataset", default=None, choices=[d.name for d in pp.KaggleDataset],
                    help="restrict the suite to a single dataset")
parser.add_argument("--dry-run", action="store_true",
                    help="list the runs without training anything")
parser.add_argument("--resume", action="store_true",
                    help="skip the runs already recorded in the summary file")
parser.add_argument("--download", action="store_true", help="download the raw datasets")
parser.add_argument("--no-checkpoints", action="store_true")
parser.add_argument("--out-dir", default="results")
parser.add_argument("--summary-file", default="summary.csv",
                    help="name of the csv written inside --out-dir")
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
             out_dir=args.out_dir, summary_file=args.summary_file,
             only=args.only, dry_run=args.dry_run,
             resume=args.resume, download=args.download, seeds=seeds,
             test_max_rows=args.test_max_rows or None,
             save_checkpoints=not args.no_checkpoints, verbose=args.verbose)
