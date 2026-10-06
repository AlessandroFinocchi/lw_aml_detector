"""V&V suite: all the tests of the spec, in spec order, with the final report.

    Verifica     V1.1-V1.6, V2.1-V2.5   modello e attacchi       (vv_verify_model)
                 V4.1-V4.6              coerenza di sistema      (vv_verify_system)
    Validazione  A3, A1, A2, A4         robustezza               (vv_validate_robustness)
                 M1-M7                  meccanismi del metodo    (vv_validate_mechanisms)

Verification ends in PASS or FAIL, validation in PASS, FAIL or INCONCLUSIVE;
a part that cannot run (stage 1 data missing, no GPU) is a SKIP. The exit code
is 1 when some test ends in FAIL or ERROR.

Run (from the repo root):

    python -m libs.tests.vv_suite
    python -m libs.tests.vv_suite --only V1,V2
    python -m libs.tests.vv_suite --only A --s1-csv results/summary_s1_UNSW_BW15.csv
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import sys

import libs.tests.vv_verify_model as vm
import libs.tests.vv_verify_system as vs
import libs.tests.vv_validate_robustness as vr
import libs.tests.vv_validate_mechanisms as vme

TESTS = vm.TESTS + vs.TESTS + vr.TESTS + vme.TESTS


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
