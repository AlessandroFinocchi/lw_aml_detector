"""V&V suite - Verifica: modello e attacchi (V1.1-V1.6, V2.1-V2.4).

V1 checks the construction rules and that the wrappers leave the backbone
untouched; V2 checks the attacks on the small models trained for 3 epochs on
the fixed 4096-row train subset.

Run:  python -m libs.tests.vv_verify_model [--only V2]
"""
import libs.tests.vv_common as vc    # first: sets up determinism before any CUDA op

import math
import sys
from dataclasses import replace

import torch
import torch.nn as nn

import libs.model.lwad_wrapper as lw
import libs.model.lwad_config as lc
import libs.attacks.lwad_attack as la
import libs.training.lwad_trainer as lt
import libs.training.lwad_margin as lm
import libs.evaluation.lwad_evaluator as le
import libs.experiments.lwad_experiments as lx

TESTS: list = []

GRAD_NOISE = 1e-8           # V1.4: a gradient under it is rounding, not signal

# --- early exit (V1.5) ------------------------------------------------------
EE_SAMPLES = 512
EE_SCORE_TOL = 1e-6

# --- margins (V1.6) ---------------------------------------------------------
MARGIN_FACTORS = (2.0, 20.0)

# --- attack setup (V2) ------------------------------------------------------
V2_SAMPLES = 128
V2_EPS = (0.05, 0.2)
V2_STEPS = 10


# ===========================================================================
# V1 - model
# ===========================================================================
@vc.vv_test(TESTS, "V1.1", "Vincoli di costruzione", vc.VERIFY,
            "Combinazioni incoerenti di layer, flag e opzioni devono sollevare ValueError.")
def v1_1(t:vc.Report, S:vc.Session):
    x, y = S.rows("train", vc.SUBSET_ROWS)
    mask, nf = S.bundle.attack_mask, S.bundle.n_features

    # 1) CloserAL never shares a network with a detector
    t.raises("LWADSequential con CloserAL + DetectorLayer", ValueError,
             lambda: lw.LWADSequential(
                 lw.DetectorLayer(nn.Linear(8, 8), detector=lw.default_detector(8)),
                 nn.ReLU(), lw.CloserAL(nn.Linear(8, 2))))
    t.raises("LWADSequential con CloserAL + FurtherAL", ValueError,
             lambda: lw.LWADSequential(
                 lw.FurtherAL(nn.Linear(8, 8), detector=lw.default_detector(8)),
                 nn.ReLU(), lw.CloserAL(nn.Linear(8, 2))))

    # 2) the activation loss needs every sample in both versions
    adv = S.small_model("adv", trained=False)
    flag = torch.cat([torch.zeros(10), torch.ones(22)]).to(x.device)
    t.raises("forward CloserAL con is_adv sbilanciato (10 zeri, 22 uno)", ValueError,
             lambda: adv(x[:32], is_adv=flag))

    # 3) one margin per FurtherAL, in network order
    t.raises("act_margin con 2 valori su 3 FurtherAL", ValueError,
             lambda: replace(vc.FUR, act_margin=(0.5, 0.05)).build_model(nf))
    t.raises("act_margin con 4 valori su 3 FurtherAL", ValueError,
             lambda: replace(vc.FUR, act_margin=(0.5, 0.05, 0.1, 0.2)).build_model(nf))
    model = replace(vc.FUR, wrap_at=(0, 2), act_margin=(0.5, 0.05)).build_model(nf)
    ms = [layer.margin for layer in model.layers if isinstance(layer, lw.FurtherAL)]
    t.check("act_margin=(0.5, 0.05) su 2 FurtherAL applicato nell'ordine della rete",
            ms == [0.5, 0.05], f"margini {ms}")

    # 4) early exit only under "max" and only with detectors
    det = S.small_model("det", trained=False)
    t.raises("DetectorModelConfig(early_exit=True, score_reduce='mean')", ValueError,
             lambda: lc.DetectorModelConfig(early_exit=True, score_reduce="mean"))
    t.raises("le.predict con early_exit=True e reduce='mean'", ValueError,
             lambda: le.predict(det, x[:8], reduce="mean", early_exit=True))
    t.raises("le.predict con early_exit=True su un modello senza detector", ValueError,
             lambda: le.predict(adv, x[:8], reduce="max", early_exit=True))

    # 5) detector-only procedures
    t.raises("pgd_adaptive su un modello senza detector", ValueError,
             lambda: la.generate_attack(adv, x[:32], y[:32], 0.2, "pgd_adaptive", mask=mask))
    t.raises("lt.select_threshold su un modello senza detector", ValueError,
             lambda: lt.select_threshold(adv, x[:32], y[:32], eps=0.2, attack_mask=mask))

    # class hierarchy
    t.check("FurtherAL sottoclasse di DetectorLayer e di ActivationLoss",
            issubclass(lw.FurtherAL, lw.DetectorLayer) and issubclass(lw.FurtherAL, lw.ActivationLoss))
    mro = lw.FurtherAL.__mro__[1:4]
    t.check("MRO di FurtherAL nell'ordine atteso (DetectorLayer, ActivationLoss, PassThrough)",
            mro == (lw.DetectorLayer, lw.ActivationLoss, lw.PassThrough),
            " -> ".join(c.__name__ for c in mro))
    t.check("AdvTrainingModelConfig().early_exit is False",
            lc.AdvTrainingModelConfig().early_exit is False)


@vc.vv_test(TESTS, "V1.2", "Validazioni di configurazione", vc.VERIFY,
            "Indici, larghezze e opzioni fuori dominio devono essere rifiutati dalla config.")
def v1_2(t:vc.Report, S:vc.Session):
    nf = S.bundle.n_features
    for cls in (lc.DetectorModelConfig, lc.AdvTrainingModelConfig):
        n = cls.__name__
        for w in ((2,), (-3,)):
            t.raises(f"{n}: wrap_at={w} con hidden_dims=(8, 8), build_model", ValueError,
                     lambda: cls(hidden_dims=(8, 8), wrap_at=w).build_model(nf))
        r = cls(hidden_dims=(8, 8), wrap_at=(-1, 1, 0, 0)).resolved_wrap_at()
        t.check(f"{n}: wrap_at=(-1, 1, 0, 0) -> resolved_wrap_at() == (0, 1)",
                r == (0, 1), f"ottenuto {r}")
        for h in ((), (8, 0), (8, -4)):
            t.raises(f"{n}: hidden_dims={h}, build_model", ValueError,
                     lambda: cls(hidden_dims=h).build_model(nf))
        t.raises(f"{n}: score_mode='foo'", ValueError, lambda: cls(score_mode="foo"))

    t.raises("DetectorModelConfig: detector_dims=(16, 0), build_model", ValueError,
             lambda: lc.DetectorModelConfig(detector_dims=(16, 0)).build_model(nf))
    t.raises("DetectorModelConfig: score_reduce='median'", ValueError,
             lambda: lc.DetectorModelConfig(score_reduce="median"))
    for m in (0, -1):
        t.raises(f"Exp('t', DetectorModelConfig(), margin_factor={m}).validate()", ValueError,
                 lambda: lx.Exp("t", lc.DetectorModelConfig(), margin_factor=m).validate())
    t.raises("lc.ModelConfig() (classe astratta)", TypeError, lambda: lc.ModelConfig())


def _paired_batch(S, n: int = 32):
    x, _ = S.rows("train", vc.SUBSET_ROWS)
    x = x[:n]
    flag = torch.cat([torch.zeros(n), torch.ones(n)]).to(x.device)
    return x, torch.cat([x, x + 0.05]), flag


@vc.vv_test(TESTS, "V1.3", "I wrapper non alterano la backbone", vc.VERIFY,
            "Su det, fur e adv non addestrati i logit coincidono con la sola backbone e "
            "ogni detector legge la pre-attivazione del suo base.")
def v1_3(t:vc.Report, S:vc.Session):
    x, xb, flag = _paired_batch(S)

    # FurtherAL adds no parameter: from the same seed det and fur start from
    # the same weights, so comparing them measures the activation loss alone
    sd_d = S.small_model("det", trained=False).state_dict()
    sd_f = S.small_model("fur", trained=False).state_dict()
    t.check("det e fur non addestrati: stesse chiavi e stessi tensori nello state_dict",
            sd_d.keys() == sd_f.keys() and all(torch.equal(sd_d[k], sd_f[k]) for k in sd_d),
            f"{len(sd_d)} vs {len(sd_f)} tensori")

    for kind in ("det", "fur", "adv"):
        model = S.small_model(kind, trained=False)
        with torch.no_grad():
            logits, _ = model(x)
            logits_p, _ = model(xb, is_adv=flag)
            t.check(f"{kind}: logit di model(x) identici alla composizione dei layer.base",
                    torch.equal(logits, vc.bare_forward(model, x)))
            t.check(f"{kind}: idem con is_adv su batch accoppiato",
                    torch.equal(logits_p, vc.bare_forward(model, xb)))

        if not model.has_detectors:
            continue
        # what each detector receives, against what its base produced
        seen, hooks = {}, []
        for i, layer in enumerate(model.layers):
            if isinstance(layer, lw.DetectorLayer):
                hooks.append(layer.base.register_forward_hook(
                    lambda m, inp, out, i=i: seen.setdefault(i, {}).update(base=out)))
                hooks.append(layer.detector.register_forward_pre_hook(
                    lambda m, inp, i=i: seen.setdefault(i, {}).update(det=inp[0])))
        try:
            with torch.no_grad():
                model(x)
        finally:
            for h in hooks:
                h.remove()
        for i, s in sorted(seen.items()):
            same = torch.equal(s["det"], s["base"])
            neg = int((s["det"] < 0).sum())
            nxt = model.layers[i + 1].base if i + 1 < len(model.layers) else None
            relu = isinstance(nxt, nn.ReLU)
            t.check(f"{kind}: layer {i} ({type(model.layers[i]).__name__}) input del detector "
                    f"= uscita del base, con negativi, seguito da ReLU",
                    same and neg > 0 and relu,
                    f"identico={same}, negativi={neg}/{s['det'].numel()}, "
                    f"successivo={type(nxt).__name__}")


@vc.vv_test(TESTS, "V1.4", "Contenuto di FlowState", vc.VERIFY,
            "Loss, detection e exit_layer prodotti dal forward, con e senza is_adv, "
            "su cat([x, x + 0.05]); il gradiente di act_loss arriva al primo Linear.")
def v1_4(t:vc.Report, S:vc.Session):
    _, xb, flag = _paired_batch(S)
    B = len(xb)
    # model: (det_loss is a tensor, act_loss is a tensor, number of detections)
    expected = {"fur": (True, True, 3), "det": (True, False, 3), "adv": (False, True, 0)}
    for kind, (want_det, want_act, n_det) in expected.items():
        model = S.small_model(kind, trained=False)
        with torch.no_grad():
            _, st = model(xb, is_adv=flag)
            _, st0 = model(xb)
            # reference detections, computed layer by layer in network order
            h, refs = xb, []
            for layer in model.layers:
                h = layer.base(h)
                if isinstance(layer, lw.DetectorLayer):
                    refs.append(layer.detector(h))

        def what(v):
            return "Tensore" if isinstance(v, torch.Tensor) else repr(v)

        t.check(f"{kind}: con is_adv det_loss {'Tensore' if want_det else 'None'}",
                isinstance(st.det_loss, torch.Tensor) if want_det else st.det_loss is None,
                what(st.det_loss))
        t.check(f"{kind}: con is_adv act_loss {'Tensore' if want_act else 'None'}",
                isinstance(st.act_loss, torch.Tensor) if want_act else st.act_loss is None,
                what(st.act_loss))
        t.check(f"{kind}: con is_adv len(detections) == {n_det}",
                len(st.detections) == n_det, f"{len(st.detections)}")
        t.check(f"{kind}: senza is_adv det_loss e act_loss None",
                st0.det_loss is None and st0.act_loss is None,
                f"det_loss={what(st0.det_loss)}, act_loss={what(st0.act_loss)}")
        if n_det:
            t.check(f"{kind}: senza is_adv {n_det} detection",
                    len(st0.detections) == n_det, f"{len(st0.detections)}")
            shapes = {tuple(d.shape) for d in st.detections + st0.detections}
            t.check(f"{kind}: ogni detection ha forma (B, 1) = ({B}, 1)",
                    shapes == {(B, 1)}, f"forme {sorted(shapes)}")
            in_order = all(torch.equal(a, r) for a, r in zip(st.detections, refs)) and \
                       all(torch.equal(a, r) for a, r in zip(st0.detections, refs))
            t.check(f"{kind}: detection nell'ordine della rete", in_order)
        else:
            t.check(f"{kind}: senza is_adv adv_score() None", st0.adv_score() is None)
        t.check(f"{kind}: senza exit_threshold state.exit_layer None",
                st.exit_layer is None and st0.exit_layer is None)

        # the activation loss trains the backbone, down to its first Linear.
        # Not on xb: the input LayerNorm cancels the constant shift of
        # x + 0.05, real and adv coincide and the gradient is rounding noise
        if want_act:
            x0 = xb[:B // 2]
            sign = torch.ones(x0.shape[1], device=x0.device)
            sign[1::2] = -1
            _, st_g = model(torch.cat([x0, x0 + 0.05 * sign]), is_adv=flag)
            first = next(layer.base for layer in model.layers
                         if isinstance(layer.base, nn.Linear))
            (g,) = torch.autograd.grad(st_g.act_loss, first.weight, allow_unused=True)
            g_max = 0.0 if g is None else g.abs().max().item()
            t.check(f"{kind}: con x + 0.05 * (+1, -1, ...) il gradiente di act_loss raggiunge "
                    f"il primo Linear, sopra il rumore numerico ({GRAD_NOISE:g})",
                    g_max > GRAD_NOISE, "None" if g is None else f"max |g| = {g_max:.3g}")


@vc.vv_test(TESTS, "V1.5", "Early exit equivalente al forward completo", vc.VERIFY,
            "Con reduce='max' vale max_i p_i > t <=> qualche p_i > t: un campione alla volta "
            "l'early exit segnala gli stessi campioni del forward completo e lascia label e score "
            f"di chi non esce ({EE_SAMPLES} campioni di test, puliti e adversarial); un batch "
            "esce solo se tutti i suoi campioni sono segnalati.")
def v1_5(t:vc.Report, S:vc.Session):
    X, y = S.rows("test", EE_SAMPLES)
    Xv, yv = S.rows("val", EE_SAMPLES)
    mask = S.bundle.attack_mask
    for kind in ("det", "fur"):
        model, cfg = S.small_model(kind, trained=True), vc.SMALL[kind]
        # threshold from the validation procedure, as run_experiment picks it
        torch.manual_seed(vc.SEED)
        thr, _ = lt.select_threshold(model, Xv, yv, eps=cfg.eps, attack_mask=mask,
                                     attack=cfg.train_attack,
                                     attack_kwargs=cfg.attack_kwargs(), reduce="max")
        torch.manual_seed(vc.SEED)
        x_adv = la.generate_attack(model, X, y, cfg.eps, cfg.eval_attack, mask=mask,
                                   **cfg.attack_kwargs())
        t.info(f"{kind}: soglia {thr:.3f} con reduce='max'")

        # 1) sample by sample, as le.inference_time runs
        exits = {}
        for name, data in (("puliti", X), ("adversarial", x_adv)):
            n_exit = bad_flag = bad_exit = bad_keep = 0
            for i in range(len(data)):
                x = data[i:i + 1]
                lab_f, sc_f, fl_f = le.predict(model, x, threshold_det=thr, reduce="max")
                lab_e, sc_e, fl_e = le.predict(model, x, threshold_det=thr, reduce="max",
                                               early_exit=True)
                bad_flag += bool(fl_f) != bool(fl_e)
                if lab_e is None:
                    # the partial max is a lower bound of the full one, still over t
                    n_exit += 1
                    bad_exit += (not bool(fl_f)) or float(sc_e) > float(sc_f) + EE_SCORE_TOL
                else:
                    bad_keep += not (torch.equal(lab_f, lab_e) and torch.allclose(sc_f, sc_e))
            exits[name] = n_exit
            t.check(f"{kind} {name}: flag identici al forward completo", bad_flag == 0,
                    f"{bad_flag} diversi su {len(data)}, {n_exit} uscite anticipate")
            t.check(f"{kind} {name}: chi esce e' segnalato, con score parziale <= completo",
                    bad_exit == 0, f"{bad_exit} violazioni su {n_exit} uscite")
            t.check(f"{kind} {name}: chi non esce ha label e score del forward completo",
                    bad_keep == 0, f"{bad_keep} diversi su {len(data) - n_exit}")
        t.check(f"{kind}: almeno un adversarial esce in anticipo (controllo non vuoto)",
                exits["adversarial"] > 0, f"{exits['adversarial']}/{len(x_adv)}")

        # 2) a batch exits only when every sample in it is flagged
        _, _, flags = le.predict(model, x_adv, threshold_det=thr, reduce="max")
        flagged, unflagged = x_adv[flags], x_adv[~flags]
        if len(flagged):
            lab, _, _ = le.predict(model, flagged, threshold_det=thr, reduce="max",
                                   early_exit=True)
            t.check(f"{kind}: batch di {len(flagged)} adversarial tutti segnalati esce "
                    f"(label None)", lab is None)
        if not (len(flagged) and len(unflagged)):
            t.skip(f"{kind}: batch misto", "servono campioni segnalati e non segnalati")
            continue
        mixed = torch.cat([flagged[:4], unflagged[:1]])
        lab, _, fl = le.predict(model, mixed, threshold_det=thr, reduce="max", early_exit=True)
        _, _, fl_full = le.predict(model, mixed, threshold_det=thr, reduce="max")
        t.check(f"{kind}: batch misto ({len(mixed) - 1} segnalati + 1 no) non esce, "
                f"flag come il forward completo",
                lab is not None and len(lab) == len(mixed) and torch.equal(fl, fl_full))


@vc.vv_test(TESTS, "V1.6", "Calibrazione e selezione dei margini", vc.VERIFY,
            "suggest_margins e select_margins danno un margine positivo per FurtherAL, pari a "
            "factor * distanza naturale, nell'ordine della rete e applicabile alla config; "
            "senza FurtherAL rispondono None e ValueError.")
def v1_6(t:vc.Report, S:vc.Session):
    X, y = S.rows("train", vc.SUBSET_ROWS)
    Xv, yv = S.rows("val", vc.SUBSET_ROWS)
    nf = S.bundle.n_features
    kw = dict(attack_mask=S.bundle.attack_mask, device=vc.DEVICE,
              class_weights=S.bundle.class_weights, verbose=False)
    n_fur = sum(isinstance(layer, lw.FurtherAL) for layer in vc.FUR.build_model(nf).layers)

    # 1) calibration: margin_i = factor * d_i, d_i measured after a warmup at
    #    a fixed seed, so two calls differ only by the factor
    vc.progress("suggest_margins su fur con factor 2 e 4")
    with vc.quiet():
        m2 = lm.suggest_margins(vc.FUR, X, y, factor=2.0, **kw)
        m4 = lm.suggest_margins(vc.FUR, X, y, factor=4.0, **kw)
    t.check(f"suggest_margins su fur: {n_fur} margini positivi e finiti",
            m2 is not None and len(m2) == n_fur and all(0 < v < math.inf for v in m2), f"{m2}")
    t.check("suggest_margins: factor=4 da' il doppio di factor=2 (stessa distanza naturale)",
            m2 is not None and m4 == tuple(2 * v for v in m2), f"{m2} -> {m4}")
    with vc.quiet():
        m_adv = lm.suggest_margins(vc.ADV, X, y, warmup_epochs=0, **kw)
    t.check("suggest_margins su adv (nessun FurtherAL): None", m_adv is None, f"{m_adv!r}")

    # 2) selection: one short training per factor, the best validation score wins
    vc.progress(f"select_margins su fur con i fattori {MARGIN_FACTORS}")
    with vc.quiet():
        res = lm.select_margins(vc.FUR, X, y, Xv, yv, factors=MARGIN_FACTORS,
                                search_epochs=1, **kw)
    scores = [c["score"] for c in res.candidates]
    t.check(f"select_margins: un candidato per fattore, nell'ordine {MARGIN_FACTORS}",
            tuple(c["factor"] for c in res.candidates) == MARGIN_FACTORS,
            f"{[c['factor'] for c in res.candidates]}")
    t.check("select_margins: vince il candidato con lo score di validazione massimo",
            res.factor in MARGIN_FACTORS and res.score == max(scores),
            f"factor {res.factor:g}, score {res.score:.4f}, candidati "
            f"{[round(s, 4) for s in scores]}")
    t.check(f"select_margins: {n_fur} margini, factor * distanze naturali",
            len(res.margins) == n_fur
            and res.margins == tuple(res.factor * d for d in res.base_distances),
            f"{res.margins}")
    model = replace(vc.FUR, act_margin=res.margins).build_model(nf)
    applied = tuple(layer.margin for layer in model.layers if isinstance(layer, lw.FurtherAL))
    t.check("margini selezionati applicati dalla config nell'ordine della rete",
            applied == res.margins, f"{applied}")
    t.raises("select_margins su adv (nessun FurtherAL)", ValueError,
             lambda: lm.select_margins(vc.ADV, X, y, Xv, yv, warmup_epochs=0, **kw))


# ===========================================================================
# V2 - attacks
# ===========================================================================
def _cases():
    """(model, attack, reduce): FGSM and PGD on the three models, adaptive
    PGD on the detector models with both reductions."""
    for kind in ("det", "fur", "adv"):
        for attack in ("fgsm", "pgd"):
            yield kind, attack, None
    for kind in ("det", "fur"):
        for reduce in ("mean", "max"):
            yield kind, "pgd_adaptive", reduce


def _name(kind, attack, reduce) -> str:
    return f"{kind} {attack}" + (f"[{reduce}]" if reduce else "")


def _attack(S, kind, attack, eps, mask, reduce=None, evade_weight=None):
    """One attack on the trained small model, seed fixed right before it."""
    model = S.small_model(kind, trained=True)
    x, y = S.rows("test", V2_SAMPLES)
    torch.manual_seed(vc.SEED)
    return la.generate_attack(model, x, y, eps, attack, mask=mask, steps=V2_STEPS,
                              alpha=2*eps / V2_STEPS, evade_weight=evade_weight, reduce=reduce)


@vc.vv_test(TESTS, "V2.1", "Ammissibilita' L-inf", vc.VERIFY,
            f"Con e senza maschera x_adv resta nella palla L-inf di raggio eps, con forma, "
            f"dtype e device di x, valori finiti e senza grafo ({V2_SAMPLES} campioni di test).")
def v2_1(t:vc.Report, S:vc.Session):
    x, _ = S.rows("test", V2_SAMPLES)
    for kind, attack, reduce in _cases():
        for eps in V2_EPS:
            problems, linf = [], {}
            for tag, mask in (("mask", S.bundle.attack_mask), ("no-mask", None)):
                xa = _attack(S, kind, attack, eps, mask, reduce)
                linf[tag] = (xa - x).abs().max().item()
                if linf[tag] > eps + vc.LINF_TOL:
                    problems.append(f"{tag}: L-inf > eps")
                if (xa.shape, xa.dtype, xa.device) != (x.shape, x.dtype, x.device):
                    problems.append(f"{tag}: forma/dtype/device")
                if not bool(torch.isfinite(xa).all()):
                    problems.append(f"{tag}: valori non finiti")
                if xa.requires_grad:
                    problems.append(f"{tag}: requires_grad")
            t.check(f"{_name(kind, attack, reduce):22s} eps={eps:<4g} ||x_adv - x||inf <= eps + 1e-5",
                    not problems,
                    f"mask {linf['mask']:.5f}, no-mask {linf['no-mask']:.5f}; "
                    + ("; ".join(problems) if problems else "forma/dtype/device/finiti/no-grad ok"))


@vc.vv_test(TESTS, "V2.2", "Maschera rispettata", vc.VERIFY,
            "Le feature non attaccabili restano intatte, almeno una attaccabile cambia, "
            "e con maschera nulla l'attacco non modifica nulla.")
def v2_2(t:vc.Report, S:vc.Session):
    x, _ = S.rows("test", V2_SAMPLES)
    mask = S.bundle.attack_mask
    frozen, free = mask == 0, mask != 0
    t.info(f"maschera UNSW-NB15: {int(frozen.sum())} colonne non attaccabili su {len(mask)}")
    if not bool(frozen.any()):
        t.skip("colonne mascherate", "la maschera non ha colonne a zero, controllo vuoto")
    for kind, attack, reduce in _cases():
        for eps in V2_EPS:
            xa = _attack(S, kind, attack, eps, mask, reduce)
            same = torch.equal(xa[:, frozen], x[:, frozen])
            changed = int((xa[:, free] != x[:, free]).any(dim=0).sum())
            t.check(f"{_name(kind, attack, reduce):22s} eps={eps:<4g} mascherate identiche, "
                    f"almeno una attaccabile cambia", same and changed > 0,
                    f"mascherate identiche={same}, attaccabili cambiate {changed}/{int(free.sum())}")
        xz = _attack(S, kind, attack, 0.2, torch.zeros_like(mask), reduce)
        t.check(f"{_name(kind, attack, reduce):22s} maschera tutta a zero -> x_adv identico a x",
                torch.equal(xz, x))


@vc.vv_test(TESTS, "V2.3", "Con beta = 0 l'attacco adattivo coincide con PGD", vc.VERIFY,
            "A parita' di seed pgd_adaptive con evade_weight=0 riproduce PGD bit per bit; "
            "con evade_weight=1 se ne discosta.")
def v2_3(t:vc.Report, S:vc.Session):
    mask = S.bundle.attack_mask
    for kind in ("det", "fur"):
        for eps in V2_EPS:
            pgd = _attack(S, kind, "pgd", eps, mask)
            for reduce in ("mean", "max"):
                a0 = _attack(S, kind, "pgd_adaptive", eps, mask, reduce, evade_weight=0.0)
                a1 = _attack(S, kind, "pgd_adaptive", eps, mask, reduce, evade_weight=1.0)
                n0, n1 = int((a0 != pgd).sum()), int((a1 != pgd).sum())
                t.check(f"{kind} eps={eps:<4g} [{reduce}] beta=0: identico a PGD",
                        torch.equal(a0, pgd), f"{n0} elementi diversi")
                t.check(f"{kind} eps={eps:<4g} [{reduce}] beta=1: diverso da PGD",
                        n1 > 0, f"{n1} elementi diversi")


@vc.vv_test(TESTS, "V2.4", "Arresti del gradiente durante l'attacco", vc.VERIFY,
            "Su det (detach=True) il detector e' staccato fuori dall'attacco, collegato dentro, "
            "i flag tornano come prima e l'attacco non tocca i pesi.")
def v2_4(t:vc.Report, S:vc.Session):
    model = S.small_model("det", trained=True)
    x, y = S.rows("test", V2_SAMPLES)
    layers = [m for m in model.modules() if isinstance(m, lw.DetectorLayer)]
    t.check("det: ogni DetectorLayer ha detach=True", all(layer.detach for layer in layers),
            f"{[layer.detach for layer in layers]}")

    # 1) gradient of the adversarial score with respect to the input
    def score_grad():
        xg = x.clone().requires_grad_(True)
        _, st = model(xg)
        s = st.adv_score().sum()
        if not s.requires_grad:
            return None
        (g,) = torch.autograd.grad(s, xg, allow_unused=True)
        return g

    g_out = score_grad()
    t.check("fuori dall'attacco: grad di adv_score().sum() rispetto a x None o nullo",
            g_out is None or not bool(g_out.any()),
            "None" if g_out is None else f"max |g| = {g_out.abs().max().item():.3g}")
    with la._detector_grad_enabled(model):
        g_in = score_grad()
    t.check("dentro _detector_grad_enabled: gradiente diverso da zero",
            g_in is not None and bool(g_in.any()),
            "None" if g_in is None else f"max |g| = {g_in.abs().max().item():.3g}")

    # 2) detach flags restored, mixed and after an exception
    initial = [layer.detach for layer in layers]
    mixed = [i % 2 == 0 for i in range(len(layers))]
    try:
        for layer, v in zip(layers, mixed):
            layer.detach = v
        with la._detector_grad_enabled(model):
            inside = [layer.detach for layer in layers]
        after = [layer.detach for layer in layers]
        t.check("flag misti: dentro il blocco detach=False ovunque", not any(inside), f"{inside}")
        t.check("flag misti: all'uscita tornano ai valori iniziali", after == mixed,
                f"{after}, attesi {mixed}")
        try:
            with la._detector_grad_enabled(model):
                raise RuntimeError("eccezione di prova")
        except RuntimeError:
            pass
        after = [layer.detach for layer in layers]
        t.check("dopo un'eccezione nel blocco i flag tornano ai valori iniziali",
                after == mixed, f"{after}, attesi {mixed}")
    finally:
        for layer, v in zip(layers, initial):
            layer.detach = v

    # 3) the attack leaves the weights and their gradients alone
    model.zero_grad(set_to_none=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    torch.manual_seed(vc.SEED)
    la.generate_attack(model, x, y, 0.2, "pgd_adaptive", mask=S.bundle.attack_mask,
                       steps=V2_STEPS, alpha=0.05)
    after_sd = model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after_sd[k])]
    t.check("dopo pgd_adaptive lo state_dict e' identico a prima",
            before.keys() == after_sd.keys() and not changed,
            f"{len(changed)} tensori cambiati" if changed else None)
    after = [layer.detach for layer in layers]
    t.check("dopo pgd_adaptive i flag detach sono quelli di prima", after == initial,
            f"{after}, attesi {initial}")
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    t.check("dopo pgd_adaptive ogni p.grad e' None", not grads,
            f"{len(grads)} parametri con grad" if grads else None)


if __name__ == "__main__":
    sys.exit(vc.main(TESTS))
