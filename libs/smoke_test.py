"""Smoke test su dati sintetici: verifica entrambe le architetture."""
import torch
import torch.nn as nn

import libs.lwad_wrapper as lw
import libs.lwad_config as lc
import libs.lwad_attack as la
import libs.lwad_trainer as lt
import libs.lwad_evaluator as le

torch.manual_seed(0)
N, F_DIM = 256, 20
X = torch.randn(N, F_DIM)
y = (X[:, 0] + X[:, 1] > 0).long()
mask = torch.ones(F_DIM)
loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=True)

print("== 1) vincolo del contenitore: NearestAL + detector deve fallire ==")
try:
    lw.LWADSequential(
        lw.DetectorLayer(nn.Linear(F_DIM, 8), detector=lw.default_detector(8)),
        lw.NearestAL(nn.Linear(8, 2)),
    )
    raise AssertionError("la validazione NON e' scattata!")
except ValueError as e:
    print("   OK ->", str(e)[:60], "...")

try:
    lw.LWADSequential(
        lw.FurtherAL(nn.Linear(F_DIM, 8), detector=lw.default_detector(8)),
        lw.NearestAL(nn.Linear(8, 2)),
    )
    raise AssertionError("la validazione NON e' scattata con FurtherAL!")
except ValueError as e:
    print("   OK (FurtherAL e' un DetectorLayer) ->", str(e)[:40], "...")

print("\n== 2) MRO di FurtherAL ==")
print("  ", [c.__name__ for c in lw.FurtherAL.__mro__[:5]])
assert issubclass(lw.FurtherAL, lw.DetectorLayer)
assert issubclass(lw.FurtherAL, lw.ActivationLoss)

for name, cfg in [("Architettura 1 (DetectorArchConfig, FurtherAL)", lc.DetectorArchConfig(hidden=32, eps=0.1)),
                  ("Architettura 1 senza act loss (DetectorLayer)", lc.DetectorArchConfig(hidden=32, use_act_loss=False)),
                  ("Architettura 2 (AlignmentArchConfig, NearestAL)", lc.AlignmentArchConfig(hidden=32))]:
    print(f"\n== 3) {name} ==")
    built = lc.create_architecture(cfg, F_DIM)
    model, opt = built.model, built.optimizer
    print("   has_detectors =", model.has_detectors,
          "| task_loss_on_adv =", cfg.task_loss_on_adv,
          "| pgd_alpha =", cfg.pgd_alpha)

    # forward con is_adv: verifica quali loss compaiono
    xb = torch.cat([X[:32], X[:32] + 0.05])
    flag = torch.cat([torch.zeros(32), torch.ones(32)])
    _, state = model(xb, is_adv=flag)
    print("   det_loss:", "presente" if state.det_loss is not None else "assente",
          "| act_loss:", "presente" if state.act_loss is not None else "assente")

    # il gradiente della act loss deve raggiungere la backbone
    if state.act_loss is not None:
        g = torch.autograd.grad(state.act_loss,
                                next(model.backbone_parameters()),
                                retain_graph=True, allow_unused=True)[0]
        assert g is not None and g.abs().sum() > 0, "act_loss non raggiunge la backbone!"
        print("   gradiente act_loss -> backbone: OK")

    # un'epoca di training + valutazione
    stats = lt.train_epoch(model, loader, opt, eps=cfg.eps,
                           lambda_det=getattr(cfg, "lambda_det", 0.0),
                           lambda_act=cfg.lambda_act,
                           task_loss_on_adv=cfg.task_loss_on_adv,
                           attack_mask=mask, attack=cfg.train_attack,
                           threshold=getattr(cfg, "threshold", 0.5),
                           attack_kwargs=cfg.attack_kwargs())
    print("   train stats:", {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in stats.items()})
    m = le.evaluate(model, X, y, eps=cfg.eps, attack_mask=mask,
                    attack=cfg.eval_attack, threshold=getattr(cfg, "threshold", 0.5),
                    attack_kwargs=cfg.attack_kwargs(), batch_size=128)
    print("   eval: task_clean acc =", round(m["task_clean"]["acc"], 3),
          "| task_adv acc =", round(m["task_adv"]["acc"], 3),
          "| det_clean_acc =", m["det_clean_acc"] if m["det_clean_acc"] is None else round(m["det_clean_acc"], 3))

    if model.has_detectors:
        thr, bal = lt.select_threshold(model, X, y, eps=cfg.eps, attack_mask=mask,
                                       attack=cfg.train_attack,
                                       attack_kwargs=cfg.attack_kwargs())
        print("   select_threshold:", round(thr, 3), "bal =", round(bal, 3))
        x_adv = la.generate_attack(model, X[:32], y[:32], cfg.eps, "pgd_adaptive",
                                   mask=mask, **cfg.attack_kwargs())
        assert (x_adv - X[:32]).abs().max() <= cfg.eps + 1e-5
        # il detach dei DetectorLayer deve essere ripristinato dopo l'attacco
        assert all(m.detach for m in model.modules() if isinstance(m, lw.DetectorLayer))
        print("   pgd_adaptive: OK (proiezione L-inf e ripristino detach)")
    else:
        try:
            la.generate_attack(model, X[:32], y[:32], cfg.eps, "pgd_adaptive", mask=mask)
            raise AssertionError("pgd_adaptive doveva fallire senza detector!")
        except ValueError:
            print("   pgd_adaptive senza detector: rifiutato correttamente")
        try:
            lt.select_threshold(model, X, y, eps=cfg.eps, attack_mask=mask)
            raise AssertionError("select_threshold doveva fallire senza detector!")
        except ValueError:
            print("   select_threshold senza detector: rifiutato correttamente")

print("\n== 4) split_pairs con batch sbilanciato deve fallire ==")
built = lc.create_architecture(lc.AlignmentArchConfig(hidden=16), F_DIM)
try:
    bad_flag = torch.cat([torch.zeros(10), torch.ones(22)])
    built.model(X[:32], is_adv=bad_flag)
    raise AssertionError("split_pairs NON ha rilevato lo sbilanciamento!")
except ValueError as e:
    print("   OK ->", str(e)[:60], "...")

print("\n== 5) margini per layer ==")
cfg = lc.DetectorArchConfig(hidden=32, act_margin=(0.5, 0.05))
model = cfg.build_model(F_DIM)
ms = [l.margin for l in model.layers if isinstance(l, lw.FurtherAL)]
assert ms == [0.5, 0.05], ms
print("   margini applicati in ordine:", ms)

try:
    lc.DetectorArchConfig(hidden=32, act_margin=(0.5, 0.05, 0.1)).build_model(F_DIM)
    raise AssertionError("lunghezza margini errata NON rilevata!")
except ValueError as e:
    print("   OK ->", str(e)[:60], "...")

from libs.lwad_margin import suggest_margins, select_margins
m = suggest_margins(lc.DetectorArchConfig(hidden=32), X, y,
                    attack_mask=mask, verbose=False)
assert m is not None and len(m) == 2 and all(v > 0 for v in m)
print("   suggest_margins:", tuple(round(v, 5) for v in m))
assert suggest_margins(lc.AlignmentArchConfig(hidden=32), X, y,
                       attack_mask=mask, verbose=False) is None
print("   suggest_margins su architettura 2: None (corretto)")

res = select_margins(lc.DetectorArchConfig(hidden=32), X, y, X, y,
                     attack_mask=mask, factors=(2.0, 20.0),
                     search_epochs=1, verbose=False)
assert len(res.margins) == 2 and res.factor in (2.0, 20.0)
assert len(res.candidates) == 2
print("   select_margins: factor =", res.factor,
      " margini =", tuple(round(v, 5) for v in res.margins))
cfg_best = lc.DetectorArchConfig(hidden=32, act_margin=res.margins)
mm = [l.margin for l in cfg_best.build_model(F_DIM).layers
      if isinstance(l, lw.FurtherAL)]
assert tuple(mm) == res.margins
print("   margini selezionati applicabili alla config: OK")

try:
    select_margins(lc.AlignmentArchConfig(hidden=32), X, y, X, y,
                   attack_mask=mask, verbose=False)
    raise AssertionError("select_margins doveva fallire senza FurtherAL!")
except ValueError:
    print("   select_margins su architettura 2: rifiutato correttamente")

print("\nTUTTI I TEST SUPERATI")
