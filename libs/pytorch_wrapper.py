"""
detector_layers.py
==================

Rete neurale con detector di input adversarial incorporati nei layer,
addestrata sul dataset UNSW-NB15 (classificazione binaria: 0 = traffico
normale, 1 = attacco).

Componenti:

  - DetectorLayer (LD)   : wrappa un layer L; oltre all'output di L produce
                           una classificazione real/adversarial delle sue
                           attivazioni tramite un detector interno.
  - PassThrough          : versione "propagante" di un layer standard: applica
                           il layer e inoltra invariato lo stato (label + flag
                           real/adversarial), così i LD possono stare ovunque.
  - FlowState            : lo stato che viaggia con le attivazioni; accumula i
                           logit dei detector e la loro loss.
  - DetectorSequential   : container sequenziale che gestisce (attivazioni,
                           stato) e avvolge automaticamente i layer normali
                           in PassThrough.

I dati arrivano da preprocess.py (get_train_val_test_set): il validation set
per ora viene ignorato. Le colonne `id` e `attack_cat` vengono scartate:
`attack_cat` è la versione multi-classe della label (leakage totale: la
predirebbe da sola), `id` è solo un identificatore di riga. Le feature
categoriche (proto, service, state) sono ESCLUSE dalla perturbazione FGSM
tramite una maschera, perché una perturbazione continua su una codifica
label-encoded non corrisponde a nessun input reale.

Uso:
    python detector_layers.py [cartella_del_dataset]

(default: la cartella corrente; la cartella deve contenere i due CSV
 UNSW_NB15_training-set.csv e UNSW_NB15_testing-set.csv)
"""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import preprocess  # il tuo file: deve stare nella stessa cartella

# ===========================================================================
# Configurazione
# ===========================================================================
EPOCHS = 2
#EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-3                   # learning rate del backbone
LR_DET = 3e-3               # learning rate dei detector
EPS = 0.3                   # intensità FGSM, in unità standardizzate
LAMBDA_DET = 1.0            # peso della loss dei detector
TASK_LOSS_ON_ADV = False    # True = anche adversarial training del backbone
THRESHOLD = 0.5             # soglia sul punteggio adversarial in inferenza
SEED = 42
CHECKPOINT = "modello_detector.pt"

CATEGORICAL_COLS = ["proto", "service", "state"]  # escluse dall'attacco FGSM


# ===========================================================================
# Libreria: stato propagato + layer
# ===========================================================================
class FlowState:
    """Metadati che viaggiano accanto alle attivazioni durante il forward.

    labels     : label del task (propagata per completezza / usi futuri).
    is_adv     : (BATCH_SIZE,) 1.0 se l'istanza è adversarial, 0.0 se reale; 
                 se presente, ogni DetectorLayer accumula la propria BCE.
    detections : lista dei logit (BATCH_SIZE,1) dei detector attraversati.
    det_loss   : somma delle loss dei detector.
    """

    def __init__(self, labels=None, is_adv=None):
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
        """Punteggio aggregato (BATCH_SIZE,) che l'input sia adversarial.
        reduce="max" è più sensibile: basta che un detector si allarmi."""
        if not self.detections:
            return None

        probs = torch.sigmoid(torch.cat(self.detections, dim=-1))
        return probs.max(dim=-1).values if reduce == "max" else probs.mean(dim=-1)


def default_detector(in_dim: int, hidden: int = 64) -> nn.Module:
    """Detector binario per attivazioni tabellari: la LayerNorm stabilizza le
    scale delle attivazioni (che si spostano mentre il backbone si addestra),
    poi un piccolo MLP classifica real/adversarial."""
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
        nn.Linear(hidden, 1),
    )


class PassThrough(nn.Module):
    """Adatta un layer standard all'interfaccia (x, state) -> (y, state),
    inoltrando lo stato invariato.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x, state: FlowState):
        return self.base(x), state


class DetectorLayer(nn.Module):
    """LD: wrappa il layer `base` e classifica le sue attivazioni.

    detach=True (default): il detector osserva y.detach(), quindi la sua loss
    aggiorna solo i parametri del detector senza toccare il backbone.
    detach=False: il gradiente del detector risale anche nel backbone, che
    impara rappresentazioni che rendono gli attacchi più riconoscibili.
    """

    def __init__(self, base: nn.Module, detector: nn.Module, detach: bool = True):
        super().__init__()
        self.base = base
        self.detector = detector
        self.detach = detach

    def forward(self, x, state: FlowState):
        y = self.base(x)                          # output di L
        feats = y.detach() if self.detach else y
        state.add_detection(self.detector(feats)) # classificazione real/adv
        return y, state                           # y prosegue nella rete


class DetectorSequential(nn.Module):
    """Come nn.Sequential, ma propaga (attivazioni, FlowState); i moduli
    normali vengono avvolti automaticamente in PassThrough."""

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

    # id(p) è l'identificativo univoco dell'oggetto p 
    # (di fatto il suo indirizzo in memoria)
    # così escludiamo tutti i detector_parameters
    def backbone_parameters(self) -> Iterable[nn.Parameter]:
        det_ids = {id(p) for p in self.detector_parameters()}
        return (p for p in self.parameters() if id(p) not in det_ids)


# ===========================================================================
# Attacco (FGSM con maschera sulle feature attaccabili)
# ===========================================================================
def fgsm(model, x, y, eps, mask=None):
    """FGSM sulla task loss. `mask` (n_features,) con 1 sulle feature
    continue perturbabili e 0 su quelle categoriche da lasciare intatte."""
    x_adv = x.clone().detach().requires_grad_(True)
    logits, _ = model(x_adv)
    loss = F.cross_entropy(input=logits, target=y)
    (grad,) = torch.autograd.grad(outputs=loss, inputs=x_adv)
    delta = eps * grad.sign()
    if mask is not None:
        delta = delta * mask.to(x.device)
    return (x_adv + delta).detach()


# ===========================================================================
# Training e valutazione
# ===========================================================================
def train_epoch(model, loader, optimizer, eps, lambda_det=1.0,
                task_loss_on_adv=False, class_weights=None,
                attack_mask=None, device="cpu"):
    model.train()
    tot_task_loss, tot_det_loss = 0.0, 0.0
    tot_task_correct_clean_preds, tot_task_correct_adv_preds = 0, 0
    tot_det_correct_clean_preds, tot_det_correct_adv_preds = 0, 0
    tot_n = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # 1) gemello adversarial del batch reale
        x_adv = fgsm(model, x, y, eps, mask=attack_mask)

        # 2) batch misto + flag real(0)/adversarial(1) da propagare
        xb = torch.cat([x, x_adv])
        yb = torch.cat([y, y])
        adv_flag = torch.cat([torch.zeros(len(x), device=device),
                              torch.ones(len(x_adv), device=device)])

        # 3) forward: ogni LD accumula la sua loss in state.det_loss
        logits, state = model(xb, labels=yb, is_adv=adv_flag)

        # 4) task loss (solo campioni reali, salvo adversarial training)
        if task_loss_on_adv:
            task_loss = F.cross_entropy(logits, yb, weight=class_weights)
        else:
            real = adv_flag == 0
            task_loss = F.cross_entropy(logits[real], yb[real],
                                        weight=class_weights)

        loss = task_loss + lambda_det * state.det_loss
        optimizer.zero_grad()   # gradients accumlate, so always reset them before backward pass
        loss.backward()         # compute the gradients traversing autgrad graph
        optimizer.step()        # updates model parameters based on their lr

        n = len(x)
        tot_n += n
        tot_task_loss += task_loss.item() * n
        tot_det_loss += state.det_loss.item() * n

        tot_task_correct_clean_preds += (logits[:n].argmax(-1) == y).sum().item()
        tot_task_correct_adv_preds += (logits[n:].argmax(-1) == y).sum().item()

        with torch.no_grad():
            det_pred_adv = state.adv_score() > THRESHOLD # => adversarial sample
        tot_det_correct_clean_preds += (~det_pred_adv[:n]).sum().item()
        tot_det_correct_adv_preds += det_pred_adv[n:].sum().item()

    return {"task_loss": tot_task_loss / tot_n,
            "det_loss": tot_det_loss / tot_n,
            "task_clean_acc": tot_task_correct_clean_preds / tot_n,
            "task_adv_acc": tot_task_correct_adv_preds / tot_n,
            "det_clean_acc": tot_det_correct_clean_preds / tot_n,
            "det_adv_acc": tot_det_correct_adv_preds / tot_n}


@torch.no_grad()
def predict(model, x, threshold=THRESHOLD, reduce="mean"):
    """(label predette, punteggio adversarial, flag input-adversarial)."""
    model.eval()
    logits, state = model(x)
    score = state.adv_score(reduce=reduce)
    return logits.argmax(-1), score, score > threshold

def _binary_metrics(pred, true, positive=1):
    """Accuracy, precision e recall per una classificazione binaria.
    `positive` indica quale classe conta come "positiva" per precision/recall
    (per il task: 1 = attacco; per il detector: 1 = adversarial)."""
    pred = pred.long()
    true = true.long()
    tp = int(((pred == positive) & (true == positive)).sum())
    fp = int(((pred == positive) & (true != positive)).sum())
    fn = int(((pred != positive) & (true == positive)).sum())
    tn = int(((pred != positive) & (true != positive)).sum())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {"acc": acc, "precision": precision, "recall": recall}

def evaluate(model, X_te, y_te, eps, attack_mask=None, device="cpu",
             batch_size=4096, threshold=THRESHOLD):
    """Metriche di task e detector su test set pulito e adversarial (a batch).
 
    Ritorna un dizionario con, per il TASK (positivo = attacco) e per il
    DETECTOR (positivo = adversarial), accuracy/precision/recall:
 
      task_clean : metriche del task sugli input puliti
      task_adv   : metriche del task sugli input adversarial
      det_clean_acc : accuracy del detector sui puliti  (= 1 - FPR)
      det_adv_acc   : accuracy del detector sugli adversarial (= TPR = recall)
      detector   : precision/recall/accuracy del detector sull'insieme misto
                   puliti+adversarial (precision e recall richiedono entrambi
                   i tipi: i falsi positivi vengono dai puliti, i falsi
                   negativi dagli adversarial)
    """
    model.eval()
    res = {"lab_c": [], "lab_a": [], "sc_c": [], "sc_a": []}
    for i in range(0, len(X_te), batch_size):
        x = X_te[i:i + batch_size].to(device)
        y = y_te[i:i + batch_size].to(device)
        x_adv = fgsm(model, x, y, eps, mask=attack_mask)  # serve il gradiente
        lab_c, sc_c, _ = predict(model, x, threshold=threshold)
        lab_a, sc_a, _ = predict(model, x_adv, threshold=threshold)
        res["lab_c"].append(lab_c.cpu()); res["sc_c"].append(sc_c.cpu())
        res["lab_a"].append(lab_a.cpu()); res["sc_a"].append(sc_a.cpu())
 
    lab_c, lab_a = torch.cat(res["lab_c"]), torch.cat(res["lab_a"])
    sc_c, sc_a = torch.cat(res["sc_c"]), torch.cat(res["sc_a"])
 
    # --- TASK (positivo = attacco, label 1) --------------------------------
    task_clean = _binary_metrics(lab_c, y_te, positive=1)
    task_adv = _binary_metrics(lab_a, y_te, positive=1)
 
    # --- DETECTOR (positivo = adversarial) ---------------------------------
    # ground truth: puliti = 0 (non adversarial), adversarial = 1
    det_pred_clean = (sc_c > threshold).long()   # dovrebbe essere 0
    det_pred_adv = (sc_a > threshold).long()     # dovrebbe essere 1
    det_true_clean = torch.zeros_like(det_pred_clean)
    det_true_adv = torch.ones_like(det_pred_adv)
 
    det_clean_acc = (det_pred_clean == det_true_clean).float().mean().item()
    det_adv_acc = (det_pred_adv == det_true_adv).float().mean().item()
 
    det_pred = torch.cat([det_pred_clean, det_pred_adv])
    det_true = torch.cat([det_true_clean, det_true_adv])
    detector = _binary_metrics(det_pred, det_true, positive=1)
 
    return {"task_clean": task_clean,
            "task_adv": task_adv,
            "det_clean_acc": det_clean_acc,
            "det_adv_acc": det_adv_acc,
            "detector": detector,
            "score_clean": sc_c.mean().item(),
            "score_adv": sc_a.mean().item()}


# ===========================================================================
# Dati: UNSW-NB15 tramite il tuo preprocess.py
# ===========================================================================
def load_unsw(dataset_path: str):
    """Carica i set da preprocess.py  e costruisce la maschera d'attacco."""
    X_tr, y_tr, _, _, X_te, y_te = preprocess.get_train_val_test_set(
        dataset_path, download_dataset=False, verbose=False
    )

    feature_names = list(X_tr.columns)

    # 1.0 = feature continua attaccabile, 0.0 = categorica intoccabile
    attack_mask = torch.tensor(
        [0.0 if c in CATEGORICAL_COLS else 1.0 for c in feature_names]
    )

    to_x = lambda df: torch.from_numpy(df.to_numpy(dtype="float32"))
    to_y = lambda s: torch.from_numpy(s.to_numpy()).long()
    return (to_x(X_tr), to_y(y_tr), to_x(X_te), to_y(y_te),
            feature_names, attack_mask)


# ===========================================================================
# Modello
# ===========================================================================
def build_model(n_features: int, n_classes: int = 2,
                hidden: int = 128, detach: bool = True) -> DetectorSequential:
    """BackboneMLPMLP con due DetectorLayer a profondità diverse; gli altri
    layer vengono avvolti automaticamente nella versione propagante."""
    return DetectorSequential(
        nn.Linear(n_features, hidden), nn.ReLU(),
        DetectorLayer(nn.Linear(hidden, hidden),
                      detector=default_detector(hidden), detach=detach), nn.ReLU(),
        DetectorLayer(nn.Linear(hidden, 64),
                      detector=default_detector(64), detach=detach), nn.ReLU(),
        nn.Linear(64, n_classes),
    )


# ===========================================================================
# Main
# ===========================================================================
def main():
    dataset_path = "dataset/unsw-nb15/"
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- dati --------------------------------------------------------------
    X_tr, y_tr, X_te, y_te, feature_names, attack_mask = load_unsw(dataset_path)
    n_attack = int((y_tr == 1).sum())
    print(f"UNSW-NB15: \n"
          f"\t{len(X_tr)} train / {len(X_te)} test \n"
          f"\t{len(feature_names)} features \n"
          f"\tcategorical features excluded from attack: {[f for f in CATEGORICAL_COLS if f in feature_names]}\n"
          f"\ttraining set class 1 (attack instances) = {n_attack / len(y_tr):.1%}\n"
          f"device: {device}")

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    # pesi di classe (il dataset è sbilanciato verso gli attacchi)
    counts = torch.bincount(y_tr, minlength=2).float()
    class_weights = (len(y_tr) / (2 * counts)).to(device)

    # ---- modello e optimizer -----------------------------------------------
    model = build_model(len(feature_names)).to(device)
    optimizer = torch.optim.Adam(params=[
        {"params": list(model.backbone_parameters()), "lr": LR},
        {"params": list(model.detector_parameters()), "lr": LR_DET},
    ])

    # ---- training ----------------------------------------------------------
    print("\n== addestramento ==")
    for epoch in range(EPOCHS):
        stats = train_epoch(model, loader, optimizer,
                            eps=EPS, lambda_det=LAMBDA_DET,
                            task_loss_on_adv=TASK_LOSS_ON_ADV,
                            class_weights=class_weights,
                            attack_mask=attack_mask, device=device)
        print(f"epoch {epoch:3d}:\n"
              f"\ttask loss={stats['task_loss']:.4f}\n"
              f"\tdet loss={stats['det_loss']:.4f}\n"
              f"\ttask clean samples acc={stats['task_clean_acc']:.4f}\n"
              f"\ttask adv samples acc={stats['task_adv_acc']:.4f}\n"
              f"\tdet clean samples acc={stats['det_clean_acc']:.4f}\n"
              f"\tdet adv samples acc={stats['det_adv_acc']:.4f}")

    # ---- valutazione sul test set ufficiale --------------------------------
    print("\n== valutazione su test set ==")
    m = evaluate(model, X_te, y_te, eps=EPS, attack_mask=attack_mask,
                 device=device)
 
    tc, ta, det = m["task_clean"], m["task_adv"], m["detector"]
    print("TASK (positivo = attacco)")
    print(f"  clean       : acc={tc['acc']:.4f}  prec={tc['precision']:.4f}  rec={tc['recall']:.4f}")
    print(f"  adversarial : acc={ta['acc']:.4f}  prec={ta['precision']:.4f}  rec={ta['recall']:.4f}")
    print("DETECTOR (positivo = adversarial)")
    print(f"  accuracy sui puliti      : {m['det_clean_acc']:.4f}")
    print(f"  accuracy sugli adversarial: {m['det_adv_acc']:.4f}")
    print(f"  precision / recall (misto): {det['precision']:.4f} / {det['recall']:.4f}")
    print(f"  score medio puliti / adv  : {m['score_clean']:.4f} / {m['score_adv']:.4f}")
 
    # ---- checkpoint ---------------------------------------------------------
    torch.save({"state_dict": model.state_dict(),
                "feature_names": feature_names,
                "attack_mask": attack_mask}, CHECKPOINT)
    print(f"\ncheckpoint salvato in {CHECKPOINT}")


if __name__ == "__main__":
    main()