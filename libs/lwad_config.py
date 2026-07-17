# ===========================================================================
# Configuration
# ===========================================================================
#EPOCHS = 2
EPOCHS = 10
BATCH_SIZE = 512
LR = 1e-3                   # backbone learning rate
LR_DET = 3e-3               # detector learning rate
LAMBDA_DET = 1.0            # detector loss weight
TASK_LOSS_ON_ADV = False    # True = backbone adversarial training
THRESHOLD = 0.5             # starting threshold
SEED = 42
CHECKPOINT = "modello_detector.pt"


# --- early stopping config ---------------------------------------
PATIENCE = 5                # epochs without improvements before stopping
MIN_DELTA = 1e-4            # minimum improvement for patience reset