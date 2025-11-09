import torch
import torch.nn as nn
import torch.nn.functional as F

# === Mask Layer ==========================================================
class MaskLayer(nn.Module):
    """Apply building mask (0 = building, 1 = free space) to concentration map."""
    def __init__(self, mask):
        super().__init__()
        mask = torch.tensor(mask, dtype=torch.float32)
        self.register_buffer("mask", mask)

    def forward(self, x):
        return x * self.mask  # element-wise masking


# === EarlyStopping utility ===============================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        self.verbose = verbose

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict()
            if self.verbose:
                print(f"[EarlyStopping] Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True


# === MCxM_PIML network ===================================================
class MCxM_PIML(nn.Module):
    """
    Physics-Informed Mask Correction Module (PIML variant)
    ------------------------------------------------------
    Inputs:
        gauss_disp   : (B,1,H,W) raw dispersion map
        wind_features: (B,2) [speed, dir_deg]
        global_feat  : (B,G) optional physical/global features
    Output:
        corrected dispersion map (B,H,W)
    """

    def __init__(self, mask, m=500, dropout_p=0.2,
                 n_channel=1, n_global_features=0, wind_dim=2):
        super().__init__()
        self.m = m
        self.n_channel = n_channel
        self.wind_dim = wind_dim
        self.n_global_features = n_global_features

        # --- Mask layer ---
        self.mask_layer = MaskLayer(mask)

        # --- CNN encoder-decoder (spatial feature extractor) ---
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channel, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_p)
        )

        # dimensione ridotta (m/4 x m/4)
        reduced_size = (m // 4) * (m // 4) * 64
        fc_input = reduced_size + wind_dim + n_global_features

        self.mlp = nn.Sequential(
            nn.Linear(fc_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(512, m * m)  # ricostruzione mappa
        )

        # inizializzazione pesi
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    # ---------------------------------------------------------------------
    def forward(self, gauss_disp, wind_features, global_features=None):
        """
        gauss_disp   : (B,1,H,W)
        wind_features: (B,2)
        global_feat  : (B,G) or None
        """
        # 1. Applica la maschera urbana
        u = self.mask_layer(gauss_disp)

        # 2. Estrai feature spaziali
        z = self.encoder(u)
        z = torch.flatten(z, 1)

        # 3. Concatenazione fisica (vento + globali)
        if global_features is not None:
            z = torch.cat([z, wind_features, global_features], dim=1)
        else:
            z = torch.cat([z, wind_features], dim=1)

        # 4. Decodifica (ricostruzione)
        out = self.mlp(z)
        out = out.view(-1, self.m, self.m)

        # 5. Impone la maschera finale
        out = self.mask_layer(out)
        return out
