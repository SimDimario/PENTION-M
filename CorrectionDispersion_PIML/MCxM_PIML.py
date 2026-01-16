import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout_p=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        x = self.dropout(x)
        return x


class MCxM_PIML(nn.Module):
    """
    UNet 2D:
    - input: gaussian_map (1 channel) + building_mask (1 channel) -> 2 channel
    - wind: modulation in the bottleneck via MLP
    - output: m x m
    """

    def __init__(
        self,
        mask_unused,
        m=500,
        dropout_p=0.1,
        n_channel=1,
        n_global_features=0,
        wind_dim=2,
    ):
        super().__init__()
        self.m = m
        self.wind_dim = wind_dim
        self.n_global_features = n_global_features

        if isinstance(mask_unused, np.ndarray):
            bmap = torch.from_numpy(mask_unused.astype("float32"))
        elif isinstance(mask_unused, torch.Tensor):
            bmap = mask_unused.detach().float().cpu()
        else:
            raise TypeError("mask_unused deve essere np.ndarray o torch.Tensor")

        if bmap.dim() != 2:
            raise ValueError("mask_unused deve avere shape [H, W]")

        bmap = bmap.unsqueeze(0).unsqueeze(0)
        self.register_buffer("building", bmap)
        in_ch = n_channel + 1
        self.enc1 = ConvBlock(in_ch, 32, dropout_p=dropout_p)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(32, 64, dropout_p=dropout_p)
        self.bottleneck = ConvBlock(64, 128, dropout_p=dropout_p)
        self.up2 = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(32 + 32, 32, dropout_p=dropout_p)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.wind_mlp = nn.Sequential(
            nn.Linear(wind_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, gauss_disp, wind_features, global_features=None):
        """
        gauss_disp: [B,1,H,W] (normalized)
        wind_features: [B,2] = [wind_speed, wind_dir_deg]
        """
        B, _, H, W = gauss_disp.shape
        bmap = self.building
        if bmap.shape[-2:] != (H, W):
            bmap = F.interpolate(bmap, size=(H, W), mode="nearest")
        bmap = bmap.expand(B, -1, -1, -1)
        x = torch.cat([gauss_disp, bmap], dim=1)
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        b = self.bottleneck(e2)
        wind_embed = self.wind_mlp(wind_features)
        wind_embed = wind_embed.view(B, 128, 1, 1)
        b = b + wind_embed
        u2 = self.up2(b)
        if u2.shape[-2:] != e1.shape[-2:]:
            u2 = F.interpolate(
                u2, size=e1.shape[-2:], mode="bilinear", align_corners=False
            )
        d1 = torch.cat([u2, e1], dim=1)
        d1 = self.dec1(d1)
        out = self.out_conv(d1)
        out = out.squeeze(1)
        return out
