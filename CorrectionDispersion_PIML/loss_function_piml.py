import torch
import torch.nn.functional as F

def grad(field):
    """Compute finite differences gradient (central differences)."""
    dx = field[:, 1:, :] - field[:, :-1, :]
    dy = field[:, :, 1:] - field[:, :, :-1]
    # Pad to match original size
    dx = F.pad(dx, (0,0,1,0))
    dy = F.pad(dy, (0,1,0,0))
    return dx, dy

def physics_masked_loss_piml(output, sim, mask, wind_vector=None,
                             alpha=1.0, beta=10.0, gamma=0.1, delta=1.0, 
                             eps=0.1, zeta=0.5):
    """
    Physics-informed loss for dispersion correction (PIML version)
    
    output: [B, m, m] - model prediction
    sim:    [B, m, m] - simulated dispersion field
    mask:   [m, m]    - 1=free space, 0=building
    wind_vector: optional tuple (cosθ, sinθ)
    
    alpha: weight for MSE on free space
    beta:  weight for building penalty
    gamma: weight for smoothness (∇C small)
    delta: weight for non-negativity
    eps:   weight for physical alignment with wind
    zeta:  weight for GeoMask consistency
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)

    # Expand mask for batch
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)

    # --- 1. Coerenza con la simulazione (zone libere)
    mse_free = torch.mean(((output - sim) ** 2) * mask)

    # --- 2. Penalità concentrazione su edifici
    building_penalty = torch.mean((output ** 2) * (1 - mask))

    # --- 3. Penalità negatività
    nonneg = torch.mean(torch.relu(-output) ** 2)

    # --- 4. Smoothness spaziale (gradiente)
    dx, dy = grad(output)
    smoothness = torch.mean(dx ** 2 + dy ** 2)

    # --- 5. Vincolo fisico: direzionalità del gradiente vs vento
    if wind_vector is not None:
        wx, wy = wind_vector
        wx = torch.tensor(wx, dtype=torch.float32, device=output.device)
        wy = torch.tensor(wy, dtype=torch.float32, device=output.device)
        norm = torch.sqrt(dx**2 + dy**2 + 1e-8)
        alignment = (dx*wx + dy*wy) / norm   # coseno dell'angolo
        physics = torch.mean((1 - alignment) ** 2 * mask)
    else:
        physics = torch.tensor(0.0, device=output.device)

    # --- 6. Penalità su maschera geometrica (GeoMask)
    geomask = torch.mean(torch.relu(output) * (1 - mask))

    # --- Loss totale
    loss = (alpha * mse_free +
            beta  * building_penalty +
            gamma * smoothness +
            delta * nonneg +
            eps   * physics +
            zeta  * geomask)

    # --- Return anche breakdown per analisi e logging
    components = {
        "total": loss.item(),
        "mse_free": mse_free.item(),
        "building_penalty": building_penalty.item(),
        "smoothness": smoothness.item(),
        "nonneg": nonneg.item(),
        "physics": physics.item(),
        "geomask": geomask.item()
    }

    return loss, components
