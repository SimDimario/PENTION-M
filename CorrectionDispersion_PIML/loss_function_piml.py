import torch
import torch.nn.functional as F

def grad(field):
    dx = field[:, 1:, :] - field[:, :-1, :]
    dy = field[:, :, 1:] - field[:, :, :-1]
    dx = F.pad(dx, (0, 0, 1, 0))
    dy = F.pad(dy, (0, 1, 0, 0))
    return dx, dy


def physics_masked_loss_piml(
    output,
    sim,
    mask,
    wind_vector=None,
    alpha=1.0,   # mse free
    beta=0.2,    # building penalty
    gamma=1e-3,  # smoothness
    delta=0.1,   # non-negativity
    eps=0.05,    # wind alignment
    zeta=0.3     # geomask
):
    """
    output: [B,m,m]
    sim:    [B,m,m]
    mask:   [m,m] (1=free, 0=building)
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device).float()

    if mask.dim() == 2:
        mask = mask.unsqueeze(0)  # [1,m,m]

    # broadcast su batch
    while mask.dim() < output.dim():
        mask = mask.unsqueeze(0)  # [1,1,m,m] se serve

    # 1) MSE solo su free space
    mse_free = torch.mean(((output - sim) ** 2) * mask)

    # 2) penalità su edifici
    building_penalty = torch.mean((output ** 2) * (1.0 - mask))

    # 3) non negatività
    nonneg = torch.mean(torch.relu(-output) ** 2)

    # 4) smoothness
    dx, dy = grad(output)
    smoothness = torch.mean(dx ** 2 + dy ** 2)

    # 5) allineamento vento / gradiente
    if wind_vector is not None:
        wx, wy = wind_vector
        wx = torch.tensor(wx, dtype=torch.float32, device=output.device)
        wy = torch.tensor(wy, dtype=torch.float32, device=output.device)
        norm = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)
        alignment = (dx * wx + dy * wy) / norm  # cos(theta)
        physics = torch.mean((1.0 - alignment) ** 2 * mask)
    else:
        physics = torch.tensor(0.0, device=output.device)

    # 6) geomask (output positivo su edifici)
    geomask = torch.mean(torch.relu(output) * (1.0 - mask))

    loss = (
        alpha * mse_free +
        beta * building_penalty +
        gamma * smoothness +
        delta * nonneg +
        eps * physics +
        zeta * geomask
    )

    components = {
        "total": loss.item(),
        "mse_free": mse_free.item(),
        "building_penalty": building_penalty.item(),
        "smoothness": smoothness.item(),
        "nonneg": nonneg.item(),
        "physics": physics.item(),
        "geomask": geomask.item(),
    }

    return loss, components
