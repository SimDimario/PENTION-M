import torch
import torch.nn.functional as F

def total_variation(x):
    dh = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean()
    dw = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean()
    return dh + dw

def physics_masked_loss(output, sim, mask, alpha=1.0, beta=10.0, gamma=0.1, delta=1.0, sigma=5.0):
    """
    output: [B, m, m] -> predizioni rete correttiva
    sim:    [B, m, m] -> simulazione grezza
    mask:   [m, m]    -> 1 = spazio libero, 0 = edificio

    alpha: peso coerenza con simulazione nelle zone libere
    beta:  peso penalità concentrazione sugli edifici
    gamma: peso regolarizzazione smoothness
    delta: peso positività
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)

    #mask = mask.unsqueeze(0)  # broadcast su batch [1, H, W]

    # Coerenza con la simulazione nelle zone libere
    mse_free = torch.mean(((output - sim) ** 2) * mask)

    # Penalità dentro gli edifici
    building_penalty = torch.mean((output ** 2) * (1 - mask))

    # Smoothness spaziale
    dx = (output[:, 1:, :] - output[:, :-1, :]) #* mask[:, 1:, :]
    dy = (output[:, :, 1:] - output[:, :, :-1]) #* mask[:, :, 1:]
    smoothness = torch.mean(dx ** 2) + torch.mean(dy ** 2)

    # Penalità valori negativi
    negativity = torch.mean(torch.relu(-output) ** 2)

    # TV regularization on free space
    #L_tv = total_variation(output * mask)

    # Loss totale
    loss = alpha * mse_free + beta * building_penalty + gamma * smoothness + delta * negativity #+ sigma * L_tv
    return loss

def physics_informed_loss(output, mc, mask, alpha=1.0, beta=10.0, gamma=0.1):
    """
    output: [B, m, m] -> predizioni rete correttiva
    mc: [B, m, m] -> simulazione grezza in ingresso
    mask: [m, m] -> 1 = zona libera, 0 = edificio
    alpha, beta, gamma, delta: pesi delle varie componenti della loss
    """

    if not isinstance(mask, torch.Tensor):
        mask = torch.tensor(mask, dtype=torch.float32, device=output.device)
    else:
        mask = mask.to(output.device)
    
    # 1. Coerenza con la simulazione grezza fuori dagli edifici 
    mse_free = torch.mean(((output - mc[:, 0]) ** 2) * mask)

    # 2. Penalità dentro gli edifici 
    building_penalty = torch.mean((output ** 2) * (1 - mask))

    #  Smoothness spaziale 
    dx = output[:, 1:, :] - output[:, :-1, :]
    dy = output[:, :, 1:] - output[:, :, :-1]
    smoothness = torch.mean(dx ** 2) + torch.mean(dy ** 2)

    loss = alpha * mse_free + beta * building_penalty + gamma * smoothness

    return loss