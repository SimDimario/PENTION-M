import torch
import torch.nn as nn
from MCxM import EarlyStopping
from tqdm import tqdm
from loss_function import physics_masked_loss

def train(epochs, model, train_loader, val_loader, binary_map, device):

    train_losses  = []
    val_losses = []
    val_maes     = []

    #loss_fn = torch.nn.MSELoss()
    #optimizer=torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    #decay_rate = 0.95
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    early_stopper = EarlyStopping(patience=5)

    for epoch in tqdm(range(epochs), desc="Training epochs"):
        model.train()
        running_loss, running_mae = 0.0, 0.0

        for batch_idx, batch in enumerate(train_loader):
            batch = batch.to(device)

            mc = batch.local_maps
            wind_speed = batch.wind_speed.view(-1, 1)
            wind_dir = batch.wind_dir.view(-1, 1)     
            wind_features = torch.cat([wind_speed, wind_dir], dim=1)  
            global_features = batch.global_features if hasattr(batch, 'global_features') else None

            optimizer.zero_grad()
            output = model(mc, wind_features, global_features)

            loss = physics_masked_loss(output, mc[:, 0], binary_map, alpha=1.0, beta=1000.0, gamma=0.1, delta=5.0)            
            mae  = torch.mean(torch.abs((output - mc[:, 0]) * torch.tensor(binary_map, dtype=torch.float32, device=device)))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_mae  += mae.item()

        ##scheduler.step()

        avg_train_loss = running_loss / len(train_loader)
        avg_train_mae  = running_mae / len(train_loader)
        train_losses.append(avg_train_loss)

        print(f"[INFO] Epoch {epoch+1}: Train Loss={avg_train_loss:.6f}, Train MAE={avg_train_mae:.6f}")

        # --- Validation
        val_loss, val_mae = validate(model, val_loader, binary_map, device)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        scheduler.step(val_loss)

        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f"[INFO] Early stopping at epoch {epoch+1}")
            if early_stopper.best_model_state is not None:
                model.load_state_dict(early_stopper.best_model_state)
            break

    return model, output, train_losses, val_losses, val_maes # type: ignore

def validate(model, val_loader, binary_mask, device):
    model.eval()
    total_loss, total_mae, n_batches = 0, 0, 0

    mask= torch.tensor(binary_mask, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            mc = batch.local_maps          
            wind_speed = batch.wind_speed.view(-1, 1)
            wind_dir = batch.wind_dir.view(-1, 1)     
            wind_features = torch.cat([wind_speed, wind_dir], dim=1) 
            global_features = batch.global_features if hasattr(batch, 'global_features') else None

            output = model(mc, wind_features, global_features)
 
            loss = physics_masked_loss(output, mc[:, 0], mask, alpha=1.0, beta=1000.0, gamma=0.1, delta=5.0)
            mae  = torch.mean(torch.abs((output - mc[:, 0]) * mask))

            total_loss += loss.item()
            total_mae  += mae.item()
            n_batches  += 1

    avg_loss = total_loss / n_batches
    avg_mae  = total_mae / n_batches

    print(f"[Validation] Loss: {avg_loss:.6f}, MAE: {avg_mae:.6f}")
    return avg_loss, avg_mae
    