import torch
from torch_geometric.data import Dataset, Data

class CNNDataset(Dataset):
    def __init__(self, concentration_maps, building_maps, wind_dir, wind_speed, global_features, m=500, path_saved="./CorrectionDispersion/CNNdataset"):
        super().__init__(root=path_saved)
        self.concentration_maps = concentration_maps  # lista o array di (m, m)
        self.building_maps = building_maps       # [m, m]
        self.wind_dir = wind_dir                      # lista o array di scalari
        self.wind_speed = wind_speed
        self.global_features = global_features   # [N, n_global_features]
        self.m = m

    def __len__(self):
        return len(self.concentration_maps)

    def __getitem__(self, idx):
        local_maps = torch.stack([
            torch.tensor(self.concentration_maps[idx], dtype=torch.float32),
            torch.tensor(self.building_maps, dtype=torch.float32)
        ])  # [2, m, m]

        local_maps = local_maps.unsqueeze(0) #[1, 2, m, m]

        wind_speed = torch.tensor([self.wind_speed[idx]], dtype=torch.float32)  # shape (1,)
        wind_dir = torch.tensor([self.wind_dir[idx]], dtype=torch.float32)    # shape (2,)
        global_features = torch.tensor(self.global_features[idx], dtype=torch.float32).unsqueeze(0)

        data = Data(
            local_maps=local_maps,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            global_features=global_features
        )
        return data
    
class CNNDataset2(Dataset):
    def __init__(self, concentration_maps, wind_dir, wind_speed, global_features=None, m=500, path_saved="./CorrectionDispersion/CNNdataset"):
        super().__init__(root=path_saved)
        self.concentration_maps = concentration_maps  # lista o array di (m, m)
        self.wind_dir = wind_dir                      # lista o array di scalari
        self.wind_speed = wind_speed
        self.global_features = global_features   # [N, n_global_features]
        self.m = m

    def __len__(self):
        return len(self.concentration_maps)

    def __getitem__(self, idx):
        local_maps = torch.stack([
            torch.tensor(self.concentration_maps[idx], dtype=torch.float32),
        ])  # [1, m, m]

        local_maps = local_maps.unsqueeze(0) #[1, 1, m, m]

        wind_speed = torch.tensor([self.wind_speed[idx]], dtype=torch.float32)  # shape (1,)
        wind_dir = torch.tensor([self.wind_dir[idx]], dtype=torch.float32)    # shape (1,)
        
        if self.global_features is not None:
            global_features = torch.tensor(self.global_features[idx], dtype=torch.float32).unsqueeze(0)
            data = Data(
                local_maps=local_maps,
                wind_speed=wind_speed,
                wind_dir=wind_dir,
                global_features=global_features
            )
        else:
            data = Data(
                local_maps=local_maps,
                wind_speed=wind_speed,
                wind_dir=wind_dir,
            )
        return data
