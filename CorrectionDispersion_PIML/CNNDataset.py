from torch.utils.data import Dataset
import torch
import numpy as np

class CNNDataset2(Dataset):
    def __init__(self, concentration_maps, wind_dir, wind_speed, global_features=None, m=500):
        self.concentration_maps = concentration_maps
        self.wind_dir = wind_dir
        self.wind_speed = wind_speed
        self.global_features = global_features  # es. [[gps_x, gps_y], ...]
        self.m = m

    def __len__(self):
        return len(self.concentration_maps)

    def __getitem__(self, idx):
        conc_map = np.array(self.concentration_maps[idx], dtype=np.float32)
        wind_dir = np.array(self.wind_dir[idx], dtype=np.float32)
        wind_speed = np.array(self.wind_speed[idx], dtype=np.float32)

        global_feat = np.zeros(0, dtype=np.float32)

        conc_map_tensor = torch.tensor(conc_map, dtype=torch.float32).unsqueeze(0)  # [1, m, m]
        wind_dir_tensor = torch.tensor(wind_dir, dtype=torch.float32)
        wind_speed_tensor = torch.tensor(wind_speed, dtype=torch.float32)
        global_feat_tensor = torch.tensor(global_feat, dtype=torch.float32)

        return conc_map_tensor, wind_dir_tensor, wind_speed_tensor, global_feat_tensor
