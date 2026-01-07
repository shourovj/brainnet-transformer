import torch.nn as nn
from torch.utils import data
import scipy.io as sio
import torch

class MRIDataset(data.Dataset):
    def __init__(self, data_dir, df):
        self.df = df
        self.data_dir = data_dir

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        feature_file = row['feature']
        cluster_file = row['cluster']
        label = row['label']

        # Load .mat files
        f_data = sio.loadmat(self.data_dir + feature_file)
        f_mat = f_data['feature_mat']  # Shape: (400, 1632) - ROI features
        c_data = sio.loadmat(self.data_dir + cluster_file)
        c_mat = c_data['cluster_index_mat']  # Shape: (45, 54, 45) - Cluster indices

        # Convert to tensors
        f_mat = torch.FloatTensor(f_mat)  # Feature matrix
        c_mat = torch.LongTensor(c_mat)  # Cluster indices (integers)
        label = torch.LongTensor([label - 1])  # Convert to 0-indexed (1,2 -> 0,1)

        return f_mat, c_mat, label

