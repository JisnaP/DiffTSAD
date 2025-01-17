import pickle
import torch
from scipy.interpolate import interp1d
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import SlidingWindowDataset
from torch.utils.data import DataLoader

class AnomalyScoreLoader:
    def __init__(self, window_size, N):
        self.window_size = window_size
        self.N = N
        self.scaler = MinMaxScaler()

    def load_anomaly_scores(self, file_path):
        """Load the entire anomaly score DataFrame and convert it to a PyTorch tensor"""
        with open(file_path, 'rb') as f:
            anomaly_scores_df = pickle.load(f)

        # Convert DataFrame to NumPy array
        anomaly_scores_np = anomaly_scores_df.values
        
        # Interpolate to match the required length
        f = interp1d(np.arange(self.window_size, self.N), anomaly_scores_np, axis=0, fill_value="extrapolate")
        interpolated_anomaly_scores = f(np.arange(self.N))
        
        # Convert to PyTorch tensor
        anomaly_scores_tensor = torch.tensor(interpolated_anomaly_scores, dtype=torch.float32)
        print(f"anomaly score tensor shape:{anomaly_scores_tensor.shape}")
        return anomaly_scores_tensor

    def normalize_anomaly_scores(self, anomaly_scores_tensor):
        """Normalize the anomaly scores using MinMaxScaler"""
        # Convert to NumPy array for MinMaxScaler compatibility
        anomaly_scores_np = anomaly_scores_tensor.numpy()

        # Fit and transform using MinMaxScaler
        normalized_scores_np = self.scaler.fit_transform(anomaly_scores_np)

        # Convert back to PyTorch tensor
        normalized_scores_tensor = torch.tensor(normalized_scores_np, dtype=torch.float32)

        return normalized_scores_tensor
        
    def create_anomaly_score_dataset(self,normalized_anomaly_scores_train,normalized_anomaly_scores_test=None):
      #create sliding window dataset for loading anomaly scores to dataloader
        anomaly_score_dataset_train=SlidingWindowDataset(normalized_anomaly_scores_train,window_size=100,horizon=1,stride=1) 
        anomaly_score_dataset_test=None
        if normalized_anomaly_scores_test is not None:
          anomaly_score_dataset_test=SlidingWindowDataset(normalized_anomaly_scores_test,window_size=100,horizon=1,stride=1)
        print(f"anomaly score dataset size: {len(anomaly_score_dataset_train)}")
        return anomaly_score_dataset_train,anomaly_score_dataset_test

    def create_anomalyscores_loaders(self, anomaly_score_dataset_train, batch_size, shuffle=False, anomaly_score_dataset_test=None):
        """Create DataLoader for train and test anomaly score datasets"""
        anomaly_score_train_loader = DataLoader(dataset=anomaly_score_dataset_train, batch_size=batch_size, shuffle=shuffle,drop_last=True)
        anomaly_score_test_loader = None

        if anomaly_score_dataset_test is not None:
            anomaly_score_test_loader = DataLoader(dataset=anomaly_score_dataset_test, batch_size=batch_size, shuffle=shuffle,drop_last=True)
        for i,(x,y) in enumerate(anomaly_score_train_loader):
          print(f"shape of anoamly score train : {x.shape},{y.shape}")
          if i<1:
            break
        return anomaly_score_train_loader, anomaly_score_test_loader




    