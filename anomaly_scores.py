import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import pickle
from utils import SlidingWindowDataset  
import pandas as pd
class AnomalyScoreGenerator:
    def __init__(self, model, window_size, batch_size, target_dims, use_cuda, gamma=1, scale_scores=False):
        self.model = model
        self.window_size = window_size
        self.batch_size = batch_size
        self.target_dims = target_dims
        self.use_cuda = use_cuda
        self.gamma = gamma
        self.scale_scores = scale_scores
        if isinstance(target_dims, dict):
            self.target_dims = list(target_dims.values())  # or keys, depending on your needs
        else:
            self.target_dims = target_dims
    def get_scores(self, values):
        """Calculate anomaly scores for each feature"""
        print("Calculating anomaly scores for each feature...")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        recons = []
        with torch.no_grad():
           for x, y in tqdm(loader):
               x = x.to(device)
               y = y.to(device)

               y_hat, _ = self.model(x)

               # Shifting input to include the observed value (y) when doing the reconstruction
               recon_x = torch.cat((x[:, 1:, :], y), dim=1)
               _, window_recon = self.model(recon_x)

               preds.append(y_hat.detach().cpu().numpy())
               recons.append(window_recon[:, -1, :].detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        recons = np.concatenate(recons, axis=0)
        print(f"preds_shape:{preds.shape}")
        if isinstance(values, torch.Tensor):
          actual = values.detach().cpu().numpy()[self.window_size:]
        else:
            actual = values[self.window_size:]
        anomaly_scores = np.zeros_like(actual)
        feature_scores = {}
        for i in range(preds.shape[1]):
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2) + self.gamma * np.sqrt(
               (recons[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
               q75, q25 = np.percentile(a_score, [75, 25])
               iqr = q75 - q25
               median = np.median(a_score)
               a_score = (a_score - median) / (1 + iqr)
            anomaly_scores[:,i]=a_score
            feature_scores[f"A_Score_{i}"] = a_score
        anomaly_scores_global=np.mean(anomaly_scores, 1)
        df = pd.DataFrame(anomaly_scores_global)
        print(f"first 10 values of anomaly scores: anomaly_scores[:4,0:38]")
        print(f"shape of anaomaly_scores_global:{df.shape},df.head(10)")
        print(f"Shape_of_anomaly_scores:{anomaly_scores.shape}")
        
        return df
    
    def save_scores(self, scores, save_path):
     """Save the entire anomaly score DataFrame (shape: [28379, 38])"""
     if not os.path.exists(save_path):
        os.makedirs(save_path)

     file_path = os.path.join(save_path, "anomaly_scores.pkl")
     with open(file_path, 'wb') as f:
        pickle.dump(scores, f)  # Save the entire DataFrame
    
     print(f"Anomaly scores saved to {file_path}")

        
       

    def generate_and_save_scores(self, train_data, test_data, save_path):
        """Generate and save anomaly scores for train and test data"""
        
        train_scores = self.get_scores(train_data)
        test_scores = self.get_scores(test_data)

        train_save_path = os.path.join(save_path, 'train')
        test_save_path = os.path.join(save_path, 'test')

        self.save_scores(train_scores, train_save_path)
        self.save_scores(test_scores, test_save_path)
      
    save_path = 'anomaly_scores'  # This will create a folder named 'anomaly_scores'
    
    
