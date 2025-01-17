import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from tqdm.auto import tqdm

class Trainer:
    """Trainer class for LatentDiffusionModel model.

    :param model: LatentDiffusion model
    :param optimizer: Optimizer used to minimize the loss function
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param target_dims: dimension of input features to forecast and reconstruct
    :param n_epochs: Number of iterations/epochs
    :param batch_size: Number of windows in a single batch
    :param init_lr: Initial learning rate of the module
    :param loss: diffusion loss.
    :param boolean use_cuda: To be run on GPU or not
    :param dload: Download directory where models are to be dumped
    :param log_dir: Directory where SummaryWriter logs are written to
    :param print_every: At what epoch interval to print losses
    :param log_tensorboard: Whether to log loss++ to tensorboard
    :param args_summary: Summary of args that will also be written to tensorboard if log_tensorboard
    """

    def __init__(
        self,
        model,
        optimizer,
        window_size,
        n_features,
        target_dims=None,
        n_epochs=30,
        batch_size=32,
        init_lr=0.001,
        use_cuda=True,
        dload="",
        log_dir="output_ld/",
        print_every=1,
        log_tensorboard=True,
        args_summary="",
    ):

        self.model = model
        self.optimizer = optimizer
        self.window_size = window_size
        self.n_features = n_features
        self.target_dims = target_dims
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_lr = init_lr
        self.device = "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        self.dload = dload
        self.log_dir = log_dir
        self.print_every = print_every
        self.log_tensorboard = log_tensorboard

        self.losses = {
            "train_total": []
        }
        self.epoch_times = []

        if self.device == "cuda":
            self.model.cuda()

        if self.log_tensorboard:
            self.writer = SummaryWriter(f"{log_dir}")
            self.writer.add_text("args_summary", args_summary)

    def fit(self, train_loader,anomaly_score_train_loader):
        """Train model for self.n_epochs.
        :train losses stored in self.losses

        :param train_loader: train loader of input data
        :param anomaly_score_train_loader: anomaly score loader of train data
        """

        init_train_loss = self.evaluate(train_loader,anomaly_score_train_loader)
        print(f"Init total train loss: {init_train_loss:5f}")

        
        print(f"Training model for {self.n_epochs} epochs..")
        train_start = time.time()
        for epoch in tqdm(range(self.n_epochs)):
            epoch_start = time.time()
            # Training Mode
            self.model.train()
            
            recon_losses = []

            for (X,y) in tqdm(train_loader):
                
                X = X.to(self.device)
                y = y.to(self.device)
                print(f"Shape of X -: {X.shape}")
                #X=X.permute(0,2,1)
                #print(f"Shape of X input after permute: {X.shape}")
                #t = torch.randint(0, self.time_steps, (X.shape[0],), device=X.device).long()
                (anomaly_scores,_) = next(iter(anomaly_score_train_loader))
                #print(f"Shape of anomaly_score:{anomaly_scores.shape}")
                anomaly_scores=anomaly_scores.permute(0,2,1)
                anomaly_scores=anomaly_scores.to(self.device)
                #print(f"Shape of anomaly_score after permute:{anomaly_scores.shape}")
                # Optimizer zero grad
                self.optimizer.zero_grad()
                if self.target_dims is not None:
                    X = X[:, :, self.target_dims]
                    y = y[:, :, self.target_dims].squeeze(-1)
                # Forward pass
                loss, recons = self.model(X,anomaly_scores)

                # Loss backward
                loss.backward()

                #Optimizer step
                self.optimizer.step()

                recon_losses.append(loss.item()) 
            recon_losses = np.array(recon_losses)

            
            recon_epoch_loss = np.sqrt((recon_losses ** 2).mean())

            self.losses["train_total"].append(recon_epoch_loss)

            if self.log_tensorboard:
                self.write_loss(epoch)

            epoch_time = time.time() - epoch_start
            self.epoch_times.append(epoch_time)

            if epoch % self.print_every == 0:
                s = (
                    f"[Epoch {epoch + 1}] "
                    f"[Train_Loss: {recon_epoch_loss:.5f}]"
                )

            self.save(f"model.pt")

        train_time = int(time.time() - train_start)
        if self.log_tensorboard:
            self.writer.add_text("total_train_time", str(train_time))
        print(f"-- Training done in {train_time}s.")
    
    def evaluate(self, test_loader,anomaly_score_test_loader):
        """Evaluate model

        :param data_loader: data loader of input data 
        :param anomaly_score_loader:anomaly_score_loader of test data
        :return test loss
        """

        self.model.eval()
        
       
        recon_losses = []
        
        with torch.no_grad():
            for (X,y) in tqdm(test_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                #X=X.permute(0,2,1)
                #t = torch.randint(0, self.time_steps, (X.shape[0],), device=X.device).long()
               
                (anomaly_scores,_) = next(iter(anomaly_score_test_loader))
                print(anomaly_scores.shape)
                anomaly_scores=anomaly_scores.permute(0,2,1)
                anomaly_scores=anomaly_scores.to(self.device)
                #  Forward pass and calculate loss
                loss, recons = self.model(X,anomaly_scores)

                if self.target_dims is not None:
                    X = X[:, :, self.target_dims-1]
                    y = y[:, :, self.target_dims-1].squeeze(-1)

                #if preds.ndim == 3:
                    #preds = preds.squeeze(1)
                #if y.ndim == 3:
                    #y = y.squeeze(1)

                recon_losses.append(loss.item())       
        recon_losses = np.array(recon_losses)       
        recon_loss = np.sqrt((recon_losses ** 2).mean())
        return recon_loss

    def save(self, file_name):
        """
        Pickles the model parameters to be retrieved later
        :param file_name: the filename to be saved as,`dload` serves as the download directory
        """
        PATH = self.dload + "/" + file_name
        if os.path.exists(self.dload):
            pass
        else:
            os.mkdir(self.dload)
        torch.save(self.model.state_dict(), PATH)

    def load(self, PATH):
        """
        Loads the model's parameters from the path mentioned
        :param PATH: Should contain pickle file
        """
        self.model.load_state_dict(torch.load(PATH, map_location=self.device))

    def write_loss(self, epoch):
        for key, value in self.losses.items():
            if len(value) != 0:
                self.writer.add_scalar(key, value[-1], epoch)

