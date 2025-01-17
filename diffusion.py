import torch
import torch.nn.functional as F
from tqdm import tqdm
from unet import Unet1D
class DiffusionProcess:
    def __init__(self, timesteps, schedule):
        self.timesteps = timesteps
        self.schedule=schedule
        self.betas = self._get_beta_schedule(schedule)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Posterior variance calculation
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def _get_beta_schedule(self, schedule):
        """Select beta schedule"""
        if schedule == 'linear':
            return self.linear_beta_schedule()
        elif schedule == 'cosine':
            return self.cosine_beta_schedule()
        elif schedule == 'quadratic':
            return self.quadratic_beta_schedule()
        elif schedule == 'sigmoid':
            return self.sigmoid_beta_schedule()
        else:
            raise NotImplementedError(f"Unknown schedule: {schedule}")

    def cosine_beta_schedule(self, s=0.008):
        steps = self.timesteps + 1
        x = torch.linspace(0, self.timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def linear_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps)

    def quadratic_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start**0.5, beta_end**0.5, self.timesteps) ** 2

    def sigmoid_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(-6, 6, self.timesteps)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

    def extract(self, a, t, x_shape):
        t = t.clamp(0, a.size(0) - 1)
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, t, noise=None, loss_type="l2", anomaly_scores=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        t = torch.randint(0, self.timesteps, (x_start.shape[0],), device=x_start.device).long()

        # Forward diffusion step
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        predicted_noise = denoise_model(x_noisy, t, anomaly_scores=anomaly_scores)

        # Calculate loss
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, step_index, anomaly_scores=None):
        betas_t = self.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self.extract(self.sqrt_recip_alphas, t, x.shape)

        # Use model to predict the noise
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t, anomaly_scores=anomaly_scores) / sqrt_one_minus_alphas_cumprod_t
        )

        if step_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, x_start, denoise_steps, anomaly_scores=None):
        timesteps = denoise_steps
        b = shape[0]

        # Generate initial noisy input based on x_start
        noise = torch.randn_like(x_start)
        img = self.q_sample(
            x_start=x_start,
            t=torch.full((b,), timesteps, device=x_start.device, dtype=torch.long),
            noise=noise
        )

        # Perform reverse diffusion, starting from the noisy input
        for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
            t = torch.full((b,), i, device=x_start.device, dtype=torch.long)

            # Perform one reverse diffusion step, passing anomaly_scores
            img = self.p_sample(model, img, t, i, anomaly_scores=anomaly_scores)

        return img

    @torch.no_grad()
    def sample(self, model, shape, x_start, denoise_steps, anomaly_scores=None):
        return self.p_sample_loop(
            model,
            shape=shape,
            x_start=x_start,
            denoise_steps=denoise_steps,
            anomaly_scores=anomaly_scores
        )
import torch.nn as nn

class ConditionalDiffusionTrainingNetwork(nn.Module):
    def __init__(self,n_features,window_size,batch_size,timesteps,schedule,noise_steps,denoise_steps,dim,init_dim,dim_mults, channels, groups):
     super().__init__()
        # Timesteps for forward and reverse diffusion
     self.timesteps = noise_steps
     self.denoise_steps = denoise_steps
     self.n_features=n_features
     self.window_size=window_size,
     self.batch_size=batch_size
        # Define the UNet for denoising
     self.denoise_fn = Unet1D(
            dim,
            init_dim,
            dim_mults,
            channels,
            groups
        )

        # Create an instance of the DiffusionProcess
     self.diffusion = DiffusionProcess(timesteps,schedule)

    def forward(self, x, anomaly_scores):
        x=x.permute(0,2,1)
        
        # Randomly sample timesteps for diffusion during training
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=x.device).long()
        diffusion_loss = self.diffusion.p_losses(
            denoise_model=self.denoise_fn, 
            x_start=x, 
            t=t, 
            anomaly_scores=anomaly_scores
        )

        

    
      
        # Denoise the input during inference (reverse diffusion process)
        x_recon = self.diffusion.sample(
            model=self.denoise_fn,
            shape=(x.shape[0], self.n_features, self.window_size),
            x_start=x,
            denoise_steps=self.denoise_steps,
            anomaly_scores=anomaly_scores
        )

        return diffusion_loss ,x_recon
        