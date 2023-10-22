import torch
import hydra
import wandb
import os
import matplotlib.pyplot as plt

from diffusion_planner.model import TemporalUnet
from diffusion_planner.diffusion import GaussianDiffusion

import torch

class Trainer:
    def __init__(self, model: GaussianDiffusion) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    def train(self, batch):
        loss, infos = self.model.loss(batch, None)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, infos

def dataset(batch_size: int, device: str="cuda"):
    T = 128
    t = torch.linspace(0, torch.pi * 2, T)
    x = torch.stack([
        torch.sin(t),
        torch.cos(t),
        torch.sin(t) * torch.cos(t),
    ], dim=1).to(device)
    v = torch.stack([
        torch.cos(t),
        -torch.sin(t),
        torch.cos(t)**2 - torch.sin(t)**2,
    ], dim=1).to(device)
    low = torch.tensor([1., 1., 1.], device=device)
    high = torch.tensor([1.5, 1.5, 1.5], device=device)
    while True:
        length = torch.randint(64, 128, (batch_size, 1, 1), device=device)
        scale = (
            torch.rand(batch_size, 1,  3, device=device)
             * (high - low) + low
        )
        samples = torch.cat([
            x.unsqueeze(0) * scale,
            v.unsqueeze(0) * scale,
        ], dim=-1)
        terminal_state = samples.take_along_dim(length, dim=1)
        
        yield samples


def main():
    run = wandb.init(project="diffusion-planner")

    model = TemporalUnet(6)

    diffusion = GaussianDiffusion(
        model=model, 
        horizon=128,
        observation_dim=3,
        action_dim=3,
    ).to("cuda")

    trainer = Trainer(diffusion)
    
    torch.manual_seed(0)
    for i, batch in enumerate(dataset(128)):
        loss, infos = trainer.train(batch)
        
        if i % 1000 == 0:
            print(loss, infos)
            samples = trainer.model.sample(8)
            fig = plt.figure()
            axes = fig.subplots(2, 4, subplot_kw={'projection': '3d'})
            xs = samples[..., :3]
            for ax, x in zip(axes.flatten(), xs):
                ax.set_xlim(-2, 2)
                ax.set_ylim(-2, 2)
                ax.set_zlim(-2, 2)
                ax.plot(*x.T.cpu().numpy())
            fig.tight_layout()
            plt.close(fig)

            run.log({
                "loss": loss.item(),
                "samples": wandb.Image(fig),
            })
    

if __name__ == "__main__":
    main()