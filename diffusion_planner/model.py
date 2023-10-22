import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TemporalUnet(nn.Module):
    def __init__(
        self, 
        output_dim: int,
        time_dim: int=32,
    ) -> None:
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.Mish(),
            nn.Linear(time_dim * 4, time_dim),
        )

        module_list = []
        for i in range(3):
            module_list.append(
                nn.Sequential(
                    nn.LazyConv1d(64, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SELU()
                )
            )
        self.downsample = nn.ModuleList(module_list)

        module_list = []
        for i in range(3):
            module_list.append(
                nn.Sequential(
                    nn.LazyConvTranspose1d(64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.GroupNorm(8, 64),
                    nn.SELU()
                )
            )
        self.upsample = nn.ModuleList(module_list)

        self.initial = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
        )
        self.middel = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
            nn.SELU()
        )
        self.final = nn.Sequential(
            nn.LazyConv1d(64, kernel_size=3, padding=1),
            nn.SELU(),
            nn.LazyConv1d(output_dim, kernel_size=3, padding=1),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = einops.rearrange(x, 'b t d -> b d t')
        x = self.initial(x)
        time_emb = einops.repeat(self.time_mlp(t), "b d -> b d t", t=x.shape[-1])
        x = torch.cat([x, time_emb], dim=1)

        downsampled = []
        for downsample in self.downsample:
            downsampled.append(x)
            x = downsample(x)

        x = self.middel(x)

        for upsample, downsample in zip(self.upsample, reversed(downsampled)):
            # x = upsample(x) + downsample
            x = torch.cat([upsample(x), downsample], dim=1)
        
        x = self.final(x)
        x = einops.rearrange(x, 'b d t -> b t d')
        return x
        
