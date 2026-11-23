from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    latent_dim: int = 256
    base_channels: int = 16

    architecture: str = "vector"
    latent_channels: int = 8
    residual_blocks: int = 2
    attention: bool = True
    downsample_factor: int = 16
    gradient_checkpointing: bool = False

    @property
    def encoded_dim(self):
        return self.latent_channels * (240//self.downsample_factor) * (256//self.downsample_factor) if self.architecture == "spatial" else self.latent_dim

    def __post_init__(self):
        if self.downsample_factor not in (8,16):
            raise ValueError('downsample_factor must be 8 or 16')
        if self.architecture not in ('vector', 'spatial') or self.latent_channels < 1 or self.residual_blocks < 1:
            raise ValueError('Invalid spatial architecture configuration')
        if self.latent_dim < 1 or self.base_channels < 4 or self.base_channels % 4:
            raise ValueError('latent_dim must be positive; base_channels must be a multiple of 4 >= 4')


class VisionVAE(nn.Module):
    """RGB [B,3,240,256] -> diagonal Gaussian [B,Z] -> native RGB.

    The legacy vector architecture pads internally; the spatial one stays native.
    decode(mu) is the deterministic reconstruction used for evaluation.
    """
    def __init__(self, config=VAEConfig()):
        super().__init__()
        self.config = config
        if config.architecture == "spatial":
            self.spatial = SpatialBackbone(config)
            return
        b = config.base_channels
        channels = [3, b, 2*b, 4*b, 8*b, 8*b]
        layers = []
        for a, c in zip(channels, channels[1:]):
            layers += [nn.Conv2d(a, c, 4, 2, 1), nn.GroupNorm(4, c), nn.SiLU()]
        self.encoder = nn.Sequential(*layers)
        self.feature_channels = channels[-1]
        flattened = channels[-1] * 8 * 8
        self.mu = nn.Linear(flattened, config.latent_dim)
        self.logvar = nn.Linear(flattened, config.latent_dim)
        self.from_latent = nn.Linear(config.latent_dim, flattened)
        layers = []
        for a, c in zip(reversed(channels[1:]), list(reversed(channels[1:-1])) + [3]):
            layers += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(a, c, 3, padding=1)]
            if c != 3:
                layers += [nn.GroupNorm(4, c), nn.SiLU()]
        self.decoder = nn.Sequential(*layers)

    def encode(self, rgb):
        if rgb.ndim != 4 or tuple(rgb.shape[1:]) != (3, 240, 256):
            raise ValueError('Expected RGB tensor [B,3,240,256] normalized to [0,1]')
        if self.config.architecture == "spatial":
            return self.spatial.encode(rgb)
        h = self.encoder(F.pad(rgb, (0, 0, 0, 16), mode='replicate')).flatten(1)
        return self.mu(h), self.logvar(h).clamp(-12, 8)

    def decode(self, z):
        if self.config.architecture == "spatial":
            return self.spatial.decode(z)
        h = self.from_latent(z).reshape(-1, self.feature_channels, 8, 8)
        return self.decoder(h).sigmoid()[:, :, :240, :256]

    def forward(self, rgb, sample=None):
        mu, logvar = self.encode(rgb)
        if sample is None:
            sample = self.training
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        return self.decode(z), mu, logvar


def vae_loss(reconstruction, target, mu, logvar, beta):
    # Both terms are means: reconstruction per pixel/channel; KL per latent dimension.
    mse = F.mse_loss(reconstruction, target)
    kl = (-0.5 * (1 + logvar - mu.square() - logvar.exp())).mean()
    return mse + beta * kl, mse, kl


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.GroupNorm(4, channels), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GroupNorm(4, channels),
            nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class SpatialAttention(nn.Module):
    """Global context at the compressed 15x16 or 30x32 bottleneck."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(4, channels)
        self.qkv = nn.Conv2d(channels, channels*3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.heads = 4

    def forward(self, x):
        b,c,h,w = x.shape
        q,k,v = self.qkv(self.norm(x)).reshape(b,3,self.heads,c//self.heads,h*w).unbind(1)
        y = F.scaled_dot_product_attention(q.transpose(-1,-2), k.transpose(-1,-2), v.transpose(-1,-2))
        return x + self.proj(y.transpose(-1,-2).reshape(b,c,h,w))


class SpatialBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        b = config.base_channels
        channels = [b, b*2, b*4, b*4] + ([b*4] if config.downsample_factor == 16 else [])
        encoder = [nn.Conv2d(3,b,3,padding=1)]
        for a,c in zip(channels, channels[1:]):
            encoder += [ResidualBlock(a) for _ in range(config.residual_blocks)]
            encoder += [nn.Conv2d(a,c,4,stride=2,padding=1)]
        encoder += [ResidualBlock(channels[-1])]
        if config.attention: encoder += [SpatialAttention(channels[-1])]
        encoder += [nn.GroupNorm(4,channels[-1]),nn.SiLU(),
                    nn.Conv2d(channels[-1],2*config.latent_channels,1)]
        self.encoder = nn.Sequential(*encoder)
        decoder = [nn.Conv2d(config.latent_channels,channels[-1],3,padding=1)]
        if config.attention: decoder += [SpatialAttention(channels[-1])]
        decoder += [ResidualBlock(channels[-1])]
        for a,c in zip(reversed(channels[1:]), reversed(channels[:-1])):
            decoder += [nn.Upsample(scale_factor=2,mode='nearest'),nn.Conv2d(a,c,3,padding=1)]
            decoder += [ResidualBlock(c) for _ in range(config.residual_blocks)]
        decoder += [nn.GroupNorm(4,b),nn.SiLU(),nn.Conv2d(b,3,3,padding=1)]
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x):
        mu, logvar = self._run(self.encoder,x).chunk(2,dim=1)
        # Flat interface remains compatible with VMC; metadata preserves C,H,W.
        return mu.flatten(1), logvar.flatten(1).clamp(-12,8)

    def decode(self, z):
        f=self.config.downsample_factor
        return self._run(self.decoder,z.reshape(-1,self.config.latent_channels,240//f,256//f)).sigmoid()

    def _run(self, layers, x):
        if self.config.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint_sequential
            return checkpoint_sequential(layers,len(layers),x,use_reentrant=False)
        return layers(x)


def detail_loss(reconstruction, target, mu, logvar, beta, *, l1_weight=0.,
                edge_weight=0., roi_weight=0., masks=None, free_nats=0.):
    """All reductions are means; compute in float32 even under autocast.

    ROI weighting uses simulator boxes as training supervision, never encoder input.
    The latent is the only encoder-to-decoder connection (no skip bypass).
    """
    y, x, mu, logvar = [t.float() for t in (reconstruction,target,mu,logvar)]
    mse = (y-x).square().mean()
    l1 = (y-x).abs().mean()
    edge = ((y[:,:,1:]-y[:,:,:-1])-(x[:,:,1:]-x[:,:,:-1])).abs().mean()
    edge += ((y[:,:,:,1:]-y[:,:,:,:-1])-(x[:,:,:,1:]-x[:,:,:,:-1])).abs().mean()
    roi = y.new_zeros(())
    if masks is not None and roi_weight:
        weights = masks[:,None].to(y.dtype)
        roi = ((y-x).square()*weights).sum()/(weights.sum()*3).clamp_min(1)
    kl_terms = -.5*(1+logvar-mu.square()-logvar.exp())
    kl = kl_terms.mean()
    regularized_kl = kl_terms.mean(0).clamp_min(free_nats).mean()
    loss = mse + l1_weight*l1 + edge_weight*edge + roi_weight*roi + beta*regularized_kl
    return loss, {'mse':mse,'l1':l1,'edge':edge,'roi':roi,'kl':kl}
