import torch
import torch.nn as nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce


pair = lambda x: x if isinstance(x, tuple) else (x, x)


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


class ParallelSum(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return sum(map(lambda fn: fn(x), self.fns))


def Permutator(*, image_size, patch_size, dim, depth, num_classes, segments, expansion_factor = 4, dropout = 0.):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    assert (dim % segments) == 0, 'dimension must be divisible by the number of segments'
    height = width = image_size // patch_size
    s = segments

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * 3, dim),
        *[nn.Sequential(
            PreNormResidual(dim, nn.Sequential(
                ParallelSum(
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b w c (h s)', s = s),
                        nn.Linear(height * s, height * s),
                        Rearrange('b w c (h s) -> b h w (c s)', s = s),
                    ),
                    nn.Sequential(
                        Rearrange('b h w (c s) -> b h c (w s)', s = s),
                        nn.Linear(width * s, width * s),
                        Rearrange('b h c (w s) -> b h w (c s)', s = s),
                    ),
                    nn.Linear(dim, dim)
                ),
                nn.Linear(dim, dim)
            )),
            PreNormResidual(dim, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * expansion_factor, dim),
                nn.Dropout(dropout)
            ))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b h w c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class Classifier(nn.Module):
    def __init__(self, start_dim: int = 256):
        super(Classifier, self).__init__()
        self.head = nn.Sequential(nn.Linear(start_dim, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                  nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Linear(64, 1))

    def forward(self, x):
        x = self.head(x)

        return x


def Combined(checkpoint_path: str = None):
    base_model = MLPMixer(image_size=(1024, 64), channels=1, patch_size=16, dim=256, depth=4, num_classes=256)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        base_model.load_state_dict(checkpoint['model'])

    head = Classifier()

    return nn.Sequential(base_model, head)
