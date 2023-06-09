"""
This code is heavily based on https://github.com/lucidrains/stylegan2-pytorch, thanks to @lucidrains.
"""
import argparse
from datetime import datetime
import time
import math
import json
from math import floor, log2
from random import random
from shutil import rmtree
from functools import partial

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from adamp import AdamP
from torch.autograd import grad as torch_grad
from skimage.io import imsave
from pathlib import Path


#TODO Please fill the ... spot
# we import horovod for PyTorch using the torch module horovod.torch
import ...


from torch.utils.data import DataLoader, TensorDataset

EPS = 1e-8

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="data path (NPZ)", type=str,
    )
    parser.add_argument(
        "--name",
        help="experiment name, results will be saved in results/<name> and models/<name>",
        type=str,
    )
    parser.add_argument(
        "--image_size", help="image size", type=int, default=64,
    )
    parser.add_argument(
        "--epochs", help="number of epochs", type=int, default=1000,
    )
    parser.add_argument(
        "--batch_size", help="batch size", type=int, default=32,
    )
    args = parser.parse_args()
    return args


def get_dataset(path):
    train_images = np.load(path)["images"]
    train_images = train_images.transpose((0, 3, 1, 2))
    train_images = (torch.from_numpy(train_images) / 255.0).float()
    dataset = TensorDataset(train_images)
    return dataset


def get_dataloader(path, batch_size):
    dataset = get_dataset(path)
    # TODO: please fill the ... spots
    # Sharding happens using `torch.utils.data.distributed.DistributedSampler`
    # `num_replicas` should be the number of workers (total GPUs)
    # `rank` should be the rank of the current worker
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=..., rank=...)
    # no need for workers, data is fully in memory in this case
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        num_workers=0, 
        drop_last=True, 
        sampler=sampler,
    )
    return dataloader


# helper classes


class NanException(Exception):
    pass


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class RandomApply(nn.Module):
    def __init__(self, prob, fn, fn_else=lambda x: x):
        super().__init__()
        self.fn = fn
        self.fn_else = fn_else
        self.prob = prob

    def forward(self, x):
        fn = self.fn if random() < self.prob else self.fn_else
        return fn(x)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class PermuteToFrom(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out, loss = self.fn(x)
        out = out.permute(0, 3, 1, 2)
        return out, loss


def default(value, d):
    return d if value is None else value


def cycle(iterable, sampler):
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        for i in iterable:
            yield i
        epoch += 1
        #TODO: Please fill the ... spot.
        # We only display outputs on rank zero
        if hvd.rank() == ...:
            print(f'Epoch {epoch} finished')


def cast_list(el):
    return el if isinstance(el, list) else [el]


def is_empty(t):
    if isinstance(t, torch.Tensor):
        return t.nelement() == 0
    return t is None


def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException


def loss_backwards(fp16, loss, optimizer, loss_id, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer, loss_id) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        loss.backward(**kwargs)


def gradient_penalty(images, output, weight=10):
    batch_size = images.shape[0]
    gradients = torch_grad(
        outputs=output,
        inputs=images,
        grad_outputs=torch.ones(output.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(
        outputs=outputs,
        inputs=styles,
        grad_outputs=torch.ones(outputs.shape).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()


def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]


def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)


def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]


def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0.0, 1.0).cuda()


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)


def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


def set_requires_grad(model, bool):
    for p in model.parameters():
        p.requires_grad = bool


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (
        torch.sin(val * omega) / so
    ).unsqueeze(1) * high
    return res


# dataset


def convert_rgb_to_transparent(image):
    if image.mode == "RGB":
        return image.convert("RGBA")
    return image


def convert_transparent_to_rgb(image):
    if image.mode == "RGBA":
        return image.convert("RGB")
    return image


class expand_greyscale(object):
    def __init__(self, transparent):
        self.transparent = transparent

    def __call__(self, tensor):
        channels = tensor.shape[0]
        num_target_channels = 4 if self.transparent else 3

        if channels == num_target_channels:
            return tensor

        alpha = None
        if channels == 1:
            color = tensor.expand(3, -1, -1)
        elif channels == 2:
            color = tensor[:1].expand(3, -1, -1)
            alpha = tensor[1:]
        else:
            raise Exception(f"image with invalid number of channels given {channels}")

        if alpha is None and self.transparent:
            alpha = torch.ones(1, *tensor.shape[1:], device=tensor.device)

        return color if not self.transparent else torch.cat((color, alpha))


# augmentations


def random_float(lo, hi):
    return lo + (hi - lo) * random()


def random_crop_and_resize(tensor, scale):
    b, c, h, _ = tensor.shape
    new_width = int(h * scale)
    delta = h - new_width
    h_delta = int(random() * delta)
    w_delta = int(random() * delta)
    cropped = tensor[
        :, :, h_delta : (h_delta + new_width), w_delta : (w_delta + new_width)
    ].clone()
    return F.interpolate(cropped, size=(h, h), mode="bilinear")


def random_hflip(tensor, prob):
    if prob > random():
        return tensor
    return torch.flip(tensor, dims=(3,))


class AugWrapper(nn.Module):
    def __init__(self, D, image_size):
        super().__init__()
        self.D = D

    def forward(self, images, prob=0.0, detach=False):
        if random() < prob:
            random_scale = random_float(0.75, 0.95)
            images = random_hflip(images, prob=0.5)
            images = random_crop_and_resize(images, scale=random_scale)

        if detach:
            images.detach_()

        return self.D(images)


# stylegan2 classes


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, lr_mul=1, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))

        self.lr_mul = lr_mul

    def forward(self, input):
        return F.linear(input, self.weight * self.lr_mul, bias=self.bias * self.lr_mul)


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth, lr_mul=0.1):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([EqualLinear(emb, emb, lr_mul), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = F.normalize(x, dim=1)
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba=False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        out_filters = 3 if not rgba else 4
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(
        self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs
    ):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(
            self.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
        )

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class GeneratorBlock(nn.Module):
    def __init__(
        self,
        latent_dim,
        input_channels,
        filters,
        upsample=True,
        upsample_rgb=True,
        rgba=False,
    ):
        super().__init__()
        self.upsample = (
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            if upsample
            else None
        )

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, : x.shape[2], : x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(
            input_channels, filters, 1, stride=(2 if downsample else 1)
        )

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(),
        )

        self.downsample = (
            nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x


class Generator(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim,
        network_capacity=16,
        transparent=False,
        attn_layers=[],
        no_const=False,
        fmap_max=512,
    ):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        filters = [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][
            ::-1
        ]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        init_channels = filters[0]
        filters = [init_channels, *filters]

        in_out_pairs = zip(filters[:-1], filters[1:])
        self.no_const = no_const

        if no_const:
            self.to_initial_block = nn.ConvTranspose2d(
                latent_dim, init_channels, 4, 1, 0, bias=False
            )
        else:
            self.initial_block = nn.Parameter(torch.randn((1, init_channels, 4, 4)))

        self.initial_conv = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.blocks = nn.ModuleList([])
        self.attns = nn.ModuleList([])

        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)
            num_layer = self.num_layers - ind

            attn_fn = attn_and_ff(in_chan) if num_layer in attn_layers else None

            self.attns.append(attn_fn)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                rgba=transparent,
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise):
        batch_size = styles.shape[0]

        if self.no_const:
            avg_style = styles.mean(dim=1)[:, :, None, None]
            x = self.to_initial_block(avg_style)
        else:
            x = self.initial_block.expand(batch_size, -1, -1, -1)

        rgb = None
        styles = styles.transpose(0, 1)
        x = self.initial_conv(x)

        for style, block, attn in zip(styles, self.blocks, self.attns):
            if attn is not None:
                x = attn(x)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


class Discriminator(nn.Module):
    def __init__(
        self,
        image_size,
        network_capacity=16,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        transparent=False,
        fmap_max=512,
    ):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(64) * (2 ** i) for i in range(num_layers + 1)]

        set_fmap_max = partial(min, fmap_max)
        filters = list(map(set_fmap_max, filters))
        chan_in_out = list(zip(filters[:-1], filters[1:]))

        blocks = []
        attn_blocks = []
        quantize_blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind != (len(chan_in_out) - 1)

            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

            attn_fn = attn_and_ff(out_chan) if num_layer in attn_layers else None

            attn_blocks.append(attn_fn)

            quantize_fn = (
                PermuteToFrom(VectorQuantize(out_chan, fq_dict_size))
                if num_layer in fq_layers
                else None
            )
            quantize_blocks.append(quantize_fn)

        self.blocks = nn.ModuleList(blocks)
        self.attn_blocks = nn.ModuleList(attn_blocks)
        self.quantize_blocks = nn.ModuleList(quantize_blocks)

        chan_last = filters[-1]
        latent_dim = 2 * 2 * chan_last

        self.final_conv = nn.Conv2d(chan_last, chan_last, 3, padding=1)
        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        quantize_loss = torch.zeros(1).to(x)

        for (block, attn_block, q_block) in zip(
            self.blocks, self.attn_blocks, self.quantize_blocks
        ):
            x = block(x)

            if attn_block is not None:
                x = attn_block(x)

            if q_block is not None:
                x, loss = q_block(x)
                quantize_loss += loss

        x = self.final_conv(x)
        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze(), quantize_loss


class StyleGAN2(nn.Module):
    def __init__(
        self,
        image_size,
        latent_dim=512,
        fmap_max=512,
        style_depth=8,
        network_capacity=16,
        transparent=False,
        fp16=False,
        cl_reg=False,
        steps=1,
        lr=1e-4,
        ttur_mult=2,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        lr_mlp=0.1,
    ):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.995)

        self.S = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.G = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
            fmap_max=fmap_max,
        )
        self.D = Discriminator(
            image_size,
            network_capacity,
            fq_layers=fq_layers,
            fq_dict_size=fq_dict_size,
            attn_layers=attn_layers,
            transparent=transparent,
            fmap_max=fmap_max,
        )

        self.SE = StyleVectorizer(latent_dim, style_depth, lr_mul=lr_mlp)
        self.GE = Generator(
            image_size,
            latent_dim,
            network_capacity,
            transparent=transparent,
            attn_layers=attn_layers,
            no_const=no_const,
        )

        self.D_cl = None

        if cl_reg:
            from contrastive_learner import ContrastiveLearner

            # experimental contrastive loss discriminator regularization
            assert (
                not transparent
            ), "contrastive loss regularization does not work with transparent images yet"
            self.D_cl = ContrastiveLearner(self.D, image_size, hidden_layer="flatten")

        # wrapper for augmenting all images going into the discriminator
        self.D_aug = AugWrapper(self.D, image_size)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = AdamP(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = AdamP(
            self.D.parameters(), lr=self.lr * ttur_mult, betas=(0.5, 0.9)
        )
        #TODO: please fill the ... spots
        # We wrap both the generator and discriminator optimizers
        # using `hvd.DistributedOptimizer`
        self.G_opt = hvd.DistributedOptimizer(...)
        self.D_opt = hvd.DistributedOptimizer(...)
        self._init_weights()
        self.reset_parameter_averaging()

        self.cuda()

        self.fp16 = fp16
        if fp16:
            (
                (self.S, self.G, self.D, self.SE, self.GE),
                (self.G_opt, self.D_opt),
            ) = amp.initialize(
                [self.S, self.G, self.D, self.SE, self.GE],
                [self.G_opt, self.D_opt],
                opt_level="O1",
                num_losses=3,
            )

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(
                    m.weight, a=0, mode="fan_in", nonlinearity="leaky_relu"
                )

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()
            ):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


class Trainer:
    def __init__(
        self,
        name,
        results_dir,
        models_dir,
        image_size,
        network_capacity,
        transparent=False,
        batch_size=4,
        mixed_prob=0.9,
        gradient_accumulate_every=1,
        lr=2e-4,
        lr_mlp=1.0,
        ttur_mult=2,
        num_workers=None,
        save_every=1000,
        trunc_psi=0.6,
        fp16=False,
        cl_reg=False,
        fq_layers=[],
        fq_dict_size=256,
        attn_layers=[],
        no_const=False,
        aug_prob=0.0,
        dataset_aug_prob=0.0,
        steps_per_epoch=None,
        *args,
        **kwargs,
    ):
        self.GAN_params = [args, kwargs]
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / "config.json"

        assert log2(
            image_size
        ).is_integer(), "image size must be a power of 2 (64, 128, 256, 512, 1024)"
        self.image_size = image_size
        self.network_capacity = network_capacity
        self.transparent = transparent
        self.fq_layers = cast_list(fq_layers)
        self.fq_dict_size = fq_dict_size

        self.attn_layers = cast_list(attn_layers)
        self.no_const = no_const
        self.aug_prob = aug_prob

        self.lr = lr
        self.lr_mlp = lr_mlp
        self.ttur_mult = ttur_mult
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob

        self.save_every = save_every
        self.steps = 0

        self.av = None
        self.trunc_psi = trunc_psi

        self.pl_mean = None

        self.gradient_accumulate_every = gradient_accumulate_every

        self.fp16 = fp16

        self.cl_reg = cl_reg

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.last_cr_loss = 0
        self.q_loss = 0

        self.pl_length_ma = EMA(0.99)
        self.init_folders()

        self.loader = None
        self.dataset_aug_prob = dataset_aug_prob
        self.steps_per_epoch = steps_per_epoch

    def init_GAN(self):
        args, kwargs = self.GAN_params
        self.GAN = StyleGAN2(
            lr=self.lr,
            lr_mlp=self.lr_mlp,
            ttur_mult=self.ttur_mult,
            image_size=self.image_size,
            network_capacity=self.network_capacity,
            transparent=self.transparent,
            fq_layers=self.fq_layers,
            fq_dict_size=self.fq_dict_size,
            attn_layers=self.attn_layers,
            fp16=self.fp16,
            cl_reg=self.cl_reg,
            no_const=self.no_const,
            *args,
            **kwargs,
        )
        #TODO please fill the ... spots
        # We broadcast the discriminator and generator parameters
        # as well as the optimizer states from rank zero
        hvd.broadcast_parameters(self.GAN.state_dict(), root_rank=...)
        hvd.broadcast_optimizer_state(self.GAN.G_opt, root_rank=...)
        hvd.broadcast_optimizer_state(self.GAN.D_opt, root_rank=...)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = (
            self.config()
            if not self.config_path.exists()
            else json.loads(self.config_path.read_text())
        )
        self.image_size = config["image_size"]
        self.network_capacity = config["network_capacity"]
        self.transparent = config["transparent"]
        self.fq_layers = config["fq_layers"]
        self.fq_dict_size = config["fq_dict_size"]
        self.attn_layers = config.pop("attn_layers", [])
        self.no_const = config.pop("no_const", False)
        del self.GAN
        self.init_GAN()

    def config(self):
        return {
            "image_size": self.image_size,
            "network_capacity": self.network_capacity,
            "transparent": self.transparent,
            "fq_layers": self.fq_layers,
            "fq_dict_size": self.fq_dict_size,
            "attn_layers": self.attn_layers,
            "no_const": self.no_const,
        }

    def set_data_src(self, path):
        train_loader = get_dataloader(path, self.batch_size)
        self.loader = cycle(train_loader, train_loader.sampler)

    def train(self):
        assert (
            self.loader is not None
        ), "You must first initialize the data source with `.set_data_src(<folder of images>)`"

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.0).cuda()
        total_gen_loss = torch.tensor(0.0).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        aug_prob = self.aug_prob

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps > 5000 and self.steps % 32 == 0
        apply_cl_reg_to_generated = self.steps > 20000

        backwards = partial(loss_backwards, self.fp16)

        if self.GAN.D_cl is not None:
            self.GAN.D_opt.zero_grad()

            if apply_cl_reg_to_generated:
                for i in range(self.gradient_accumulate_every):
                    get_latents_fn = (
                        mixed_list if random() < self.mixed_prob else noise_list
                    )
                    style = get_latents_fn(batch_size, num_layers, latent_dim)
                    noise = image_noise(batch_size, image_size)

                    w_space = latent_to_w(self.GAN.S, style)
                    w_styles = styles_def_to_tensor(w_space)

                    generated_images = self.GAN.G(w_styles, noise)
                    self.GAN.D_cl(generated_images.clone().detach(), accumulate=True)

            for i in range(self.gradient_accumulate_every):
                (image_batch,) = next(self.loader)
                image_batch = image_batch.cuda()
                self.GAN.D_cl(image_batch, accumulate=True)

            loss = self.GAN.D_cl.calculate_loss()
            self.last_cr_loss = loss.clone().detach().item()
            backwards(loss, self.GAN.D_opt, 0)

            self.GAN.D_opt.step()

        # train discriminator

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            get_latents_fn = mixed_list if random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output, fake_q_loss = self.GAN.D_aug(
                generated_images.clone().detach(), detach=True, prob=aug_prob
            )

            (image_batch,) = next(self.loader)
            image_batch = image_batch.cuda()
            image_batch.requires_grad_()
            real_output, real_q_loss = self.GAN.D_aug(image_batch, prob=aug_prob)

            divergence = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()
            disc_loss = divergence

            quantize_loss = (fake_q_loss + real_q_loss).mean()
            self.q_loss = float(quantize_loss.detach().item())

            disc_loss = disc_loss + quantize_loss

            if apply_gradient_penalty:
                gp = gradient_penalty(image_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt, 1)

            total_disc_loss += (
                divergence.detach().item() / self.gradient_accumulate_every
            )
            self.nbims += len(image_batch)

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_images = self.GAN.G(w_styles, noise)
            fake_output, _ = self.GAN.D_aug(generated_images, prob=aug_prob)
            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt, 2)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not np.isnan(avg_pl_length):
            self.pl_mean = self.pl_length_ma.update_average(self.pl_mean, avg_pl_length)

        if self.steps % 10 == 0 and self.steps > 20000:
            self.GAN.EMA()

        if self.steps <= 25000 and self.steps % 1000 == 2:
            self.GAN.reset_parameter_averaging()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(
                f"NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}"
            )
            self.load(checkpoint_num)
            raise NanException

        # periodically save results
        # TODO Please fill the ... spot
        # saving the model to disk should be done on rank zero
        if (hvd.rank() == ...) and (self.steps % self.save_every == 0):
            self.save(checkpoint_num)

        # TODO Please fill the ... spot
        # evaluate should be called on rank zero
        if (hvd.rank() == ...) and (
            self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500)
        ):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num=0, num_image_tiles=8, trunc=1.0):
        self.GAN.eval()
        ext = "jpg" if not self.transparent else "png"
        num_rows = num_image_tiles

        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        # regular

        generated_images = self.generate_truncated(
            self.GAN.S, self.GAN.G, latents, n, trunc_psi=self.trunc_psi
        )
        save_image(
            generated_images,
            str(self.results_dir / self.name / f"{str(num)}.{ext}"),
            nrow=num_rows,
        )

        # moving averages

        generated_images = self.generate_truncated(
            self.GAN.SE, self.GAN.GE, latents, n, trunc_psi=self.trunc_psi
        )
        save_image(
            generated_images,
            str(self.results_dir / self.name / f"{str(num)}-ema.{ext}"),
            nrow=num_rows,
        )

        # mixing regularities

        # def tile(a, dim, n_tile):
        # init_dim = a.size(dim)
        # repeat_idx = [1] * a.dim()
        # repeat_idx[dim] = n_tile
        # a = a.repeat(*(repeat_idx))
        # order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        # return torch.index_select(a, dim, order_index)

        # nn = noise(num_rows, latent_dim)
        # tmp1 = tile(nn, 0, num_rows)
        # tmp2 = nn.repeat(num_rows, 1)

        # tt = int(num_layers / 2)
        # mixed_latents = [(tmp1, tt), (tmp2, num_layers - tt)]

        # generated_images = self.generate_truncated(self.GAN.SE, self.GAN.GE, mixed_latents, n, trunc_psi = self.trunc_psi)
        # save_image(generated_images, str(self.results_dir / self.name / f'{str(num)}-mr.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi=0.75, num_image_tiles=8):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis=0)
            self.av = np.expand_dims(self.av, axis=0)

        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks(self.batch_size, G, w_styles, noi)
        return generated_images.clamp_(0.0, 1.0)

    def print_log(self, it):
        pl_mean = default(self.pl_mean, 0)
        print(
            f"it: {it} G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {pl_mean:.2f} | CR: {self.last_cr_loss:.2f} | Q: {self.q_loss:.2f}"
        )

    def model_name(self, num):
        return str(self.models_dir / self.name / f"model_{num}.pt")

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(f"./models/{self.name}", True)
        rmtree(f"./results/{self.name}", True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        save_data = {"GAN": self.GAN.state_dict()}

        if self.GAN.fp16:
            save_data["amp"] = amp.state_dict()

        torch.save(save_data, self.model_name(num))
        self.write_config()

    def load(self, num=-1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [
                p for p in Path(self.models_dir / self.name).glob("model_*.pt")
            ]
            saved_nums = sorted(map(lambda x: int(x.stem.split("_")[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f"continuing from previous epoch - {name}")

        self.steps = name * self.save_every

        load_data = torch.load(self.model_name(name))

        # make backwards compatible
        if "GAN" not in load_data:
            load_data = {"GAN": load_data}

        self.GAN.load_state_dict(load_data["GAN"])

        if self.GAN.fp16 and "amp" in load_data:
            amp.load_state_dict(load_data["amp"])


def save_image(X, path, nrow=8):
    ncol = len(X) // nrow
    h, w = X.shape[2], X.shape[3]

    X = X.data.cpu().numpy()
    X = X.transpose((0, 2, 3, 1))
    grid = np.zeros((h * nrow, w * ncol, 3))
    it = 0
    for y in range(0, grid.shape[0], h):
        for x in range(0, grid.shape[1], w):
            grid[y : y + h, x : x + w] = X[it]
            it += 1
    grid = (grid * 255).astype("uint8")
    imsave(path, grid)


def train(
    *,
    data="data",
    data_type="image_folder",
    results_dir="results",
    models_dir="models",
    name="default",
    new=False,
    load_from=-1,
    image_size=128,
    network_capacity=16,
    transparent=False,
    batch_size=5,
    gradient_accumulate_every=1,
    num_train_steps=150000,
    learning_rate=2e-4,
    lr_mlp=0.1,
    ttur_mult=1.5,
    num_workers=1,
    save_every=1000,
    generate=False,
    generate_interpolation=False,
    save_frames=False,
    num_image_tiles=8,
    trunc_psi=0.75,
    fp16=False,
    cl_reg=False,
    fq_dict_size=256,
    fq_layers=None,
    attn_layers=None,
    no_const=False,
    aug_prob=0.0,
    dataset_aug_prob=0.0,
):
    if fq_layers is None:
        fq_layers = []
    if attn_layers is None:
        attn_layers = []

    #TODO Please fill the ... spot
    # we set the CUDA device id to the local rank
    torch.cuda.set_device(...)
    torch.backends.cudnn.benchmark = True
    model = Trainer(
        name,
        results_dir,
        models_dir,
        batch_size=batch_size,
        gradient_accumulate_every=gradient_accumulate_every,
        image_size=image_size,
        network_capacity=network_capacity,
        transparent=transparent,
        lr=learning_rate,
        lr_mlp=lr_mlp,
        ttur_mult=ttur_mult,
        num_workers=num_workers,
        save_every=save_every,
        trunc_psi=trunc_psi,
        fp16=fp16,
        cl_reg=cl_reg,
        fq_layers=fq_layers,
        fq_dict_size=fq_dict_size,
        attn_layers=attn_layers,
        no_const=no_const,
        aug_prob=aug_prob,
        dataset_aug_prob=dataset_aug_prob,
    )

    if not new:
        model.load(load_from)
    else:
        model.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f"generated-{timestamp}"
        model.evaluate(samples_name, num_image_tiles)
        print(f"sample images generated at {results_dir}/{name}/{samples_name}")
        return

    if generate_interpolation:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f"generated-{timestamp}"
        model.generate_interpolation(
            samples_name, num_image_tiles, save_frames=save_frames
        )
        print(f"interpolation generated at {results_dir}/{name}/{samples_name}")
        return

    model.set_data_src(data)

    model.nbims = 0
    start = time.time()
    for it in range(num_train_steps - model.steps):
        model.train()
        if hvd.rank() == 0 and it % 100 == 0:
            model.print_log(it + model.steps)
            duration = time.time() - start
            print("nb images per second", model.nbims / duration)
    duration = time.time() - start
    print(f"total images/sec: {(model.nbims/duration)*hvd.size()}")


if __name__ == "__main__":
    #TODO Please fill the ... spot
    # here, we initialize Horovod
    ...

    args = arg_parse()
    dataset = get_dataset(args.data_path)
    steps_per_epoch = len(dataset) // (args.batch_size * hvd.size())
    steps = steps_per_epoch * args.epochs
    train(
        image_size=args.image_size,
        name=args.name,
        data=args.data_path,
        num_train_steps=steps,
    )
