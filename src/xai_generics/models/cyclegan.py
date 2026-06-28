from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required for the local CycleGAN runtime. "
            "Install torch in the project environment before running training or inference."
        ) from exc
    return torch, nn, optim


def resolve_submodule(module: Any, dotted_path: str) -> Any:
    current = module
    if not dotted_path:
        return current
    for part in dotted_path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _conv_norm_relu(
    nn: Any,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
) -> Any:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        ),
        nn.InstanceNorm2d(out_channels),
        nn.ReLU(True),
    )


def _conv_norm_lrelu(
    nn: Any,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    normalize: bool = True,
) -> Any:
    layers = [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not normalize,
        )
    ]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_channels))
    layers.append(nn.LeakyReLU(0.2, True))
    return nn.Sequential(*layers)


def _resnet_block(nn: Any, channels: int) -> Any:
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(channels, channels, kernel_size=3, bias=False),
        nn.InstanceNorm2d(channels),
        nn.ReLU(True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(channels, channels, kernel_size=3, bias=False),
        nn.InstanceNorm2d(channels),
    )


def _build_resnet_generator(
    nn: Any,
    input_nc: int,
    output_nc: int,
    ngf: int,
    n_blocks: int = 9,
) -> Any:
    class ResidualBlock(nn.Module):
        def __init__(self, channels: int) -> None:
            super().__init__()
            self.block = _resnet_block(nn, channels)

        def forward(self, x):  # type: ignore[no-untyped-def]
            return x + self.block(x)

    layers = [
        nn.ReflectionPad2d(3),
        nn.Conv2d(input_nc, ngf, kernel_size=7, bias=False),
        nn.InstanceNorm2d(ngf),
        nn.ReLU(True),
        _conv_norm_relu(nn, ngf, ngf * 2, kernel_size=3, stride=2, padding=1),
        _conv_norm_relu(nn, ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1),
    ]
    for _ in range(n_blocks):
        layers.append(ResidualBlock(ngf * 4))
    layers.extend(
        [
            nn.ConvTranspose2d(
                ngf * 4,
                ngf * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                ngf * 2,
                ngf,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]
    )
    return nn.Sequential(*layers)


def _build_patch_discriminator(
    nn: Any,
    input_nc: int,
    ndf: int,
) -> Any:
    return nn.Sequential(
        _conv_norm_lrelu(nn, input_nc, ndf, kernel_size=4, stride=2, padding=1, normalize=False),
        _conv_norm_lrelu(nn, ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        _conv_norm_lrelu(nn, ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        _conv_norm_lrelu(nn, ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
    )


@dataclass
class CycleGANBundle:
    config: dict[str, Any]
    device: str
    netG_A: Any
    netG_B: Any
    netD_A: Any
    netD_B: Any
    optimizer_G: Any
    optimizer_D: Any

    @classmethod
    def from_config(cls, config: dict[str, Any], device: str | None = None) -> "CycleGANBundle":
        torch, nn, optim = require_torch()
        model_cfg = config["model"]
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model_cfg["netG"] != "resnet_9blocks":
            raise ValueError(f"Unsupported netG for local runtime: {model_cfg['netG']}")
        if model_cfg["netD"] != "basic":
            raise ValueError(f"Unsupported netD for local runtime: {model_cfg['netD']}")
        netG_A = _build_resnet_generator(
            nn,
            input_nc=int(model_cfg["input_nc"]),
            output_nc=int(model_cfg["output_nc"]),
            ngf=int(model_cfg["ngf"]),
            n_blocks=int(model_cfg.get("n_blocks", 9)),
        ).to(resolved_device)
        netG_B = _build_resnet_generator(
            nn,
            input_nc=int(model_cfg["output_nc"]),
            output_nc=int(model_cfg["input_nc"]),
            ngf=int(model_cfg["ngf"]),
            n_blocks=int(model_cfg.get("n_blocks", 9)),
        ).to(resolved_device)
        netD_A = _build_patch_discriminator(
            nn,
            input_nc=int(model_cfg["output_nc"]),
            ndf=int(model_cfg["ndf"]),
        ).to(resolved_device)
        netD_B = _build_patch_discriminator(
            nn,
            input_nc=int(model_cfg["input_nc"]),
            ndf=int(model_cfg["ndf"]),
        ).to(resolved_device)
        optimizer_G = optim.Adam(
            list(netG_A.parameters()) + list(netG_B.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        optimizer_D = optim.Adam(
            list(netD_A.parameters()) + list(netD_B.parameters()),
            lr=0.0002,
            betas=(0.5, 0.999),
        )
        return cls(
            config=config,
            device=resolved_device,
            netG_A=netG_A,
            netG_B=netG_B,
            netD_A=netD_A,
            netD_B=netD_B,
            optimizer_G=optimizer_G,
            optimizer_D=optimizer_D,
        )

    def parameter_counts(self) -> dict[str, int]:
        return {
            "netG_A": sum(parameter.numel() for parameter in self.netG_A.parameters()),
            "netG_B": sum(parameter.numel() for parameter in self.netG_B.parameters()),
            "netD_A": sum(parameter.numel() for parameter in self.netD_A.parameters()),
            "netD_B": sum(parameter.numel() for parameter in self.netD_B.parameters()),
        }

    def save_checkpoint(self, checkpoint_dir: Path, tag: str) -> None:
        torch, _, _ = require_torch()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.netG_A.state_dict(), checkpoint_dir / f"{tag}_net_G_A.pth")
        torch.save(self.netG_B.state_dict(), checkpoint_dir / f"{tag}_net_G_B.pth")
        torch.save(self.netD_A.state_dict(), checkpoint_dir / f"{tag}_net_D_A.pth")
        torch.save(self.netD_B.state_dict(), checkpoint_dir / f"{tag}_net_D_B.pth")

    def save_training_state(
        self,
        checkpoint_dir: Path,
        tag: str,
        *,
        epoch: int,
        global_step: int,
        history_path: str | None = None,
    ) -> Path:
        torch, _, _ = require_torch()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "config": self.config,
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "history_path": history_path,
        }
        if torch.cuda.is_available():
            state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        state_path = checkpoint_dir / f"{tag}_state.pt"
        torch.save(state, state_path)
        return state_path

    def load_checkpoint(self, checkpoint_dir: Path, tag: str, strict: bool = True) -> None:
        torch, _, _ = require_torch()
        self.netG_A.load_state_dict(
            torch.load(checkpoint_dir / f"{tag}_net_G_A.pth", map_location=self.device),
            strict=strict,
        )
        self.netG_B.load_state_dict(
            torch.load(checkpoint_dir / f"{tag}_net_G_B.pth", map_location=self.device),
            strict=strict,
        )
        self.netD_A.load_state_dict(
            torch.load(checkpoint_dir / f"{tag}_net_D_A.pth", map_location=self.device),
            strict=strict,
        )
        self.netD_B.load_state_dict(
            torch.load(checkpoint_dir / f"{tag}_net_D_B.pth", map_location=self.device),
            strict=strict,
        )

    def load_training_state(self, checkpoint_dir: Path, tag: str) -> dict[str, Any]:
        torch, _, _ = require_torch()
        state = torch.load(checkpoint_dir / f"{tag}_state.pt", map_location="cpu")
        self.optimizer_G.load_state_dict(state["optimizer_G"])
        self.optimizer_D.load_state_dict(state["optimizer_D"])
        target_device = torch.device(self.device)
        for optimizer in (self.optimizer_G, self.optimizer_D):
            for optimizer_state in optimizer.state.values():
                for key, value in list(optimizer_state.items()):
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(target_device)
        if "torch_rng_state" in state:
            torch.set_rng_state(state["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        return state

    def available_module_names(self, side: str = "A") -> list[str]:
        if side == "A":
            root = self.netG_A
        elif side == "B":
            root = self.netG_B
        else:
            raise ValueError(f"Unsupported side: {side}")
        return [name for name, _ in root.named_modules()]
