from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from xai_generics.models.cyclegan import CycleGANBundle, require_torch


def _to_3channel_tensor(torch: Any, image: np.ndarray, device: str) -> Any:
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got {array.shape}")
    normalized = (array / 127.5) - 1.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).repeat(3, 1, 1)
    return tensor.unsqueeze(0).to(device)


@dataclass
class CycleGANTrainer:
    bundle: CycleGANBundle
    lambda_cycle: float
    lambda_identity: float

    @classmethod
    def from_config(cls, config: dict[str, Any], device: str | None = None) -> "CycleGANTrainer":
        bundle = CycleGANBundle.from_config(config, device=device)
        training_cfg = config["training"]
        return cls(
            bundle=bundle,
            lambda_cycle=float(config["model"].get("lambda_cycle", 10.0)),
            lambda_identity=float(training_cfg.get("lambda_identity", 0.5)),
        )

    def _criteria(self) -> tuple[Any, Any, Any]:
        torch, nn, _ = require_torch()
        return nn.MSELoss(), nn.L1Loss(), nn.L1Loss()

    def train_step(self, source: np.ndarray, target: np.ndarray) -> dict[str, float]:
        torch, _, _ = require_torch()
        criterion_gan, criterion_cycle, criterion_idt = self._criteria()
        self.bundle.netG_A.train()
        self.bundle.netG_B.train()
        self.bundle.netD_A.train()
        self.bundle.netD_B.train()

        real_A = _to_3channel_tensor(torch, source, self.bundle.device)
        real_B = _to_3channel_tensor(torch, target, self.bundle.device)

        self.bundle.optimizer_G.zero_grad()
        same_B = self.bundle.netG_A(real_B)
        loss_idt_B = criterion_idt(same_B, real_B) * self.lambda_cycle * self.lambda_identity
        same_A = self.bundle.netG_B(real_A)
        loss_idt_A = criterion_idt(same_A, real_A) * self.lambda_cycle * self.lambda_identity

        fake_B = self.bundle.netG_A(real_A)
        pred_fake_B = self.bundle.netD_A(fake_B)
        loss_gan_A2B = criterion_gan(pred_fake_B, torch.ones_like(pred_fake_B))
        fake_A = self.bundle.netG_B(real_B)
        pred_fake_A = self.bundle.netD_B(fake_A)
        loss_gan_B2A = criterion_gan(pred_fake_A, torch.ones_like(pred_fake_A))

        rec_A = self.bundle.netG_B(fake_B)
        loss_cycle_A = criterion_cycle(rec_A, real_A) * self.lambda_cycle
        rec_B = self.bundle.netG_A(fake_A)
        loss_cycle_B = criterion_cycle(rec_B, real_B) * self.lambda_cycle

        loss_G = (
            loss_idt_A
            + loss_idt_B
            + loss_gan_A2B
            + loss_gan_B2A
            + loss_cycle_A
            + loss_cycle_B
        )
        loss_G.backward()
        self.bundle.optimizer_G.step()

        self.bundle.optimizer_D.zero_grad()
        pred_real_A = self.bundle.netD_B(real_A)
        real_A_label = torch.ones_like(pred_real_A)
        fake_A_label = torch.zeros_like(pred_real_A)
        loss_D_A_real = criterion_gan(pred_real_A, real_A_label)
        pred_fake_A = self.bundle.netD_B(fake_A.detach())
        loss_D_A_fake = criterion_gan(pred_fake_A, fake_A_label)
        loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5

        pred_real_B = self.bundle.netD_A(real_B)
        real_B_label = torch.ones_like(pred_real_B)
        fake_B_label = torch.zeros_like(pred_real_B)
        loss_D_B_real = criterion_gan(pred_real_B, real_B_label)
        pred_fake_B = self.bundle.netD_A(fake_B.detach())
        loss_D_B_fake = criterion_gan(pred_fake_B, fake_B_label)
        loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5

        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        self.bundle.optimizer_D.step()

        loss_G_A2B = loss_gan_A2B + loss_cycle_B + loss_idt_B
        loss_G_B2A = loss_gan_B2A + loss_cycle_A + loss_idt_A
        return {
            "loss_G": float(loss_G.detach().cpu().item()),
            "loss_D": float(loss_D.detach().cpu().item()),
            "loss_G_A2B": float(loss_G_A2B.detach().cpu().item()),
            "loss_G_B2A": float(loss_G_B2A.detach().cpu().item()),
            "loss_D_A2B": float(loss_D_B.detach().cpu().item()),
            "loss_D_B2A": float(loss_D_A.detach().cpu().item()),
            "loss_cycle_A": float(loss_cycle_A.detach().cpu().item()),
            "loss_cycle_B": float(loss_cycle_B.detach().cpu().item()),
            "loss_idt_A": float(loss_idt_A.detach().cpu().item()),
            "loss_idt_B": float(loss_idt_B.detach().cpu().item()),
        }

    def infer_AtoB(self, source: np.ndarray) -> np.ndarray:
        torch, _, _ = require_torch()
        self.bundle.netG_A.eval()
        with torch.no_grad():
            real_A = _to_3channel_tensor(torch, source, self.bundle.device)
            fake_B = self.bundle.netG_A(real_A)
            return fake_B.squeeze(0).detach().cpu().numpy()

    def infer_BtoA(self, source: np.ndarray) -> np.ndarray:
        torch, _, _ = require_torch()
        self.bundle.netG_B.eval()
        with torch.no_grad():
            real_B = _to_3channel_tensor(torch, source, self.bundle.device)
            fake_A = self.bundle.netG_B(real_B)
            return fake_A.squeeze(0).detach().cpu().numpy()

    def save(self, checkpoint_dir: Path, tag: str) -> None:
        self.bundle.save_checkpoint(checkpoint_dir, tag)
