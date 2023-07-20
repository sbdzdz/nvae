import argparse
import logging
import os
import pathlib

import numpy as np
import torch
import torchvision
import wandb
from torch.utils.data import DataLoader
from torchvision import transforms

from nvae.dataset import ImageFolderDataset
from nvae.utils import add_sn
from nvae.nvae import NVAE


class WarmupKLLoss:
    def __init__(
        self, init_weights, steps, M_N=0.005, eta_M_N=1e-5, M_N_decay_step=3000
    ):
        self.init_weights = init_weights
        self.M_N = M_N
        self.eta_M_N = eta_M_N
        self.M_N_decay_step = M_N_decay_step
        self.speeds = [(1.0 - w) / s for w, s in zip(init_weights, steps)]
        self.steps = np.cumsum(steps)
        self.stage = 0
        self._ready_start_step = 0
        self._ready_for_M_N = False
        self._M_N_decay_speed = (self.M_N - self.eta_M_N) / self.M_N_decay_step

    def _get_stage(self, step):
        while not self.stage > len(self.steps) - 1:
            if step <= self.steps[self.stage]:
                return self.stage
            else:
                self.stage += 1

        return self.stage

    def get_loss(self, step, losses):
        loss = 0.0
        stage = self._get_stage(step)

        for i, l in enumerate(losses):
            # Update weights
            if i == stage:
                speed = self.speeds[stage]
                t = step if stage == 0 else step - self.steps[stage - 1]
                w = min(self.init_weights[i] + speed * t, 1.0)
            elif i < stage:
                w = 1.0
            else:
                w = self.init_weights[i]

            if self._ready_for_M_N == False and i == len(losses) - 1 and w == 1.0:
                self._ready_for_M_N = True
                self._ready_start_step = step
            l = losses[i] * w
            loss += l

        if self._ready_for_M_N:
            M_N = max(
                self.M_N - self._M_N_decay_speed * (step - self._ready_start_step),
                self.eta_M_N,
            )
        else:
            M_N = self.M_N

        return M_N * loss


def main(args):
    """Main training routine."""
    wandb.init(project="nvae", dir=args.wandb_dir)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.dataset == "cifar":
        train_dataset = torchvision.datasets.CIFAR10(
            root=pathlib.Path(os.environ["WORK"]) / "datasets/cifar",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
    else:
        train_dataset = ImageFolderDataset(args.dataset, img_dim=64)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = NVAE(z_dim=512, img_dim=(64, 64))
    model.apply(add_sn)  # spectral normalization
    model.to(device)

    warmup_kl = WarmupKLLoss(
        init_weights=[1.0, 1.0 / 2, 1.0 / 8],
        steps=[4500, 3000, 1500],
        M_N=args.batch_size / len(train_dataloader.dataset),
        eta_M_N=5e-6,
        M_N_decay_step=36000,
    )
    print("M_N=", warmup_kl.M_N, "ETA_M_N=", warmup_kl.eta_M_N)

    optimizer = torch.optim.Adamax(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=15, eta_min=1e-4
    )

    step = 0
    for _ in range(args.epochs):
        model.train()

        for image in train_dataloader:
            optimizer.zero_grad()
            image = image.to(device)
            _, recon_loss, kl_losses = model(image)
            kl_loss = warmup_kl.get_loss(step, kl_losses)
            loss = recon_loss + kl_loss
            wandb.log(
                {
                    "recon_loss": recon_loss.item(),
                    "kl_loss": kl_loss.item(),
                    "loss": loss.item(),
                }
            )
            loss.backward()
            optimizer.step()
            step += 1

            if step != 0 and step % 100 == 0:
                with torch.no_grad():
                    z = torch.randn((1, 512, 2, 2)).to(device)
                    gen_img, _ = model.decoder(z)
                    gen_img = gen_img.permute(0, 2, 3, 1)
                    gen_img = gen_img[0].cpu().numpy() * 255
                    gen_img = gen_img.astype(np.uint8)
                    wandb.log({"samples_train": wandb.Image(gen_img)})
        scheduler.step()

        model.eval()
        with torch.no_grad():
            z = torch.randn((1, 512, 2, 2)).to(device)
            gen_img, _ = model.decoder(z)
            gen_img = gen_img.permute(0, 2, 3, 1)
            gen_img = gen_img[0].cpu().numpy() * 255
            gen_img = gen_img.astype(np.uint8)
            wandb.log({"samples": wandb.Image(gen_img)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trainer for state AutoEncoder model.")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs.")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="size of each sample batch"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name (cifar) or path"
    )
    parser.add_argument(
        "--wandb_dir", type=str, default="wandb", help="wandb directory"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="number of cpu workers to use during batch generation",
    )
    args = parser.parse_args()
    main(args)
