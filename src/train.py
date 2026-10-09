
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import Generator, Discriminator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--latent-dim", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Experiment:", args.experiment)

    root = Path(__file__).resolve().parent.parent
    output = root / "results" / args.experiment
    output.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.MNIST(
        root=root / "data",
        train=True,
        download=True,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    G = Generator(args.latent_dim).to(device)
    D = Discriminator().to(device)

    loss_function = nn.BCELoss()

    optimizer_G = torch.optim.Adam(
        G.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )
    optimizer_D = torch.optim.Adam(
        D.parameters(), lr=args.lr, betas=(0.5, 0.999)
    )

    fixed_noise = torch.randn(16, args.latent_dim, device=device)
    history = []

    for epoch in range(1, args.epochs + 1):
        g_losses, d_losses = [], []

        for real_images, _ in loader:
            real_images = real_images.to(device)
            batch = real_images.size(0)

            real_labels = torch.ones(batch, 1, device=device)
            fake_labels = torch.zeros(batch, 1, device=device)

            # Train Discriminator
            optimizer_D.zero_grad()

            real_loss = loss_function(
                D(real_images), real_labels
            )

            noise = torch.randn(batch, args.latent_dim, device=device)
            fake_images = G(noise)

            fake_loss = loss_function(
                D(fake_images.detach()), fake_labels
            )

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()

            noise = torch.randn(batch, args.latent_dim, device=device)
            generated = G(noise)

            g_loss = loss_function(
                D(generated), real_labels
            )

            g_loss.backward()
            optimizer_G.step()

            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        g_avg = float(np.mean(g_losses))
        d_avg = float(np.mean(d_losses))

        history.append({
            "epoch": epoch,
            "generator_loss": g_avg,
            "discriminator_loss": d_avg
        })

        # Save generated images
        G.eval()
        with torch.no_grad():
            samples = G(fixed_noise).cpu()

        fig, axes = plt.subplots(4, 4, figsize=(5, 5))
        for i, ax in enumerate(axes.flat):
            ax.imshow((samples[i, 0] + 1) / 2, cmap="gray")
            ax.axis("off")

        fig.suptitle(f"{args.experiment} - Epoch {epoch}")
        fig.tight_layout()
        fig.savefig(output / f"epoch_{epoch}.png")
        plt.close(fig)
        G.train()

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"G Loss: {g_avg:.4f} | D Loss: {d_avg:.4f}"
        )

    # Save model weights
    torch.save(G.state_dict(), output / "generator.pth")
    torch.save(D.state_dict(), output / "discriminator.pth")

    # Save training history
    with open(output / "history.csv", "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "generator_loss", "discriminator_loss"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Plot training losses
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [r["generator_loss"] for r in history],
             marker="o", label="Generator")
    plt.plot(epochs, [r["discriminator_loss"] for r in history],
             marker="o", label="Discriminator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"GAN Training Stability: {args.experiment}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output / "loss_curve.png")
    plt.close()

    print("Experiment completed!")
    print("Results saved to:", output)


if __name__ == "__main__":
    main()