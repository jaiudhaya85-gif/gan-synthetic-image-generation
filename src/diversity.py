
import csv
from pathlib import Path

import numpy as np
import torch

from models import Generator

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENTS = {
    "baseline": 100,
    "low_lr": 100,
    "high_lr": 100,
    "latent_50": 50,
    "batch_128": 100,
}

results = []

for name, latent_dim in EXPERIMENTS.items():
    folder = ROOT / "results" / name
    checkpoint = folder / "generator.pth"

    if not checkpoint.exists():
        print(f"Skipping {name}: checkpoint not found")
        continue

    model = Generator(latent_dim).to(DEVICE)
    model.load_state_dict(
        torch.load(checkpoint, map_location=DEVICE, weights_only=True)
    )
    model.eval()

    # Use the same random seed for a reproducible comparison.
    generator = torch.Generator(device=DEVICE).manual_seed(123)

    with torch.no_grad():
        noise = torch.randn(
            64, latent_dim, generator=generator, device=DEVICE
        )
        images = model(noise).cpu().view(64, -1).numpy()

    # Mean absolute pixel distance across pairs of generated images.
    distances = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            distances.append(np.mean(np.abs(images[i] - images[j])))

    score = float(np.mean(distances))

    results.append({
        "experiment": name,
        "diversity_score": round(score, 5),
    })

    print(f"{name}: diversity score = {score:.5f}")

with open(ROOT / "results" / "diversity.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["experiment", "diversity_score"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved results/diversity.csv")
print("Higher scores indicate more pixel-level variation, not necessarily better images.")