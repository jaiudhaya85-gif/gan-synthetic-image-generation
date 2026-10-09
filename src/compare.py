
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = [
    "baseline", "low_lr", "high_lr", "latent_50", "batch_128"
]

rows = []

for name in EXPERIMENTS:
    file = ROOT / "results" / name / "history.csv"

    if not file.exists():
        print(f"Skipping {name}: history.csv not found")
        continue

    with open(file, newline="") as f:
        history = list(csv.DictReader(f))

    if not history:
        continue

    g = np.array([float(r["generator_loss"]) for r in history])
    d = np.array([float(r["discriminator_loss"]) for r in history])

    rows.append({
        "experiment": name,
        "epochs": len(history),
        "final_generator_loss": round(float(g[-1]), 4),
        "final_discriminator_loss": round(float(d[-1]), 4),
        "generator_loss_std": round(float(np.std(g)), 4),
        "discriminator_loss_std": round(float(np.std(d)), 4),
    })

if rows:
    out = ROOT / "results"
    with open(out / "comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nEXPERIMENT COMPARISON")
    for row in rows:
        print(row)

    names = [r["experiment"] for r in rows]
    g_loss = [r["final_generator_loss"] for r in rows]
    d_loss = [r["final_discriminator_loss"] for r in rows]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, g_loss, width, label="Generator")
    plt.bar(x + width / 2, d_loss, width, label="Discriminator")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Final epoch loss")
    plt.title("GAN Training Experiment Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "comparison.png", dpi=150)
    plt.close()

    print("\nSaved results/comparison.csv and results/comparison.png")
else:
    print("No experiment results found. Run training first.")