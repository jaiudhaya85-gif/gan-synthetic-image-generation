# GAN-Based Synthetic Image Generation

## 1. Project Overview

This project implements a Generative Adversarial Network (GAN) to generate synthetic handwritten digit images. It investigates how different training settings affect image diversity, image quality, and training stability.

A GAN consists of two neural networks: a Generator that creates synthetic images and a Discriminator that distinguishes generated images from real training images. These networks compete during training, helping the Generator learn patterns similar to those in the training dataset.

## 2. Problem Statement

Generating realistic synthetic images using deep learning requires appropriate model architecture and training settings. Different learning rates, latent dimensions, and batch sizes can influence the training process and the generated results.

This project compares five GAN configurations to investigate the influence of these parameters on training losses, image diversity, and visual quality.

## 3. Objectives

* Implement a GAN for synthetic image generation.
* Train a baseline model.
* Experiment with different learning rates, latent dimensions, and batch sizes.
* Compare Generator and Discriminator losses.
* Estimate image diversity using pixel-level distance.
* Evaluate generated images visually.
* Analyze training behavior and identify possible improvements.

## 4. Technologies Used

* **Programming language:** Python
* **Deep learning framework:** PyTorch
* **Numerical computation:** NumPy
* **Visualization:** Matplotlib
* **Data analysis:** CSV and Python
* **Development environment:** Visual Studio Code

## 5. GAN Architecture

### Generator

The Generator receives a random latent vector as input and transforms it into a synthetic image.

### Discriminator

The Discriminator receives real and generated images and learns to distinguish between them.

### Training Process

1. Load real training images.
2. Generate random latent vectors.
3. Produce synthetic images using the Generator.
4. Train the Discriminator to distinguish real images from generated images.
5. Train the Generator to produce images that better fool the Discriminator.
6. Repeat the process for the specified number of epochs.
7. Save generated images, training history, and loss graphs.

## 6. Experimental Configurations

Five configurations were evaluated, each for three epochs.

| Experiment          | Learning Rate | Latent Dimension | Batch Size |
| ------------------- | ------------: | ---------------: | ---------: |
| Baseline            |        0.0002 |              100 |         64 |
| Low learning rate   |       0.00005 |              100 |         64 |
| High learning rate  |         0.001 |              100 |         64 |
| Latent dimension 50 |        0.0002 |               50 |         64 |
| Batch size 128      |        0.0002 |              100 |        128 |

The baseline values shown above reflect the settings used for the comparison commands. Confirm them against the project's actual training configuration before submission.

## 7. Experimental Results

### Training Loss Comparison

| Experiment          | Generator Loss | Discriminator Loss | Generator Loss Standard Deviation |
| ------------------- | -------------: | -----------------: | --------------------------------: |
| Baseline            |         0.9453 |             1.1696 |                            0.0741 |
| Low learning rate   |         0.7927 |             1.1564 |                            0.0729 |
| High learning rate  |         0.9166 |             1.2841 |                            0.0620 |
| Latent dimension 50 |         0.8994 |             1.2075 |                            0.0611 |
| Batch size 128      |         0.8467 |             1.2057 |                            0.0500 |

### Image Diversity Comparison

Image diversity was estimated using the mean absolute pixel distance between pairs of 64 generated images. A fixed random seed was used for reproducibility.

| Experiment          | Diversity Score |
| ------------------- | --------------: |
| Baseline            |         0.33228 |
| Low learning rate   |         0.35291 |
| High learning rate  |         0.33000 |
| Latent dimension 50 |         0.34477 |
| Batch size 128      |         0.34296 |

### Results Interpretation

* **Baseline:** Provides the reference for comparing other configurations.
* **Low learning rate:** Achieved the lowest final Generator loss and the highest measured pixel-level diversity.
* **High learning rate:** Produced the lowest measured pixel-level diversity and the highest final Discriminator loss among the five configurations.
* **Latent dimension 50:** Produced greater pixel-level variation than the baseline.
* **Batch size 128:** Had the lowest recorded Generator loss standard deviation, indicating less variation in the recorded Generator losses over these epochs.

These measurements do not independently establish image realism or overall GAN stability. Visual inspection of generated images and loss curves is also necessary.

## 8. Evaluation Metrics

### Generator Loss

Measures the Generator's objective during adversarial training. Its value must be interpreted alongside Discriminator behavior and generated images.

### Discriminator Loss

Measures the Discriminator's training objective. GAN losses are competitive, so lower loss is not automatically evidence of better image generation.

### Image Diversity

The project calculates average pairwise absolute pixel distance between generated images. Higher values indicate greater pixel-level variation, but random noise can also produce high scores.

### Training Stability

Loss curves and the standard deviation of recorded Generator and Discriminator losses are used as preliminary indicators of training behavior.

### Visual Quality

Generated image grids should be inspected for recognizable digit shapes, repeated patterns, and obvious visual artifacts.

## 9. Project Structure

The project uses the following main files and directories. Additional files may be present in the local implementation.

```text
gan-synthetic-image-generation/
├── src/
│   ├── train.py
│   ├── compare.py
│   ├── diversity.py
│   └── models.py
├── results/
│   ├── baseline/
│   ├── low_lr/
│   ├── high_lr/
│   ├── latent_50/
│   ├── batch_128/
│   ├── comparison.csv
│   ├── comparison.png
│   └── diversity.csv
├── report/
│   └── project_report.md
├── requirements.txt
└── README.md
```

## 10. Installation and Setup

### Prerequisites

* Python installed on your system
* Visual Studio Code
* Required Python packages listed in `requirements.txt`

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd gan-synthetic-image-generation
```

Replace the placeholder with your actual GitHub repository URL.

### Step 2: Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use the virtual environment's Python executable directly.

### Step 3: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Baseline Experiment

```bash
python src/train.py --experiment baseline --epochs 3 --batch-size 64 --lr 0.0002
```

### Step 5: Run the Remaining Experiments

Low learning rate:

```bash
python src/train.py --experiment low_lr --epochs 3 --batch-size 64 --lr 0.00005
```

High learning rate:

```bash
python src/train.py --experiment high_lr --epochs 3 --batch-size 64 --lr 0.001
```

Latent dimension 50:

```bash
python src/train.py --experiment latent_50 --epochs 3 --batch-size 64 --lr 0.0002 --latent-dim 50
```

Batch size 128:

```bash
python src/train.py --experiment batch_128 --epochs 3 --batch-size 128 --lr 0.0002
```

### Step 6: Compare Training Results

```bash
python src/compare.py
```

This creates the training comparison CSV and graph.

### Step 7: Calculate Image Diversity

```bash
python src/diversity.py
```

This calculates diversity scores and saves them to `results/diversity.csv`.

**Note:** These commands assume the argument names, model configuration, and dependencies match the current implementation. Check `python src/train.py --help` if a command differs from your local version.

## 11. Output Files

| File                       | Purpose                            |
| -------------------------- | ---------------------------------- |
| `results/comparison.csv`   | Training loss comparison           |
| `results/comparison.png`   | Training loss comparison graph     |
| `results/diversity.csv`    | Pixel-level diversity measurements |
| `results/<experiment>/`    | Experiment-specific outputs        |
| `report/project_report.md` | Project report                     |

The experiment directories may contain generated image grids, loss curves, training history, and model checkpoints depending on the implementation.

## 12. Conclusion

This project implements a GAN for synthetic handwritten image generation and compares five training configurations. The experiments demonstrate differences in Generator and Discriminator losses and pixel-level image diversity.

The low-learning-rate configuration achieved the highest measured pixel-level diversity score, while the batch-size-128 configuration had the lowest recorded Generator loss standard deviation. These findings are preliminary because each experiment used only three epochs.

The overall best configuration should be selected after visually comparing the generated images and considering image quality, diversity, and training behavior together.

## 13. Future Improvements

* Train for more epochs.
* Generate larger image samples for evaluation.
* Improve the diversity metric with feature-based measurements.
* Investigate mode collapse and repeated generated images.
* Evaluate image quality using suitable metrics such as the Fréchet Inception Distance when appropriate.
* Compare additional optimizers and model architectures.
* Include generated image grids and experimental graphs in the final report.

## 14. Author

**Name:** Jai Kathick AP

**Project:** GAN-Based Synthetic Image Generation

**Purpose:** Academic project on generative deep learning essential.

## 15. License

Add an appropriate license if this project will be publicly distributed. Ensure that the dataset and any third-party code are used according to their respective licenses.
