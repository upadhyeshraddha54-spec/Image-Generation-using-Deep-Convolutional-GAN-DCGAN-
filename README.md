
# Image Generation using Deep Convolutional GAN (DCGAN)

## Overview
This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate synthetic handwritten digit images using PyTorch and the Scikit-learn Digits dataset.

## Objectives
- Implement a DCGAN using PyTorch.
- Preprocess handwritten digit images.
- Train the Generator and Discriminator.
- Generate synthetic digit images.
- Visualize training progress and generated outputs.
- Perform latent-space interpolation.

## Dataset
- **Dataset:** Scikit-learn Digits dataset
- **Total Images:** 1,797
- **Classes:** Digits 0–9
- **Original Image Size:** 8 × 8 pixels
- **Resized Image Size:** 16 × 16 pixels
- **Image Type:** Grayscale
- **Normalization:** Pixel values scaled to [-1, 1]

## Model Architecture

### Generator
The Generator takes a random latent vector of size 64 and converts it into a 16 × 16 grayscale image using transposed convolution layers, batch normalization, and ReLU activation. The output layer uses Tanh activation.

### Discriminator
The Discriminator uses convolutional layers to classify images as real or generated. It outputs a score used during adversarial training.

## Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 16 × 16 |
| Batch Size | 64 |
| Epochs | 30 |
| Latent Dimension | 64 |
| Generator Learning Rate | 0.0002 |
| Discriminator Learning Rate | 0.00005 |
| Optimizer | Adam |
| Loss Function | BCEWithLogitsLoss |
| Random Seed | 42 |

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Jupyter Notebook

## Installation

Install the required libraries:

```bash
pip install torch numpy matplotlib pillow scikit-learn jupyter
```

## How to Run

1. Clone or download the repository.
2. Open the project directory.
3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `DCGAN.ipynb`.
5. Run all cells sequentially.

The notebook automatically uses a CUDA GPU if available; otherwise, it uses the CPU.

## Outputs

The notebook generates visualizations and model files, including:

- `training_progress.png` — generated images during training.
- `final_generated_digits.png` — final generated samples.
- `latent_interpolation.png` — images generated between two latent vectors.
- `dcgan_digits_generator.pth` — saved Generator model parameters.
- `dcgan_digits_discriminator.pth` — saved Discriminator model parameters.

## Evaluation

The model is evaluated using Generator and Discriminator losses, Discriminator accuracy, and visual inspection of generated images. The notebook also compares real and generated samples and demonstrates latent-space interpolation.

Generated digits may vary in quality because of the small dataset and low image resolution.

## Conclusion

This project demonstrates image generation using DCGAN, including data preprocessing, neural network design, adversarial training, visualization, evaluation, and model saving.

## Author
Shraddha Upadhye
