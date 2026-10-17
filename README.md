# Currency Authentication Detection Using Vision Transformer (ViT)

An AI-based project for classifying Indian currency note images as **REAL** or **FAKE** using a Vision Transformer (ViT) deep learning model.

## Features

- Image preprocessing and resizing
- Vision Transformer architecture with patch embeddings and self-attention
- Data augmentation to improve generalization
- Training with validation and early stopping
- Accuracy curves and confusion matrix
- REAL/FAKE prediction for currency images

## Technologies Used

- Python
- TensorFlow / Keras
- Vision Transformer (ViT)
- NumPy and Matplotlib
- Pillow
- Scikit-learn

## Project Structure

```text
CURRENCY-AUTHENTICATION-DETECTION/
├── mainn.py
├── requirements.txt
├── README.md
├── .gitignore
└── outputs/
```

## Dataset

The project uses the Indian Currency Real vs Fake Notes dataset. Download the dataset separately and keep it outside the Git repository.

## Setup

Create and activate a Python virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training

Run the training script with the path to your dataset ZIP:

```bash
python mainn.py --dataset "$HOME/Downloads/archive (1).zip" --epochs 50
```

Change the ZIP filename to match your downloaded dataset.

## Evaluation

The project reports measured test accuracy, a classification report, and a confusion matrix. Accuracy depends on the dataset and training results; no particular accuracy is guaranteed.

## Disclaimer

This is an academic machine learning project. Predictions should not be treated as definitive proof that a banknote is genuine or counterfeit.



