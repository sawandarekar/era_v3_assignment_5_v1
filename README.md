# ML Model Training Pipeline

[![ML Pipeline](https://github.com/sawandarekar/era_v3_assignment_5_v1/actions/workflows/ml-pipeline.yml/badge.svg)](https://github.com/sawandarekar/era_v3_assignment_5_v1/actions/workflows/ml-pipeline.yml)

This project implements a complete CI/CD pipeline for training and testing a CNN model on the MNIST dataset. The pipeline includes model training, testing, and automated validation through GitHub Actions.

## Project Structure        

```
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml
├── model.py # CNN model architecture
├── train.py # Training script
├── test_model.py # Test cases
├── .gitignore
└── README.md
```

## Model Architecture

The project uses a lightweight CNN architecture designed for MNIST digit classification:
- 2 convolutional layers (8 and 16 filters)
- 2 max pooling layers
- 2 fully connected layers (64 neurons and 10 output classes)
- Total parameters: ~22,138

## Key Features

- Automated training and testing pipeline
- Dataset size limited to 25,000 samples for faster training
- Model parameter count kept under 25,000 for efficiency
- Automated accuracy testing (>80% required)
- Model artifacts saved with timestamps

## Setup and Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate    # On Windows: venv\Scripts\activate
```


3. Install dependencies:
```bash
pip install torch torchvision pytest
```


## Local Development

1. Train the model:
```bash
python train.py
```
This will:
- Load 25,000 random samples from MNIST
- Train the CNN model
- Save the model with timestamp and device info

2. Run tests:
```bash
python -m pytest test_model.py -v
```
Tests verify:
- Model architecture (input/output shapes)
- Parameter count (<25,000)
- Model accuracy (>80% on test set)

## CI/CD Pipeline

The GitHub Actions workflow (`ml-pipeline.yml`) automatically:
1. Sets up Python environment
2. Installs dependencies
3. Trains model
4. Runs tests
5. Saves model artifact

The pipeline runs on every push to the repository.

## Model Artifacts

Trained models are saved with the naming convention:
```
model_<timestamp>_<device>.pth
```

These files are:
- Ignored by git (.gitignore)
- Uploaded as artifacts in GitHub Actions
- Used by test cases to verify model performance

## Testing

The test suite (`test_model.py`) includes:
- Architecture tests
  - Verifies input/output dimensions
  - Checks parameter count (<25,000)
- Performance tests
  - Loads latest trained model
  - Verifies accuracy >80% on MNIST test set

## Notes

- Training uses CPU by default (GPU if available)
- Model architecture is optimized for size while maintaining accuracy
- Test dataset is downloaded automatically when needed
- All trained models and datasets are excluded from git tracking

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest
