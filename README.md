
# Activity Recognition with LSTMs

This project implements **Human Activity Recognition (HAR)** using the UCI HAR Smartphone Dataset and a **Long Short-Term Memory (LSTM)** Recurrent Neural Network (RNN). The model classifies human movements into six categories:

- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## Table of Contents
- [Dataset Overview](#dataset-overview)
- [What is an LSTM?](#what-is-an-lstm)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Key Insights](#key-insights)
- [Improvements](#improvements)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Dataset Overview

The [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones) contains sensor data from smartphones (accelerometer and gyroscope) worn by participants.

Key details:
- **Sampling**: 2.56-second windows (128 readings) with 50% overlap
- **Features**: 9 input signals (body acceleration, gyroscope, total acceleration in x/y/z axes)
- **Training Data**: 7,352 sequences
- **Testing Data**: 2,947 sequences

## What is an LSTM?

An **LSTM** is an advanced RNN that models sequential data while addressing the vanishing gradient problem. This project uses a **many-to-one** architecture processing 128 timesteps × 9 features.

## Model Architecture

```python
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(128, 9)),
    LSTM(32),
    Dense(6, activation='softmax')
])
```

- **Optimizer**: Adam (lr=0.0025)
- **Loss**: Softmax cross-entropy with L2 regularization
- **Training**: 300 epochs, batch_size=1500

## Results

**Test Accuracy**: 91.65%  
**Confusion Matrix**:

| True \ Predicted | WALKING | UPSTAIRS | DOWNSTAIRS | SITTING | STANDING | LAYING |
|------------------|---------|----------|------------|---------|----------|--------|
| **WALKING**      | 15.81   | 0.07     | 0.88       | 0.00    | 0.07     | 0.00   |
| **UPSTAIRS**     | 0.17    | 14.96    | 0.85       | 0.00    | 0.00     | 0.00   |
| **DOWNSTAIRS**   | 0.03    | 0.00     | 14.22      | 0.00    | 0.00     | 0.00   |
| **SITTING**      | 0.03    | 0.03     | 0.00       | 13.44   | 2.95     | 0.20   |
| **STANDING**     | 0.07    | 0.03     | 0.00       | 2.95    | 15.00    | 0.00   |
| **LAYING**       | 0.00    | 0.00     | 0.00       | 0.00    | 0.00     | 18.22  |

## Setup and Usage

### Prerequisites
```bash
Python 3.x
TensorFlow 1.0.0
NumPy, Matplotlib, Scikit-learn
```

### Installation
```bash
git clone https://github.com/your-username/activity-recognition.git
cd activity-recognition
cd data
python download_dataset.py
cd ..
```

### Running
```bash
jupyter notebook LSTM.ipynb
# or
python lstm.py
```

## Key Insights
- Achieves 91% accuracy with minimal preprocessing
- Gyroscope data improves accuracy by ~4%
- Sitting/standing are most confused (2.95% overlap)

## Improvements
Advanced version with bidirectional LSTMs achieves [94% accuracy](https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs).

## Citation
```bibtex
@misc{chevalier2016lstms,
  title={LSTMs for human activity recognition},
  author={Chevalier, Guillaume},
  year={2016}
}
```


## Acknowledgments
- Inspired by [aymericdamien's RNN work](https://github.com/aymericdamien)
- Featured in [Awesome TensorFlow](https://github.com/jtoy/awesome-tensorflow)
```

To use:
1. Copy this text into a file named `README.md`
2. Update the GitHub URL with your actual username/repo
3. Ensure the images are in the correct path (or update paths)
4. Verify the LICENSE file exists in your repo
