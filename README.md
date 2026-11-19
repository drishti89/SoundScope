# SoundScope: Audio Classification and Noise Robustness Analysis

## Overview

SoundScope is an audio classification project that explores the use of pretrained Audio Spectrogram Transformer (AST) representations for environmental sound recognition. It combines transfer learning with traditional machine learning and investigates how background noise affects classification performance.

The project uses frozen AST embeddings as audio features and a Logistic Regression classifier to distinguish between 50 environmental sound categories. It also evaluates model performance under controlled Gaussian noise at different Signal-to-Noise Ratio (SNR) levels.

## Objectives

- Extract meaningful audio representations using a pretrained AST model.
- Classify environmental sounds across 50 categories.
- Evaluate classification performance using five-fold cross-validation.
- Investigate the effect of increasing Gaussian noise on classification accuracy.

## Dataset

**ESC-50 Environmental Sound Classification Dataset**

- 2,000 environmental audio recordings.
- 50 sound categories.
- 5 official folds.
- 40 recordings per category.

Dataset: https://github.com/karolpiczak/ESC-50

## Methodology

### 1. Audio Preprocessing
- Loaded audio recordings using Librosa.
- Converted recordings to mono and resampled them to 16 kHz.
- Processed audio using the AST feature processor.

### 2. Feature Extraction
- Used the pretrained `MIT/ast-finetuned-audioset-10-10-0.4593` checkpoint.
- Kept the AST model frozen during feature extraction.
- Extracted 768-dimensional audio embeddings through mean pooling of transformer patch-token representations.

### 3. Sound Classification
- Used Logistic Regression as the classification model.
- Applied StandardScaler within a Scikit-learn pipeline.
- Evaluated performance using the five official ESC-50 folds, training on four folds and testing on the remaining fold.

### 4. Noise Robustness Analysis
- Introduced controlled Gaussian noise into the audio recordings.
- Evaluated classification performance at 20 dB, 10 dB, 5 dB and 0 dB SNR.
- Trained classifiers using clean training embeddings and evaluated them on noisy test embeddings.
- Compared performance across clean and noisy audio conditions.

## Results

### Clean Audio Classification

| Metric | Result |
|---|---:|
| Mean five-fold accuracy | 95.40% |
| Overall macro F1-score | 0.95 |
| Number of sound categories | 50 |

### Noise Robustness

| Audio Condition | Mean Accuracy |
|---|---:|
| Clean | 95.40% |
| 20 dB SNR | 91.90% |
| 10 dB SNR | 81.45% |
| 5 dB SNR | 68.65% |
| 0 dB SNR | 45.20% |

The results show a progressive decline in classification accuracy as noise intensity increases, highlighting the importance of evaluating audio models under varying acoustic conditions.

## Repository Structure

```text
SoundScope/
├── notebooks/
│   └── SoundScope.ipynb
├── results/
│   ├── classification_report.csv
│   ├── confusion_matrix.png
│   ├── robustness_metrics.csv
│   ├── robustness_summary.csv
│   └── robustness_plot.png
├── src/
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Librosa
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Google Colab

## Running the Project

1. Clone this repository.
2. Open `notebooks/SoundScope.ipynb` in Google Colab.
3. Download the ESC-50 dataset using the dataset instructions in the notebook.
4. Install the required dependencies.
5. Run the notebook cells sequentially.

A GPU-enabled runtime is recommended for faster AST feature extraction.

## Acknowledgements

- ESC-50 dataset by Karol J. Piczak.
- Audio Spectrogram Transformer checkpoint: `MIT/ast-finetuned-audioset-10-10-0.4593`.

## License

This project is licensed under the MIT License.
