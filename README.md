# Deepfake Voice Detection & Audio Forensics Voice Guard

Voice Guard is a deep-learning-based project designed to distinguish genuine human speech from AI-generated or spoofed speech. It combines audio preprocessing, Mel-spectrogram feature extraction, deep-learning models, and a Flask-based application for audio analysis.

## Project Objectives

- Detect genuine (REAL) and AI-generated or spoofed (FAKE) speech.
- Support Kannada, Hindi, Tamil, and Telugu.
- Convert audio recordings into Mel-spectrogram representations.
- Train and compare ResNet18 and BiLSTM models.
- Evaluate model performance using classification metrics.
- Provide an application for uploading and analysing WAV recordings.

## Supported Languages

The Indian-language dataset includes:

- Kannada
- Hindi
- Tamil
- Telugu

## Dataset Description

The Indian-language dataset contains **7,403 audio samples**:

| Class | Samples |
|---|---:|
| Real | 3,395 |
| Fake | 4,008 |
| **Total** | **7,403** |

The dataset is divided into:

| Split | Samples |
|---|---:|
| Training | 5,214 |
| Validation | 1,120 |
| Testing | 1,069 |
| **Total** | **7,403** |

The project uses real speech from IndicVoices and synthetic speech from IndicSynth.

ASVspoof datasets are maintained separately for independent evaluation and are not mixed into the Indian-language training dataset.

> Dataset files, archives, generated features, and trained model checkpoints are excluded from this repository because of their size and distribution considerations.

## System Workflow

1. Collect real and synthetic audio recordings.
2. Clean and validate the audio.
3. Standardize audio to 16 kHz and mono.
4. Prepare training, validation, and test splits.
5. Generate Mel-spectrograms.
6. Train the ResNet18 and BiLSTM models.
7. Evaluate both models on the held-out test set.
8. Compare model performance.
9. Analyse WAV recordings through the Voice Guard application.

## Deep Learning Models

### ResNet18

ResNet18 is used as the primary image-based classifier. It learns patterns from Mel-spectrogram representations to classify audio as REAL or FAKE.

### BiLSTM

The Bidirectional Long Short-Term Memory (BiLSTM) model processes sequential features extracted from Mel-spectrograms. It is implemented as a second model for comparison.

Both models use the same 1,069-sample test set for the reported comparison.

## Model Evaluation Results

The following results were obtained on the held-out Indian-language test split:

| Metric | ResNet18 | BiLSTM |
|---|---:|---:|
| Accuracy | 99.72% | 98.13% |
| Fake Precision | 99.51% | 98.03% |
| Fake Recall | 100.00% | 98.68% |
| Fake F1-score | 99.75% | 98.36% |
| ROC-AUC | 1.0000 | 0.9965 |
| Inference time/sample | 136.47 ms | 37.34 ms |
| Model size | 128 MB | 6.82 MB |

ResNet18 achieved higher classification performance on this test split. BiLSTM was smaller and had lower per-sample inference time.

These are experimental results on the project's held-out test data. Further evaluation on additional independent datasets and real-world recordings is required before making broader performance claims.

## Voice Guard Application

The project includes a Flask-based application for analysing uploaded WAV recordings.

Application features include:

- WAV audio upload and analysis.
- REAL or FAKE classification.
- Fake-voice probability scoring.
- Segment-level analysis using five-second audio segments.
- Mel-spectrogram visualization.
- Local analysis history using SQLite.

The application loads the trained ResNet18 checkpoint for inference.

## Repository Structure

```text

deepfake-voice-detection/
│
├── app/
│   ├── static/
│   ├── templates/
│   ├── tests/
│   ├── voiceguard/
│   ├── requirements.txt
│   └── run.py
│
├── preprocessing/
│   ├── clean_audio.py
│   ├── create_indian_split.py
│   ├── create_melspectrograms.py
│   ├── generate_mels.py
│   └── verification scripts
│
├── training/
│   ├── dataset_loader.py
│   ├── train_config.py
│   ├── resnet18_model.py
│   ├── lstm_model.py
│   ├── train_resnet18.py
│   ├── train_lstm.py
│   ├── evaluate_resnet18.py
│   └── evaluate_lstm.py
│
├── results/
│   ├── comparison_table.csv
│   ├── compare_models.py
│   ├── confusion matrices
│   ├── comparison charts
│   └── evaluation reports
│
├── dataset/       # Local only; not tracked
├── features/      # Local only; not tracked
├── README.md
└── .gitignore
```

## Environment Setup

Clone the repository:

```powershell
git clone https://github.com/disha2005-stack/deepfake-voice-detection.git
cd deepfake-voice-detection
```

Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Install application dependencies:

```powershell
pip install -r app\requirements.txt
```

### Running the application

The trained ResNet18 checkpoint must first be available locally at:

```text
training/checkpoints/resnet18_best.pth
```

Model checkpoints are not included in the GitHub repository.

Start the application from the project root:

```powershell
python app\run.py
```

Run application tests:

```powershell
python -m pytest app\tests -q
```

## Repository Exclusions

The following are intentionally excluded from GitHub:

- Raw and cleaned datasets.
- Dataset archives.
- Generated Mel-spectrogram images.
- Trained model checkpoints.
- Virtual environment files.
- Local application databases and uploaded recordings.

Check `.gitignore` before committing changes.

## Current Project Status

### Completed

- Indian-language audio dataset preparation.
- Audio cleaning and preprocessing.
- Mel-spectrogram generation.
- Training, validation, and test dataset preparation.
- ResNet18 model implementation and evaluation.
- BiLSTM model implementation and evaluation.
- Model comparison and evaluation visualizations.
- Flask-based Voice Guard application.
- Application testing.

### Further Work

- Evaluation on additional independent datasets.
- Broader cross-language and real-world testing.
- Improving inference speed and model efficiency.
- Further robustness testing against different audio conditions.

## Project Information

**Project:** Deepfake Voice Detection & Audio Forensics â€” Voice Guard
**Technology:** Python, PyTorch, Librosa, Flask, NumPy, Scikit-learn, SQLite
**Models:** ResNet18 and BiLSTM
**Application:** Flask-based WAV audio analysis dashboard

This is an academic project and its predictions should be treated as model-based indicators, not definitive forensic proof.
