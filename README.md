# Deepfake Voice Detection & Audio Forensics

A deepfake voice detection and audio forensics project focused on identifying AI-generated and spoofed speech using audio signal processing and deep learning.

The current development stage focuses on building a reliable, reusable audio preprocessing and dataset pipeline before finalizing the deep-learning architecture.

## Project Objectives

* Detect real (bonafide) and fake (spoofed) speech.
* Support multiple Indian languages.
* Build a reusable audio preprocessing pipeline.
* Convert cleaned audio into Mel-spectrogram representations.
* Maintain proper separation between training and independent evaluation data.
* Prepare the pipeline for future deep-learning models.
* Eventually support real-time deepfake voice detection.

## Current Language Support

The current Indian-language dataset pipeline includes:

* Kannada
* Hindi
* Tamil
* Telugu

The preprocessing pipeline is designed so that additional Indian languages can be added later.

## Dataset Organization

The project uses multiple data sources.

### Indian-language data

Indian-language real and synthetic speech are prepared for the main training pipeline.

The current focus includes:

* Real speech from IndicVoices
* Synthetic/fake speech from IndicSynt
* Kannada
* Hindi
* Tamil
* Telugu

### ASVspoof

ASVspoof data is maintained separately from the Indian-language training dataset.

It is used for independent testing/evaluation and is not mixed directly into the Indian-language training data.

> Dataset files are intentionally not stored in this GitHub repository because of their large size and dataset licensing/distribution considerations.

## Repository Structure

```text
deepfake-voice-detection/
│
├── .gitignore
├── README.md
│
├── collect_indicsynth.py
├── download_indic_real.py
├── download_kannada_test.py
├── prepare_test_dataset.py
├── preprocess_indian_audio.py
├── test_indicvoices.py
│
├── metadata/
│
├── preprocessing/
│   ├── check_source_groups.py
│   ├── clean_audio.py
│   ├── create_indian_split.py
│   ├── create_melspectrograms.py
│   ├── create_test_melspectrogram.py
│   ├── extract_la_dataset.py
│   ├── generate_kannada_mels.py
│   ├── generate_mels.py
│   ├── select_balanced_dataset.py
│   ├── verify_indian_mels.py
│   ├── verify_indian_split.py
│   ├── verify_mels.py
│   └── verify_preprocessing.py
│
├── protocols/
│
├── training/
│   ├── calculate_class_weights.py
│   ├── calculate_normalization.py
│   ├── dataset_loader.py
│   ├── test_augmentation.py
│   └── train_config.py
│
├── dataset/                 # Local only - not tracked
│   ├── raw/
│   └── cleaned/
│
└── features/                # Local only - not tracked
    └── mel_spectrograms/
```

## Preprocessing Pipeline

The current preprocessing workflow is:

```text
Raw Audio
    ↓
Dataset Collection
    ↓
Audio Cleaning & Validation
    ↓
16 kHz / Mono Standardization
    ↓
Duration & Audio Quality Checks
    ↓
Normalization
    ↓
Train/Validation/Test Preparation
    ↓
Mel-Spectrogram Generation
    ↓
Mel-Spectrogram Verification
    ↓
Training Dataset
```

The preprocessing stage is designed to remain independent of the final model architecture so that different deep-learning approaches can be evaluated later.

## Audio Standardization

The cleaned audio is checked for properties including:

* 16 kHz sampling rate
* Mono channel format
* Valid audio duration
* Readable audio files
* No NaN or infinite values
* Proper amplitude normalization

## Feature Representation

The current feature pipeline generates **Mel-spectrograms** from the cleaned audio.

Generated spectrograms are stored locally under:

```text
features/mel_spectrograms/
```

These generated files are excluded from GitHub because they can be recreated from the preprocessing pipeline.

## Verification

The project includes verification scripts for checking:

* Audio validity
* Indian-language dataset integrity
* Dataset splits
* Source-group separation
* Mel-spectrogram validity
* Preprocessing results

Examples:

```powershell
python verify_audio.py
python verify_indian_dataset.py
python verify_cleaned_indian.py
```

Additional verification utilities are available under:

```text
preprocessing/
```

## Environment Setup

### 1. Clone the repository

```powershell
git clone https://github.com/disha2005-stack/deepfake-voice-detection.git
cd deepfake-voice-detection
```

### 2. Create a virtual environment

```powershell
python -m venv venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install required Python packages

Install the dependencies required by the preprocessing scripts.

The project currently uses Python 3.13 and packages including:

* NumPy
* Librosa
* SoundFile
* Scikit-learn
* Pillow
* Hugging Face Datasets

Additional packages may be required as model development progresses.

## Dataset Setup

Datasets are **not included in this repository**.

They must obtain the required datasets separately and place them in the expected local directories.

The `.gitignore` file prevents large datasets, archives, generated features, and model files from being uploaded accidentally.

### Important

Do not commit:

```text
dataset/
features/
*.zip
ASVspoof2021_DF_eval_part00/
temp_la_train/
```

## Test Dataset Preparation

The `prepare_test_dataset.py` script prepares a small ASVspoof evaluation/test subset.

Run:

```powershell
python prepare_test_dataset.py
```

The script uses project-relative paths, so it does not depend on a specific Windows username or installation directory.

## Team Development Workflow

Before starting work:

```powershell
git pull
```

After making changes:

```powershell
git status
git add .
git commit -m "Describe your changes"
git push
```

### Important

Do not commit large datasets or generated files.

Always check:

```powershell
git status
```

before pushing.

## Current Development Status

### Completed

* [x] Project repository created
* [x] Git/GitHub integration
* [x] Dataset preprocessing structure
* [x] Indian-language data pipeline
* [x] Kannada, Hindi, Tamil and Telugu pipeline preparation
* [x] Audio cleaning and validation
* [x] Mel-spectrogram generation pipeline
* [x] Dataset verification scripts
* [x] Source-group analysis
* [x] Class imbalance analysis
* [x] Training configuration preparation
* [x] Portable test dataset preparation script

### In Progress

* [ ] Final deep-learning architecture selection
* [ ] Model training
* [ ] Model evaluation
* [ ] Cross-language performance analysis
* [ ] Real-time inference pipeline
* [ ] Audio forensics analysis
* [ ] Final application/interface

## Future Direction

The project is intended to evolve toward a real-time deepfake voice detection system capable of:

* Multi-language deepfake detection
* Real-time audio analysis
* Deepfake confidence scoring
* Audio forensic indicators
* Model comparison
* Robust evaluation across different datasets and languages

## Project Team

This is a college PBL project developed as a team project.

---

**Note:** Dataset files and generated feature files are intentionally excluded from this repository. Follow the dataset setup instructions before running the complete preprocessing pipeline.
