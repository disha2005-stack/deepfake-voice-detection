# VOICEGUARD — Deepfake Voice Detection & Audio Forensics

VOICEGUARD is a local Flask dashboard for analyzing WAV recordings with the project's trained ResNet18 classifier. It reports overall and five-second segment-level probabilities, renders the first segment's mel spectrogram, and keeps up to 20 analysis summaries in a local SQLite database. Uploaded audio is temporary and deleted after analysis.

## Architecture

`run.py` creates the Flask app and loads `training/checkpoints/resnet18_best.pth` once at startup. `voiceguard/audio.py` applies the Indian audio pipeline, `voiceguard/inference.py` creates the checkpoint-compatible grayscale input and predicts each segment, `voiceguard/routes.py` serves the UI and JSON API, and `voiceguard/storage.py` maintains local history. The browser UI is under `templates/` and `static/`.

The mel feature path follows `preprocessing/generate_kannada_mels.py`: 16 kHz mono audio; linear resampling and peak normalization as in `preprocess_indian_audio.py`; consecutive five-second segments with final zero padding; 128 mel bands; FFT 1024; hop 256; 20–8000 Hz; power-to-dB with `ref=mel.max()`; 6×4 matplotlib rendering at 100 DPI; grayscale conversion; normalization with mean `0.197581` and standard deviation `0.200545`; and bilinear resize to 224×336 with `align_corners=False`. Those are the checkpoint's model inputs; the displayed chart adds readable axes and a color scale.

## Folder Structure

```text
app/
  .gitignore
  README.md
  requirements.txt
  run.py
  voiceguard/
    __init__.py
    audio.py
    inference.py
    routes.py
    storage.py
  templates/index.html
  static/css/style.css
  static/js/app.js
  tests/test_application.py
```

## Install and Launch (Windows PowerShell)

From the project directory, activate the existing virtual environment and install only the app dependencies:

```powershell
Set-Location "D:\PBL 5"
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
python -m pip install -r app\requirements.txt
```

The execution-policy command is only needed if PowerShell blocks activation. Then start the app:

```powershell
python app\run.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000). To run the automated tests:

```powershell
python -m pytest app\tests -q
```

## Using the Dashboard

Drop a WAV file on the input area or browse to select one. WAV uploads are limited to 25 MB and must contain at least one second of audio. The audio player previews the selected file. For recording, allow microphone access, select **Record**, then **Stop**. Browser MediaRecorder output is decoded and converted to mono PCM WAV in the browser before submission. Select **Analyze recording** to run inference. The same endpoint and processing pipeline are used for uploads and microphone captures.

The dashboard reports the overall class, real and fake probabilities, predicted-class confidence, processing duration, segment count, individual segment results, and the first segment's spectrogram. History shows the latest 20 records and can be cleared. SQLite data is stored in `app/instance/voiceguard.sqlite3`; that folder is ignored by Git.

## Prediction Calculation

For each segment, ResNet18 logits are passed through softmax. REAL is class 0 and FAKE is class 1. The overall fake probability is the arithmetic mean of the per-segment fake probabilities; overall real probability is `1 - fake_probability`. A fake probability of 0.5 or greater is classified as FAKE. Confidence is the probability of the predicted class, not a separate score.

## Models and Limitations

ResNet18 is the deployed primary detector and loads from the existing checkpoint. The trained BiLSTM is identified as the comparison model in the academic project but is not run by this dashboard. Model loading is strict and will prevent startup if the checkpoint is missing or incompatible. CUDA is selected when available; otherwise inference uses CPU.

Only WAV containers are accepted by the API. Browser microphone capture requires a secure browser context (localhost qualifies), microphone permission, and a browser that can decode its own MediaRecorder format. The backend resampler intentionally matches the project's linear interpolation. Spectrogram presentation is visual; model inference uses the training-style rendered grayscale image. History persists filenames and prediction metadata, not audio. The application is intended for local academic evaluation, not as a forensic proof or a production service.