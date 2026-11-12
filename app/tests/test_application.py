import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

APP_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_DIR.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from voiceguard import create_app
from voiceguard.audio import (
    AudioInputError,
    IMAGE_SIZE,
    SAMPLE_RATE,
    SEGMENT_SAMPLES,
    create_spectrogram,
    load_audio,
    split_audio,
)
from voiceguard.inference import VoiceDetector
from voiceguard.storage import add_analysis, get_history, initialize_database

CHECKPOINT = PROJECT_ROOT / "training" / "checkpoints" / "resnet18_best.pth"


class FakeDetector:
    device = torch.device("cpu")

    def __init__(self, fake_probabilities=None):
        self.fake_probabilities = fake_probabilities or [0.8]
        self.calls = 0

    def analyze(self, audio):
        from voiceguard.audio import create_display_spectrogram, create_spectrogram

        segment_results = []
        display = None
        for index, segment in enumerate(split_audio(audio), start=1):
            _, mel = create_spectrogram(segment)
            if display is None:
                import base64

                display = "data:image/png;base64," + base64.b64encode(
                    create_display_spectrogram(mel)
                ).decode("ascii")
            fake = self.fake_probabilities[min(self.calls, len(self.fake_probabilities) - 1)]
            self.calls += 1
            real = 1 - fake
            prediction = "FAKE" if fake >= 0.5 else "REAL"
            segment_results.append(
                {
                    "segment": index,
                    "prediction": prediction,
                    "real_probability": real,
                    "fake_probability": fake,
                    "confidence": max(real, fake),
                }
            )
        fake = sum(item["fake_probability"] for item in segment_results) / len(segment_results)
        real = 1 - fake
        prediction = "FAKE" if fake >= 0.5 else "REAL"
        return {
            "prediction": prediction,
            "real_probability": real,
            "fake_probability": fake,
            "confidence": max(real, fake),
            "processing_seconds": 0.01,
            "segment_count": len(segment_results),
            "segments": segment_results,
            "spectrogram": display,
        }


@pytest.fixture
def client(tmp_path):
    app = create_app(
        {"TESTING": True, "DATABASE": str(tmp_path / "history.sqlite3")},
        detector=FakeDetector(),
    )
    return app.test_client()


def wav_bytes(seconds=1.2, sample_rate=16000, channels=1):
    count = int(seconds * sample_rate)
    timeline = np.arange(count, dtype=np.float32) / sample_rate
    mono = 0.35 * np.sin(2 * np.pi * 220 * timeline)
    data = np.column_stack([mono] * channels) if channels > 1 else mono
    output = io.BytesIO()
    sf.write(output, data, sample_rate, format="WAV", subtype="PCM_16")
    return output.getvalue()


def test_checkpoint_loads_with_expected_architecture():
    detector = VoiceDetector.load(CHECKPOINT)
    assert detector.device.type in {"cpu", "cuda"}
    assert detector.checkpoint_metadata["class_to_label"] == {"real": 0, "fake": 1}
    assert tuple(detector.checkpoint_metadata["input_size"]) == IMAGE_SIZE


def test_load_audio_downmixes_resamples_and_peak_normalizes(tmp_path):
    rate = 8000
    data = np.column_stack(
        [np.full(rate * 2, 0.25, dtype=np.float32), np.full(rate * 2, -0.75, dtype=np.float32)]
    )
    path = tmp_path / "stereo.wav"
    sf.write(path, data, rate)
    result = load_audio(path)
    assert result.shape == (SAMPLE_RATE * 2,)
    assert np.max(np.abs(result)) == pytest.approx(1.0)
    assert np.mean(result) == pytest.approx(-1.0)


def test_rejects_audio_shorter_than_one_second(tmp_path):
    path = tmp_path / "short.wav"
    sf.write(path, np.zeros(SAMPLE_RATE // 2, dtype=np.float32), SAMPLE_RATE)
    with pytest.raises(AudioInputError, match="one second"):
        load_audio(path)


def test_splits_audio_into_five_second_windows_and_pads_tail():
    audio = np.ones(SEGMENT_SAMPLES + SAMPLE_RATE, dtype=np.float32)
    segments = split_audio(audio)
    assert len(segments) == 2
    assert all(len(segment) == SEGMENT_SAMPLES for segment in segments)
    assert np.all(segments[-1][SAMPLE_RATE:] == 0)


def test_exact_five_second_audio_has_one_segment():
    assert len(split_audio(np.zeros(SEGMENT_SAMPLES, dtype=np.float32))) == 1


def test_mel_spectrogram_tensor_has_training_dimensions():
    signal = np.sin(2 * np.pi * 440 * np.arange(SEGMENT_SAMPLES) / SAMPLE_RATE).astype(np.float32)
    tensor, mel = create_spectrogram(signal)
    assert tensor.shape == (1, 1, *IMAGE_SIZE)
    assert mel.shape[0] == 128
    assert torch.isfinite(tensor).all()


def test_model_prediction_returns_normalized_class_probabilities():
    detector = VoiceDetector.load(CHECKPOINT)
    result = detector.predict_tensor(torch.zeros(1, 1, *IMAGE_SIZE))
    assert set(result) == {"real", "fake"}
    assert result["real"] + result["fake"] == pytest.approx(1.0, abs=1e-6)


def test_multisegment_fake_probability_uses_arithmetic_mean():
    detector = FakeDetector([0.8, 0.2])
    signal = np.zeros(SEGMENT_SAMPLES + SAMPLE_RATE, dtype=np.float32)
    audio = detector.analyze(signal)
    assert audio["segment_count"] == 2
    assert audio["fake_probability"] == pytest.approx(0.5)
    assert audio["real_probability"] == pytest.approx(0.5)
    assert audio["prediction"] == "FAKE"


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json["checkpoint_loaded"] is True
    assert response.json["labels"] == {"real": 0, "fake": 1}


def test_dashboard_route_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"VOICEGUARD" in response.data


def test_upload_endpoint_analyzes_and_saves_history(client):
    response = client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(wav_bytes()), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert response.json["prediction"] == "FAKE"
    assert response.json["confidence"] == response.json["fake_probability"]
    assert response.json["segment_count"] == 1
    assert response.json["real_probability"] + response.json["fake_probability"] == pytest.approx(1)
    assert response.json["spectrogram"].startswith("data:image/png;base64,")
    history = client.get("/api/history").json["items"]
    assert len(history) == 1
    assert history[0]["filename"] == "sample.wav"


def test_microphone_and_predict_routes_use_same_pipeline(client):
    for route in ("/api/microphone", "/api/predict"):
        response = client.post(
            route,
            data={"audio": (io.BytesIO(wav_bytes()), "capture.wav")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 200


def test_invalid_extension_and_content_are_rejected(client):
    extension = client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(b"not audio"), "sample.mp3")},
        content_type="multipart/form-data",
    )
    assert extension.status_code == 415
    corrupt = client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(b"not audio"), "sample.wav")},
        content_type="multipart/form-data",
    )
    assert corrupt.status_code == 400


def test_short_upload_is_rejected(client):
    response = client.post(
        "/api/upload",
        data={"audio": (io.BytesIO(wav_bytes(seconds=0.5)), "short.wav")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "one second" in response.json["error"]


def test_clear_history_endpoint(client):
    client.post(
        "/api/analyze",
        data={"audio": (io.BytesIO(wav_bytes()), "sample.wav")},
        content_type="multipart/form-data",
    )
    response = client.delete("/api/history")
    assert response.status_code == 200
    assert client.get("/api/history").json["items"] == []


def test_history_retains_only_the_latest_twenty_records(tmp_path):
    database = tmp_path / "history.sqlite3"
    initialize_database(database)
    result = {
        "prediction": "REAL",
        "confidence": 0.9,
        "real_probability": 0.9,
        "fake_probability": 0.1,
        "processing_seconds": 0.01,
        "segment_count": 1,
    }
    for index in range(23):
        add_analysis(database, f"record-{index}.wav", result)
    history = get_history(database)
    assert len(history) == 20
    assert history[0]["filename"] == "record-22.wav"
    assert history[-1]["filename"] == "record-3.wav"