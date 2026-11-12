import base64
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as functional

from .audio import create_display_spectrogram, create_spectrogram, split_audio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.resnet18_model import create_resnet18


class VoiceDetector:
    def __init__(self, model, device, checkpoint_metadata=None):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.model.eval()

    @classmethod
    def load(cls, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError("The ResNet18 checkpoint was not found.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
            raise ValueError("The checkpoint does not contain model_state_dict.")
        if checkpoint.get("class_to_label") != {"real": 0, "fake": 1}:
            raise ValueError("Checkpoint labels do not match REAL=0 and FAKE=1.")
        if tuple(checkpoint.get("input_size", ())) != (224, 336):
            raise ValueError("Checkpoint input size does not match (224, 336).")

        model = create_resnet18(pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        return cls(
            model,
            device,
            {key: checkpoint.get(key) for key in ("epoch", "input_size", "class_to_label")},
        )

    def predict_tensor(self, tensor):
        tensor = tensor.to(self.device, dtype=torch.float32)
        if tuple(tensor.shape[-2:]) != (224, 336):
            tensor = functional.interpolate(
                tensor,
                size=(224, 336),
                mode="bilinear",
                align_corners=False,
            )
        with torch.inference_mode():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0].cpu().tolist()
        return {"real": float(probabilities[0]), "fake": float(probabilities[1])}

    def analyze(self, audio):
        started = time.perf_counter()
        segment_results = []
        display_image = None

        for index, segment in enumerate(split_audio(audio), start=1):
            tensor, mel_db = create_spectrogram(segment)
            probabilities = self.predict_tensor(tensor)
            prediction = "FAKE" if probabilities["fake"] >= 0.5 else "REAL"
            segment_results.append(
                {
                    "segment": index,
                    "prediction": prediction,
                    "real_probability": probabilities["real"],
                    "fake_probability": probabilities["fake"],
                    "confidence": max(probabilities.values()),
                }
            )
            if display_image is None:
                display_image = create_display_spectrogram(mel_db)

        overall_fake = sum(item["fake_probability"] for item in segment_results) / len(segment_results)
        overall_real = 1.0 - overall_fake
        prediction = "FAKE" if overall_fake >= 0.5 else "REAL"
        return {
            "prediction": prediction,
            "real_probability": overall_real,
            "fake_probability": overall_fake,
            "confidence": max(overall_real, overall_fake),
            "processing_seconds": time.perf_counter() - started,
            "segment_count": len(segment_results),
            "segments": segment_results,
            "spectrogram": "data:image/png;base64," + base64.b64encode(display_image).decode("ascii"),
        }