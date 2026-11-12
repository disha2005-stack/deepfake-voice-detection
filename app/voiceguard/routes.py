import os
import tempfile
from pathlib import Path

import soundfile as sf
from flask import current_app, jsonify, render_template, request

from .audio import AudioInputError, load_audio
from .storage import add_analysis, clear_history, get_history


def register_routes(app):
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        detector = current_app.extensions["voiceguard_detector"]
        return jsonify(
            status="ok",
            model="ResNet18",
            device=str(detector.device),
            checkpoint_loaded=True,
            labels={"real": 0, "fake": 1},
        )

    def analyze_upload():
        uploaded = request.files.get("audio")
        if uploaded is None or not uploaded.filename:
            return jsonify(error="Choose a WAV audio file to analyze."), 400
        if Path(uploaded.filename).suffix.lower() != ".wav":
            return jsonify(error="Only WAV audio files are supported."), 415

        safe_name = Path(uploaded.filename).name[:180]
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temporary:
                temporary_path = temporary.name
            uploaded.save(temporary_path)
            if os.path.getsize(temporary_path) == 0:
                raise AudioInputError("The audio file is empty.")
            try:
                audio_format = sf.info(temporary_path).format
            except Exception as error:
                raise AudioInputError("The uploaded file is not a readable WAV audio file.") from error
            if audio_format != "WAV":
                return jsonify(error="The file contents are not WAV audio."), 415

            audio = load_audio(temporary_path)
            result = current_app.extensions["voiceguard_detector"].analyze(audio)
            stored = add_analysis(current_app.config["DATABASE"], safe_name, result)
            result.update(filename=safe_name, **stored)
            return jsonify(result)
        except AudioInputError as error:
            return jsonify(error=str(error)), 400
        except Exception:
            current_app.logger.exception("Audio analysis failed")
            return jsonify(error="Audio analysis failed. Check the file and try again."), 500
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.remove(temporary_path)

    app.add_url_rule("/api/analyze", "analyze", analyze_upload, methods=["POST"])
    app.add_url_rule("/api/upload", "upload", analyze_upload, methods=["POST"])
    app.add_url_rule("/api/predict", "predict", analyze_upload, methods=["POST"])
    app.add_url_rule("/api/microphone", "microphone", analyze_upload, methods=["POST"])

    @app.get("/api/history")
    def history():
        return jsonify(items=get_history(current_app.config["DATABASE"]))

    @app.delete("/api/history")
    def delete_history():
        clear_history(current_app.config["DATABASE"])
        return jsonify(status="cleared")