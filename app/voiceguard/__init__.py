from pathlib import Path

from flask import Flask, jsonify
from werkzeug.exceptions import RequestEntityTooLarge

from .routes import register_routes
from .storage import initialize_database


def create_app(test_config=None, detector=None):
    app = Flask(
        __name__,
        instance_relative_config=True,
        template_folder="../templates",
        static_folder="../static",
    )
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        DATABASE=str(Path(app.instance_path) / "voiceguard.sqlite3"),
        CHECKPOINT=str(
            Path(__file__).resolve().parents[2]
            / "training"
            / "checkpoints"
            / "resnet18_best.pth"
        ),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    initialize_database(app.config["DATABASE"])

    if detector is None:
        from .inference import VoiceDetector

        detector = VoiceDetector.load(app.config["CHECKPOINT"])
    app.extensions["voiceguard_detector"] = detector

    register_routes(app)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return jsonify(error="Audio file exceeds the 25 MB upload limit."), 413

    return app