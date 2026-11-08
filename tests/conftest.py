import pytest

import src.models.trainer as trainer
import src.fairness.mitigators as mitigators
import src.db.db_manager as db_manager


@pytest.fixture(autouse=True)
def isolate_artifacts(tmp_path, monkeypatch):
    """Redirect model files and the SQLite log to a temp dir so tests never overwrite
    the committed models/*.joblib artifacts or data/fairlens.db."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(trainer, "MODELS_DIR", models_dir)
    monkeypatch.setattr(mitigators, "MODELS_DIR", models_dir)
    monkeypatch.setattr(db_manager, "DB_PATH", tmp_path / "fairlens_test.db")
