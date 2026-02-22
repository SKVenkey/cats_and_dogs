"""Unit tests for model utilities and FastAPI inference endpoint (M3 requirement)."""

import sys
import json
import io
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from catsml.model import SimpleCNN, TransferCNN, get_model


# ─── Model architecture tests ─────────────────────────────────────────────────

class TestSimpleCNN:
    def test_output_shape(self):
        model = SimpleCNN(num_classes=2)
        x = torch.randn(4, 3, 224, 224)
        out = model(x)
        assert out.shape == (4, 2)

    def test_single_sample(self):
        model = SimpleCNN(num_classes=2)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        assert out.shape == (1, 2)

    def test_output_is_logits_not_probs(self):
        """Output should be raw logits, not probabilities."""
        model = SimpleCNN(num_classes=2)
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        # Logits should not all sum to 1
        row_sums = out.sum(dim=1)
        assert not torch.allclose(row_sums, torch.ones(8), atol=0.01)

    def test_trainable_params_exist(self):
        model = SimpleCNN()
        params = [p for p in model.parameters() if p.requires_grad]
        assert len(params) > 0


class TestTransferCNN:
    def test_output_shape(self):
        model = TransferCNN(num_classes=2)
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)


class TestGetModel:
    def test_get_simple_cnn(self):
        model = get_model("simple_cnn")
        assert isinstance(model, SimpleCNN)

    def test_get_transfer_resnet18(self):
        model = get_model("transfer_resnet18")
        assert isinstance(model, TransferCNN)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError):
            get_model("unknown_model_xyz")


# ─── FastAPI endpoint tests ───────────────────────────────────────────────────

class TestAPIEndpoints:
    """Integration tests for the FastAPI app using TestClient."""

    @pytest.fixture
    def mock_model(self):
        """Return a SimpleCNN instance for testing (no real weights needed)."""
        model = SimpleCNN(num_classes=2)
        model.eval()
        return model

    @pytest.fixture
    def client(self, mock_model, tmp_path):
        """Create a TestClient with the model pre-loaded."""
        import api.app as app_module
        from fastapi.testclient import TestClient

        # Patch model loading
        app_module._model = mock_model
        app_module._class_names = ["Cat", "Dog"]
        app_module.DEVICE = "cpu"

        # Create dummy model files so lifespan doesn't fail
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        torch.save(mock_model.state_dict(), models_dir / "model.pt")
        with open(models_dir / "model_meta.json", "w") as f:
            json.dump({"model_type": "simple_cnn", "class_names": ["Cat", "Dog"]}, f)

        with TestClient(app_module.app) as c:
            yield c

    def _make_image_bytes(self):
        from PIL import Image
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def test_health_endpoint(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_predict_endpoint_returns_label(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("cat.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert "label" in body
        assert body["label"] in ["Cat", "Dog"]

    def test_predict_endpoint_returns_probabilities(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("dog.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200
        body = r.json()
        assert "probabilities" in body
        probs = body["probabilities"]
        assert "Cat" in probs and "Dog" in probs
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_predict_endpoint_returns_confidence(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        body = r.json()
        assert 0.0 <= body["confidence"] <= 1.0

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
