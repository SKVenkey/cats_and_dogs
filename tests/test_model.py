"""Unit tests for model utilities and FastAPI inference endpoint."""

import sys
import json
import io
import os
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from catsml.model import SimpleCNN, TransferCNN, get_model


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
        model = SimpleCNN(num_classes=2)
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
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


class TestAPIEndpoints:
    @pytest.fixture
    def client(self, tmp_path):
        import api.app as app_module
        from fastapi.testclient import TestClient

        # Create dummy model files
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        dummy_model = SimpleCNN(num_classes=2)
        model_path = models_dir / "model.pt"
        torch.save(dummy_model.state_dict(), model_path)
        meta_path = models_dir / "model_meta.json"
        with open(meta_path, "w") as f:
            json.dump({"model_type": "simple_cnn", "class_names": ["Cat", "Dog"]}, f)

        # Patch environment variables
        app_module.MODEL_PATH = str(model_path)
        app_module.MODEL_META_PATH = str(meta_path)
        app_module.DEVICE = "cpu"

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
        assert r.json()["status"] == "ok"

    def test_predict_endpoint_returns_label(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("cat.jpg", img_bytes, "image/jpeg")})
        assert r.status_code == 200
        assert r.json()["label"] in ["Cat", "Dog"]

    def test_predict_endpoint_returns_probabilities(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("dog.jpg", img_bytes, "image/jpeg")})
        body = r.json()
        assert "probabilities" in body
        assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01

    def test_predict_endpoint_returns_confidence(self, client):
        img_bytes = self._make_image_bytes()
        r = client.post("/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")})
        assert 0.0 <= r.json()["confidence"] <= 1.0

    def test_metrics_endpoint(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200