# 🐱🐶 Cats vs Dogs — End-to-End MLOps Pipeline

**MLOps Assignment 2 | BITS Pilani | S1-25_AIMLCZG523**

An end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) covering model development, experiment tracking, containerisation, CI/CD, deployment, and monitoring.

---

## 📁 Project Structure

```
cats-dogs-mlops/
├── .github/workflows/ci-cd.yml   # M3 & M4: GitHub Actions CI/CD
├── api/
│   └── app.py                    # M2: FastAPI inference service
├── deployment/
│   ├── docker-compose.yml        # M4: Docker Compose deployment
│   ├── k8s-deployment.yaml       # M4: Kubernetes Deployment
│   ├── k8s-service.yaml          # M4: Kubernetes Service
│   └── prometheus.yml            # M5: Prometheus scrape config
├── scripts/
│   ├── prepare_data.py           # M1: Dataset preparation
│   ├── train.py                  # M1: Training + MLflow tracking
│   ├── smoke_test.py             # M4: Post-deploy smoke tests
│   └── monitor.py                # M5: Post-deployment monitoring
├── src/catsml/
│   ├── data.py                   # M1: Data loading & preprocessing
│   └── model.py                  # M1: CNN model definitions
├── tests/
│   ├── test_data.py              # M3: Unit tests — preprocessing
│   └── test_model.py             # M3: Unit tests — model & API
├── dvc.yaml                      # M1: DVC pipeline stages
├── params.yaml                   # M1: Hyperparameter config
├── Dockerfile                    # M2: Container image definition
├── Makefile                      # Developer convenience commands
├── pytest.ini                    # Test configuration
└── requirements.txt              # Pinned dependencies
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
# or
make install
```

### 2. Prepare data (M1)
Place the Kaggle Cats vs Dogs dataset in `data/raw/train/Cat/` and `data/raw/train/Dog/`, then:
```bash
# Download automatically (requires Kaggle API key):
python scripts/prepare_data.py --download --data_dir data/raw

# Or verify manually placed data:
python scripts/prepare_data.py --data_dir data/raw

# Track with DVC:
dvc add data/raw
git add data/raw.dvc .gitignore
git commit -m "Track dataset with DVC"
```

### 3. Train model (M1)
```bash
# Baseline CNN:
make train

# Transfer learning (ResNet-18, better accuracy):
make train-transfer

# View experiments:
make mlflow-ui   # → http://localhost:5000
```

### 4. Run inference API locally (M2)
```bash
make serve   # → http://localhost:8000/docs
```

Test with curl:
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -F "file=@my_cat.jpg"
```

### 5. Run unit tests (M3)
```bash
make test
```

### 6. Build & run Docker image (M2 & M3)
```bash
make docker-build
make docker-run
```

### 7. Deploy with Docker Compose (M4)
```bash
make deploy-compose
make smoke-test
```

### 8. Monitor (M5)
```bash
python scripts/monitor.py --host http://localhost:8000 --n_samples 100
```

Prometheus metrics: http://localhost:9090  
Grafana dashboard: http://localhost:3000 (admin/admin)

---

## 🔄 CI/CD (GitHub Actions)

On every `push` to `main` the pipeline automatically:
1. Checks out code
2. Installs dependencies
3. Runs unit tests (`pytest`)
4. Builds Docker image
5. Pushes to Docker Hub
6. Deploys and runs smoke tests

**Required GitHub Secrets:**
| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |

---

## 📊 Module Summary

| Module | Component | Tool |
|---|---|---|
| M1 | Code versioning | Git |
| M1 | Data versioning | DVC |
| M1 | Model training | PyTorch |
| M1 | Experiment tracking | MLflow |
| M2 | REST API | FastAPI + Uvicorn |
| M2 | Containerisation | Docker |
| M3 | Unit tests | pytest |
| M3 | CI pipeline | GitHub Actions |
| M3 | Image registry | Docker Hub |
| M4 | Deployment | Docker Compose / Kubernetes |
| M4 | CD pipeline | GitHub Actions |
| M4 | Smoke tests | Custom script |
| M5 | Metrics | Prometheus + Grafana |
| M5 | Performance tracking | Custom monitoring script |

---

## 🏗️ Architecture

```
[Git + DVC] → [Train Script] → [MLflow] → [Model Artifact]
                                                ↓
                                        [FastAPI Service]
                                                ↓
                                          [Dockerfile]
                                                ↓
               [GitHub Actions CI] → [Docker Hub Registry]
                                                ↓
                                    [Docker Compose / k8s]
                                                ↓
                               [Prometheus + Grafana Monitoring]
```

---

## 📦 Dataset

[Cats and Dogs dataset on Kaggle](https://www.kaggle.com/datasets/salader/dogs-vs-cats)  
- Images resized to **224×224 RGB**  
- Split: **80% train / 10% val / 10% test**  
- Augmentation: horizontal flip, rotation ±10°, colour jitter
