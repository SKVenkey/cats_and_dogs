.PHONY: install data train test docker-build docker-run deploy clean

## ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install --upgrade pip
	pip install -r requirements.txt

## ── Data (M1) ─────────────────────────────────────────────────────────────────
data:
	python scripts/prepare_data.py --data_dir data/raw
	dvc add data/raw
	git add data/raw.dvc .gitignore

dvc-push:
	dvc push

## ── Train (M1) ────────────────────────────────────────────────────────────────
train:
	python scripts/train.py \
		--data_dir data/raw \
		--model_type simple_cnn \
		--epochs 15

train-transfer:
	python scripts/train.py \
		--data_dir data/raw \
		--model_type transfer_resnet18 \
		--epochs 10

## ── Test (M3) ─────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v

## ── API (M2) ──────────────────────────────────────────────────────────────────
serve:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

## ── Docker (M2 & M3) ──────────────────────────────────────────────────────────
docker-build:
	docker build -t cats-dogs-mlops:latest .

docker-run:
	docker run -p 8000:8000 -v $(PWD)/models:/app/models:ro cats-dogs-mlops:latest

docker-push:
	docker tag cats-dogs-mlops:latest $(DOCKERHUB_USERNAME)/cats-dogs-mlops:latest
	docker push $(DOCKERHUB_USERNAME)/cats-dogs-mlops:latest

## ── Deploy (M4) ───────────────────────────────────────────────────────────────
deploy-compose:
	cd deployment && docker compose up -d

smoke-test:
	python scripts/smoke_test.py --host http://localhost:8000

deploy-k8s:
	kubectl apply -f deployment/k8s-deployment.yaml
	kubectl apply -f deployment/k8s-service.yaml
	kubectl rollout status deployment/cats-dogs-api

## ── MLflow UI (M1) ────────────────────────────────────────────────────────────
mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

## ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache test-results.xml
