#!/usr/bin/env bash
# Enterprise Linux MLOps Bootstrap, Test & Azure Container Registry Deployment Script
set -euo pipefail

echo "[1/4] Validating Python environment and running pytest suite..."
python -m pytest -v

echo "[2/4] Generating Kaggle GPU LLM & RAG Experiment Kernels..."
python -m src.generate_insurance_kaggle_suite

echo "[3/4] Building Docker image for Azure Kubernetes Service (AKS)..."
if command -v docker >/dev/null 2>&1; then
  docker build -t insurance-llm-rag-platform:1.0.0 .
fi

echo "[4/4] Verification complete. Ready for Azure ML / AKS deployment."
