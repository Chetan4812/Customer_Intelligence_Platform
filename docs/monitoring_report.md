# Cloud Monitoring & Azure Deployment Report

## 1. Overview
This report outlines the deployment and monitoring strategies implemented for the Customer Intelligence Platform, targeting an Azure-based infrastructure. 

## 2. Infrastructure as Code (Azure Deploy)
The deployment automation is handled via `infra/azure_deploy.sh`. This script:
1. Provisions an Azure Resource Group (`rg-customer-intel`).
2. Provisions an Azure App Service Plan targeting a Linux environment.
3. Deploys the Dockerized services via a Multi-Container App Service using `docker-compose.yml`.

## 3. Containerization
- **Base Image:** A shared `Dockerfile.base` installs the fundamental heavy dependencies (like `numpy`, `pandas`, `sklearn`, `transformers`).
- **Service Images:** `Dockerfile.conversion` and `Dockerfile.rag` build from the base image and add their respective local service code and weights, optimizing the Docker caching layers.
- **Orchestration:** `docker-compose.yml` maps ports (8000 for Conversion, 8001 for RAG) and handles the networking bridge between the FastAPI services.

## 4. Monitoring Strategy
Implemented in `monitoring/azure_monitor.py` utilizing the `opencensus-ext-azure` package:
- **Application Insights Integration:** A global `AzureMonitor` wrapper connects python `logging` directly to Azure App Insights via the `APPLICATIONINSIGHTS_CONNECTION_STRING`.
- **Telemetry Collected:**
  - FastApi Request success/failure rates
  - System exceptions (500, 503 errors)
  - Custom ML tracking metrics (e.g., Conversion Prediction values, RAG retrieval latencies).

## 5. Next Steps for CI/CD
To fully automate this process, a GitHub Actions workflow can be integrated to trigger the `azure_deploy.sh` script whenever new commits are pushed to the `main` branch.
