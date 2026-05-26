# Customer Intelligence Platform

A robust, full-stack Customer Intelligence Platform built for Week 13. This platform features an **XGBoost-based campaign conversion prediction ML service** and a **FAISS/LLaMA-2 powered Retrieval-Augmented Generation (RAG) engine** for automated complaint resolution. Both services are fully containerized with Docker and orchestrated via FastAPI.

## Key Features & Requirement Fulfillment

### 1. Robust Machine Learning Pipeline (Conversion)
- **Data Validation (`scripts/validate_data.py`)**: Uses `pandera` to enforce strict schemas, check missing values, and validate 5+ business rules before training.
- **Baseline vs. XGBoost Promotion Gate (`services/conversion/train.py`)**: Evaluates a Logistic Regression baseline against XGBoost using ROC-AUC and PR-AUC. The model is only serialized if XGBoost rigorously outperforms the baseline.
- **ML Data Drift Monitoring (`scripts/generate_drift_report.py`)**: Automated script to detect statistical drift (using Kolmogorov-Smirnov tests) between training and production distributions, outputting to `monitoring/drift_report.json`.

### 2. Comprehensive APIs (`services/conversion/main.py`)
- **`POST /predict`**: Single inference endpoint with strict Pydantic validation.
- **`POST /batch-score`**: Efficient bulk scoring for multiple contacts simultaneously.
- **`POST /customer-intel`**: A unified intelligence endpoint that returns the ML conversion band (High/Medium/Low) alongside synthesized top complaint themes with cited evidence IDs.

### 3. Complaint Intelligence RAG (`services/rag/`)
- **FAISS & Sentence-Transformers**: Local, ultra-fast vector index for semantic search.
- **Refusal Logic**: Implements a strict similarity distance threshold. If retrieved context is weak, the model explicitly refuses to hallucinate an answer.
- **Automated Eval (`scripts/eval_rag.py`)**: Test suite evaluating the model's ability to answer in-domain questions and correctly refuse out-of-domain questions.

### 4. CI/CD & Cloud Infrastructure
- **GitHub Actions (`.github/workflows/ci.yml`)**: Continuous integration pipeline that enforces unit tests and data validation on every push.
- **Azure Monitoring**: Deep `opencensus-ext-azure` integration capturing custom telemetry, errors, and traces.
- **Docker Compose**: One-click local orchestration of both microservices.

---

## System Architecture

```mermaid
graph TD
    Client[Client Application / API Consumer]
    
    subgraph "API Gateway"
        Nginx[Nginx Reverse Proxy / Load Balancer]
    end

    subgraph "Conversion Service (FastAPI)"
        ConvAPI[POST /predict]
        BatchAPI[POST /batch-score]
        IntelAPI[POST /customer-intel]
        XGB[XGBoost Model]
        
        ConvAPI --> XGB
        BatchAPI --> XGB
        IntelAPI --> XGB
    end

    subgraph "Complaint Intelligence Service (FastAPI)"
        RAGAPI[POST /answer]
        FAISS[(FAISS Vector Store)]
        LLaMA[LLaMA2 LLM via local / API]
        RAGAPI --> FAISS
        RAGAPI --> LLaMA
    end
    
    subgraph "Monitoring & Analytics"
        AzureMonitor[Azure App Insights / Promtail]
    end

    Client -->|HTTP Requests| Nginx
    Nginx --> ConvAPI
    Nginx --> BatchAPI
    Nginx --> IntelAPI
    Nginx --> RAGAPI
    
    ConvAPI -.->|Telemetry| AzureMonitor
    RAGAPI -.->|Telemetry| AzureMonitor

    classDef service fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef datastore fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef monitor fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    
    class ConvAPI,BatchAPI,IntelAPI,RAGAPI service;
    class FAISS,XGB datastore;
    class AzureMonitor monitor;
```

## Detailed Documentation
Please review the `docs/` directory for full technical reports justifying these architectural decisions:
- [Comparative Analysis](docs/comparative_analysis.md)
- [Architecture](docs/architecture.md)
- [ML Model Report](docs/model_report.md)
- [RAG Report](docs/rag_report.md)
- [Monitoring Report](docs/monitoring_report.md)

## Frontend

You can run the Streamlit UI locally or via Docker.

**Local (Windows)**
```bat
frontend_start.bat
```
This will launch the app on `http://localhost:8501`.

**Docker**
```bash
docker build -f docker/Dockerfile.frontend -t cip-frontend .
docker run -p 8501:8501 cip-frontend
```
The container uses the same `requirements.txt` and starts via the provided `entrypoint.sh`.

---


1. Generate Sample Data:
```bash
python scripts/generate_sample_data.py
```
2. Validate & Train Model:
```bash
python scripts/validate_data.py
python services/conversion/train.py
```
3. Run Services:
```bash
docker-compose up --build
```

---

## Deployed Services on Azure

| # | Service | URL | Status | Notes |
|---|---|---|---|---|
| 1 | ⚙️ Conversion API | [https://cip-app-13.azurewebsites.net](https://cip-app-13.azurewebsites.net) | ✅ 200 | XGBoost /predict, /batch-score, /customer-intel |
| 2 | 💬 RAG API | [https://cip-rag-13.azurewebsites.net](https://cip-rag-13.azurewebsites.net) | ✅ 200 | FAISS + Groq /answer endpoint |
| 3 | 🆕 Frontend V2 (new link) | [https://cip-frontend-v2-13.azurewebsites.net](https://cip-frontend-v2-13.azurewebsites.net) | ✅ 200 | Updated RAG call, Groq-powered |

### Azure Deployment Commands

The infrastructure can be provisioned and deployed using Azure CLI commands or helper scripts.

#### 1. Interactive CLI Login
To authenticate with Azure:
```bash
az login --use-device-code
```

#### 2. Provisioning & Initial Deployment
You can use the PowerShell provisioning script for setting up the RAG service:
```powershell
.\infra\provision_rag.ps1
```
Or run the deployment script for the conversion/RAG multi-container stack:
```bash
bash infra/azure_deploy.sh
```

#### 3. Continuous Deployment (CI/CD)
The deployment is automated via GitHub Actions on every push to the `main` branch. 
To deploy changes, push to GitHub:
```bash
git add .
git commit -m "Deploy update"
git push origin main
```

#### 4. Manual Deployment & Updates
If you need to manually configure registry credentials, update settings, or restart specific web apps:

**Retrieve Container Registry Credentials:**
```bash
az acr credential show --name cipregistry13 --query "passwords[0].value" --output tsv
```

**Deploy/Configure a Single Container Web App:**
```bash
# 1. Update the container image
az webapp config container set \
  --name cip-rag-13 \
  --resource-group cip-rg-13 \
  --docker-custom-image-name cipregistry13.azurecr.io/weel13-rag-azure:latest \
  --docker-registry-server-url https://cipregistry13.azurecr.io \
  --docker-registry-server-user cipregistry13 \
  --docker-registry-server-password <ACR_PASSWORD>

# 2. Configure app settings (e.g. for Frontend V2)
az webapp config appsettings set \
  --name cip-frontend-v2-13 \
  --resource-group cip-rg-13 \
  --settings \
    WEBSITES_PORT=8501 \
    ENVIRONMENT=production \
    CONVERSION_SERVICE_URL=https://cip-app-13.azurewebsites.net \
    RAG_SERVICE_URL=https://cip-rag-13.azurewebsites.net \
    GROQ_API_KEY=<GROQ_API_KEY>

# 3. Restart the web app to apply changes
az webapp restart --name cip-frontend-v2-13 --resource-group cip-rg-13
```

