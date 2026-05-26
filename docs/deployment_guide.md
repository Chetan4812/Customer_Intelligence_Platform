# Customer Intelligence Platform – Deployment & Resolution Runbook

This document details the sequence of steps, troubleshooting findings, and commands used to resolve the dependency conflicts and successfully deploy the **Customer Intelligence Platform** (Conversion Service, RAG Service, and Streamlit Frontend) both locally and to Azure.

---

## 1. Diagnostics & Root Cause Analysis

During verification, the deployed RAG endpoint at `https://cip-rag-13.azurewebsites.net/answer` returned `503 Service Unavailable` and eventually `504 Gateway Timeout` due to two primary issues:

### A. Dependency Conflict (`ImportError`)
In `requirements.txt`, the packages were originally defined as:
* `sentence-transformers==2.2.2`
* `huggingface_hub==0.25.2`

In `huggingface_hub` version `0.20.0` and above, the deprecated `cached_download` utility was removed. Because `sentence-transformers==2.2.2` imports `cached_download`, starting the RAG service failed with:
```text
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

### B. Azure App Service Memory Constraints (OOM)
The RAG service deployment workflow was configured with the environment variable `LLM_MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0`. 
Loading a 1.1B parameter LLM on CPU requires **> 2.2 GB of RAM**, which exceeded the memory limit of the standard `S1` Azure App Service plan (limited to **1.75 GB of RAM**). This caused the container to run out of memory (OOM) and crash repeatedly in a loop.

### C. Flaky CI/CD Smoke Test
The deployment workflow (`deploy.yml`) for the Conversion Service had a fragile **30-second fixed sleep** smoke-test check. If the container took slightly longer than 30 seconds to wake up during cold starts, the single check failed and aborted the entire deployment pipeline.

---

## 2. Implemented Fixes

To resolve these issues, the following files were updated:

1. **`requirements.txt`:** Upgraded `sentence-transformers` to `>=2.3.0` to resolve the Hugging Face Hub import conflict.
2. **`app.py`:** Updated the default environment setting dropdown to **Local (Development)** for a smoother local developer workflow.
3. **`.github/workflows/deploy.yml`:**
   * Changed `LLM_MODEL_NAME` to **`mock`** under the RAG App setting configuration to run the lightweight mock LLM on Azure and avoid memory crashes.
   * Replaced the fragile 30-second Conversion Service smoke test with a robust **3-minute polling loop** that checks `/health` every 10 seconds.

---

## 3. Sequence of Commands (Run Locally)

To run the full stack locally on your machine using your global environment (where all dependencies are correctly cached and fully working), open separate terminals and run:

### Step 1: Start the Conversion Backend Service (Port 8000)
```powershell
python -m uvicorn services.conversion.main:app --host 127.0.0.1 --port 8000
```
* **Verify Endpoint:** `http://127.0.0.1:8000/health` (should return `{"status":"ok","model_loaded":true}`)

### Step 2: Start the RAG Backend Service (Port 8001)
Use the `mock` LLM flag to save CPU memory and initialize the engine instantly:
```powershell
# In PowerShell:
$env:LLM_MODEL_NAME="mock"
python -m uvicorn services.rag.main:app --host 127.0.0.1 --port 8001

# In Bash:
LLM_MODEL_NAME="mock" python -m uvicorn services.rag.main:app --host 127.0.0.1 --port 8001
```
* **Verify Endpoint:** `http://127.0.0.1:8001/health` (should return `{"status":"ok","index_vectors":100}`)

### Step 3: Start the Streamlit Frontend Dashboard (Port 8501)
```powershell
streamlit run app.py --server.port 8501
```
* **Open Dashboard:** Go to `http://localhost:8501` in your browser. The default environment is set to `Local (Development)` and will communicate with your local backends.

---

## 4. Sequence of Commands (Deploy to Azure)

Since the CI/CD pipeline triggers on pushes to the `main` branch, use this sequence of Git commands to update the deployed environment:

### Step 1: Stage the modified files
```bash
git add requirements.txt app.py .github/workflows/deploy.yml
```

### Step 2: Commit changes
```bash
git commit -m "Fix RAG dependency version conflict, mock LLM on Azure, and flaky smoke tests"
```

### Step 3: Push changes to GitHub
```bash
git push origin main
```

### Step 4: Monitor the Deployment Run
* Check the status of the automated build and deployment on your GitHub repository under the **Actions** tab:
  `https://github.com/shubhamrangdal2001-hash/Week13_Project-customer-intelligence-platform/actions`
* Once the deployment workflow displays a **success (green check)**, the Azure containers will restart automatically.

---

## 5. Deployed Azure Endpoints (Verification)

After the pipeline successfully redeploys, you can access your platform at:

* **Streamlit Dashboard:** [https://cip-frontend-13.azurewebsites.net](https://cip-frontend-13.azurewebsites.net)
* **Conversion Service Health:** `https://cip-app-13.azurewebsites.net/health`
* **RAG Service Health:** `https://cip-rag-13.azurewebsites.net/health`
