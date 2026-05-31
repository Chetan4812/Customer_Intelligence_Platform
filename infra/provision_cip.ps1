# =============================================================================
# provision_cip.ps1 — Provision All Azure Resources for Customer Intelligence Platform
# =============================================================================
# Prerequisites:
#   • Azure CLI installed: https://aka.ms/installazurecliwindows
#   • Logged in: az login
#   • Run from repo root: .\infra\provision_cip.ps1
# =============================================================================

param(
    [string]$ResourceGroup    = "week_13_deployment_proj1",
    [string]$Location         = "centralindia",
    [string]$AcrName          = "cipregistry15",
    [string]$AppPlan          = "cip-plan",
    [string]$AppPlanLocation  = "southeastasia", # Southeast Asia supports Linux plans without quota limits
    [string]$AppPlanSku       = "B1",            # Basic Tier (recommended to avoid Free tier daily CPU minute limits with 4 concurrent apps)
    [string]$GroqApiKey       = $null
)

$ErrorActionPreference = "Stop"

function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ── 0. Load Groq API Key from .env or Env if not provided ────────────────────
if ($null -eq $GroqApiKey) {
    if (Test-Path ".env") {
        Get-Content ".env" | ForEach-Object {
            if ($_ -match "^GROQ_API_KEY=(.+)$") {
                $GroqApiKey = $Matches[1].Trim()
            }
        }
    }
    if ($null -eq $GroqApiKey -and $null -ne $env:GROQ_API_KEY) {
        $GroqApiKey = $env:GROQ_API_KEY
    }
}

if ($null -eq $GroqApiKey -or $GroqApiKey -eq "") {
    Write-Warn "GROQ_API_KEY not found in parameters, .env, or env variables. Using a placeholder."
    $GroqApiKey = "MOCK_GROQ_API_KEY_PLACEHOLDER" # Default/Fallback placeholder to pass secret scanning
}

# ── 1. Verify Azure CLI ──────────────────────────────────────────────────────
Write-Info "Checking Azure CLI..."
try {
    $null = az --version 2>&1
    Write-Ok "Azure CLI found."
} catch {
    Write-Fail "Azure CLI not found. Install from: https://aka.ms/installazurecliwindows"
}

# ── 2. Check Login Status ───────────────────────────────────────────────────
Write-Info "Verifying Azure login..."
$account = az account show --query "user.name" --output tsv 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Not logged in. Running 'az login'..."
    az login
    $account = az account show --query "user.name" --output tsv
}
Write-Ok "Logged in as: $account"

$subscriptionId = az account show --query id --output tsv
Write-Ok "Subscription ID: $subscriptionId"

# ── 3. Create Resource Group ────────────────────────────────────────────────
Write-Info "Ensuring Resource Group '$ResourceGroup' in '$Location'..."
az group create `
    --name $ResourceGroup `
    --location $Location `
    --output none
Write-Ok "Resource Group ready."

# ── 4. Create Azure Container Registry (ACR) ────────────────────────────────
Write-Info "Ensuring ACR '$AcrName'..."
$acrExists = az acr list --query "[?name=='$AcrName'].name" --output tsv

if (-not $acrExists) {
    az acr create `
        --resource-group $ResourceGroup `
        --name $AcrName `
        --sku Basic `
        --admin-enabled true `
        --output none
    Write-Ok "ACR created."
} else {
    Write-Ok "ACR already exists."
}

$AcrLoginServer = az acr show --name $AcrName --query loginServer --output tsv
$AcrPassword = az acr credential show --name $AcrName --query "passwords[0].value" --output tsv

Write-Ok "ACR Login Server: $AcrLoginServer"

# ── 5. Create Linux App Service Plan (Free F1 Tier) ────────────────────────
Write-Info "Ensuring App Service Plan '$AppPlan' (SKU=$AppPlanSku, Linux, Location=$AppPlanLocation)..."
$planExists = az appservice plan list `
    --resource-group $ResourceGroup `
    --query "[?name=='$AppPlan'].name" `
    --output tsv

if (-not $planExists) {
    az appservice plan create `
        --name $AppPlan `
        --resource-group $ResourceGroup `
        --is-linux `
        --sku $AppPlanSku `
        --location $AppPlanLocation `
        --output none
    Write-Ok "App Service Plan created."
} else {
    Write-Ok "App Service Plan already exists."
}

# Placeholder image for web app initialization
$PlaceholderImage = "mcr.microsoft.com/appsvc/staticsite:latest"

# ── 6. Ensure Microservice Web Apps Exist ──────────────────────────────────
$Apps = @("cip-app-15", "cip-rag-15", "cip-frontend-15", "cip-frontend-v2-15")

foreach ($AppName in $Apps) {
    Write-Info "Ensuring Web App '$AppName'..."
    $appExists = az webapp list `
        --resource-group $ResourceGroup `
        --query "[?name=='$AppName'].name" `
        --output tsv

    if (-not $appExists) {
        az webapp create `
            --resource-group $ResourceGroup `
            --plan $AppPlan `
            --name $AppName `
            --deployment-container-image-name $PlaceholderImage `
            --output none
        Write-Ok "Web App '$AppName' created."
    } else {
        Write-Ok "Web App '$AppName' already exists."
    }

    # Configure ACR Login Credentials on the Web App
    Write-Info "Configuring ACR credentials on '$AppName'..."
    az webapp config container set `
        --name $AppName `
        --resource-group $ResourceGroup `
        --container-registry-url "https://$AcrLoginServer" `
        --container-registry-user $AcrName `
        --container-registry-password $AcrPassword `
        --output none
    Write-Ok "ACR configured on '$AppName'."
}

# ── 7. Configure Specific App Settings ──────────────────────────────────────
Write-Info "Configuring Application Settings (Environment Variables)..."

# 1. ML Conversion Service Web App
Write-Info "Configuring settings for cip-app-15..."
az webapp config appsettings set `
    --name cip-app-15 `
    --resource-group $ResourceGroup `
    --settings `
        WEBSITES_PORT=8000 `
        SERVICE_NAME=conversion-service `
        --output none

# 2. RAG Intelligence Service Web App
Write-Info "Configuring settings for cip-rag-15..."
az webapp config appsettings set `
    --name cip-rag-15 `
    --resource-group $ResourceGroup `
    --settings `
        WEBSITES_PORT=80 `
        SERVICE_NAME=rag-service `
        LLM_MODEL_NAME=mock `
        EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2 `
        CHUNK_SIZE=400 `
        CHUNK_OVERLAP=50 `
        TOP_K=5 `
        MAX_NEW_TOKENS=512 `
        HF_HOME=/app/.cache/huggingface `
        GROQ_API_KEY=$GroqApiKey `
        --output none

# 3. Streamlit Frontend Web App
Write-Info "Configuring settings for cip-frontend-15..."
az webapp config appsettings set `
    --name cip-frontend-15 `
    --resource-group $ResourceGroup `
    --settings `
        WEBSITES_PORT=8501 `
        ENVIRONMENT=production `
        CONVERSION_SERVICE_URL=https://cip-app-15.azurewebsites.net `
        RAG_SERVICE_URL=https://cip-rag-15.azurewebsites.net `
        GROQ_API_KEY=$GroqApiKey `
        --output none

# 4. Streamlit Frontend V2 Web App
Write-Info "Configuring settings for cip-frontend-v2-15..."
az webapp config appsettings set `
    --name cip-frontend-v2-15 `
    --resource-group $ResourceGroup `
    --settings `
        WEBSITES_PORT=8501 `
        ENVIRONMENT=production `
        CONVERSION_SERVICE_URL=https://cip-app-15.azurewebsites.net `
        RAG_SERVICE_URL=https://cip-rag-15.azurewebsites.net `
        GROQ_API_KEY=$GroqApiKey `
        --output none

Write-Ok "All Application Settings successfully configured."

# ── 8. Enable Continuous Deployment webhooks ─────────────────────────────────
Write-Info "Enabling Continuous Deployment webhooks..."
$cdWebhooks = @{}
foreach ($AppName in $Apps) {
    $cdUrl = az webapp deployment container config `
        --name $AppName `
        --resource-group $ResourceGroup `
        --enable-cd true `
        --query CI_CD_URL `
        --output tsv
    $cdWebhooks[$AppName] = $cdUrl
}
Write-Ok "CD webhooks enabled."

# ── 9. Summary & Next Steps ──────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PROVISIONING AND CONFIGURATION SUCCESSFUL!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Web App Service Endpoints (Live Placeholders):" -ForegroundColor White
Write-Host "  • ML Service       : https://cip-app-15.azurewebsites.net"
Write-Host "  • RAG Service      : https://cip-rag-15.azurewebsites.net"
Write-Host "  • Streamlit Web v1 : https://cip-frontend-15.azurewebsites.net"
Write-Host "  • Streamlit Web v2 : https://cip-frontend-v2-15.azurewebsites.net"
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  GITHUB ACTIONS SECRETS TO UPDATE:" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. GROQ_API_KEY    : $GroqApiKey"
Write-Host "  2. ACR_PASSWORD    : $AcrPassword"
foreach ($entry in $cdWebhooks.GetEnumerator()) {
    Write-Host "  3. CD_WEBHOOK_URL (${entry.Key}) : ${entry.Value}"
}
Write-Host ""
Write-Host "  To generate AZURE_CREDENTIALS for your GitHub secret, run:" -ForegroundColor Cyan
Write-Host "  az ad sp create-for-rbac --name cip-azure-sp-15 --role contributor \`" -ForegroundColor White
Write-Host "    --scopes /subscriptions/$subscriptionId/resourceGroups/$ResourceGroup \`" -ForegroundColor White
Write-Host "    --sdk-auth" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
