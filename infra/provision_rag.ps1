# =============================================================================
# provision_rag.ps1 — Provision Azure Resources for CIP RAG Service
# =============================================================================
# Prerequisites:
#   • Azure CLI installed: https://aka.ms/installazurecliwindows
#   • Logged in: az login
#   • Docker Desktop running (optional — CI/CD handles the actual build/push)
#   • Run from repo root: .\infra\provision_rag.ps1
#
# What this script does:
#   1. Creates (or reuses) the Resource Group
#   2. Creates (or reuses) the Azure Container Registry (ACR)
#   3. Creates (or reuses) the App Service Plan (Linux, S1)
#   4. Creates (or reuses) the cip-rag-13 App Service (single container)
#   5. Sets all required App Settings (env vars) on the App Service
#   6. Prints GitHub Secrets you need to add
# =============================================================================

param(
    [string]$ResourceGroup   = "cip-rg-13",
    [string]$Location        = "centralindia",
    [string]$AcrName         = "cipregistry13",
    [string]$AppPlan         = "cip-plan",
    [string]$AppName         = "cip-rag-13",
    [string]$AppPlanSku      = "S1",
    [string]$GroqApiKey      = $env:GROQ_API_KEY
)

$ErrorActionPreference = "Stop"

function Write-Info  { param($msg) Write-Host "[INFO]  $msg" -ForegroundColor Cyan }
function Write-Ok    { param($msg) Write-Host "[OK]    $msg" -ForegroundColor Green }
function Write-Warn  { param($msg) Write-Host "[WARN]  $msg" -ForegroundColor Yellow }
function Write-Fail  { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red; exit 1 }

# ── 0. Verify Azure CLI ──────────────────────────────────────────────────────
Write-Info "Checking Azure CLI..."
try {
    $null = az --version 2>&1
    Write-Ok "Azure CLI found."
} catch {
    Write-Fail "Azure CLI not found. Install from: https://aka.ms/installazurecliwindows"
}

# ── 0b. Check login ──────────────────────────────────────────────────────────
Write-Info "Verifying Azure login..."
$account = az account show --query "user.name" --output tsv 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "Not logged in. Running 'az login'..."
    az login
}
Write-Ok "Logged in as: $account"

# ── 1. Resource Group ────────────────────────────────────────────────────────
Write-Info "Ensuring resource group '$ResourceGroup' in '$Location'..."
az group create `
    --name $ResourceGroup `
    --location $Location `
    --output none
Write-Ok "Resource group ready."

# ── 2. Azure Container Registry ──────────────────────────────────────────────
Write-Info "Ensuring ACR '$AcrName'..."
az acr create `
    --resource-group $ResourceGroup `
    --name $AcrName `
    --sku Basic `
    --admin-enabled true `
    --output none
Write-Ok "ACR ready."

$AcrLoginServer = az acr show `
    --name $AcrName `
    --query loginServer `
    --output tsv

$AcrPassword = az acr credential show `
    --name $AcrName `
    --query "passwords[0].value" `
    --output tsv

Write-Ok "ACR login server: $AcrLoginServer"

# ── 3. App Service Plan ───────────────────────────────────────────────────────
Write-Info "Ensuring App Service Plan '$AppPlan' (SKU=$AppPlanSku, Linux)..."

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
        --output none
    Write-Ok "App Service Plan created."
} else {
    Write-Ok "App Service Plan already exists — skipping creation."
}

# ── 4. RAG App Service (single container) ─────────────────────────────────────
Write-Info "Ensuring App Service '$AppName'..."

$appExists = az webapp list `
    --resource-group $ResourceGroup `
    --query "[?name=='$AppName'].name" `
    --output tsv

# Placeholder image — CI/CD will swap this to the real image on first deploy
$PlaceholderImage = "mcr.microsoft.com/appsvc/staticsite:latest"

if (-not $appExists) {
    Write-Info "Creating App Service '$AppName'..."
    az webapp create `
        --resource-group $ResourceGroup `
        --plan $AppPlan `
        --name $AppName `
        --deployment-container-image-name $PlaceholderImage `
        --output none
    Write-Ok "App Service created."
} else {
    Write-Ok "App Service already exists — skipping creation."
}

# ── 5. Configure ACR credentials on App Service ───────────────────────────────
Write-Info "Setting ACR credentials on App Service..."
az webapp config container set `
    --name $AppName `
    --resource-group $ResourceGroup `
    --container-registry-url "https://$AcrLoginServer" `
    --container-registry-user $AcrName `
    --container-registry-password $AcrPassword `
    --output none
Write-Ok "ACR credentials configured."

# ── 6. App Settings (environment variables) ───────────────────────────────────
Write-Info "Setting App Service environment variables..."

$settings = @(
    "GROQ_API_KEY=$GroqApiKey",
    "LLM_MODEL_NAME=mock",
    "EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2",
    "CHUNK_SIZE=400",
    "CHUNK_OVERLAP=50",
    "TOP_K=5",
    "WEBSITES_PORT=80",
    "SCM_DO_BUILD_DURING_DEPLOYMENT=false"
)

az webapp config appsettings set `
    --name $AppName `
    --resource-group $ResourceGroup `
    --settings @settings `
    --output none

Write-Ok "App Settings configured."

# ── 7. Enable Continuous Deployment webhook ────────────────────────────────────
Write-Info "Enabling Continuous Deployment webhook..."
$cdWebhookUrl = az webapp deployment container config `
    --name $AppName `
    --resource-group $ResourceGroup `
    --enable-cd true `
    --query CI_CD_URL `
    --output tsv

Write-Ok "Continuous deployment enabled."

# ── 8. Get public URL ─────────────────────────────────────────────────────────
$AppUrl = az webapp show `
    --name $AppName `
    --resource-group $ResourceGroup `
    --query defaultHostName `
    --output tsv

# ── 9. Print summary & GitHub secrets ─────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  PROVISIONING COMPLETE" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  RAG Endpoint   : https://$AppUrl" -ForegroundColor White
Write-Host "  Health Check   : https://$AppUrl/health" -ForegroundColor White
Write-Host "  Swagger Docs   : https://$AppUrl/docs" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host "  GITHUB SECRETS — Add these to your repo:" -ForegroundColor Yellow
Write-Host "  (Settings → Secrets and variables → Actions → New secret)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Yellow
Write-Host ""

# Generate Azure credentials JSON for GitHub
Write-Info "Generating AZURE_CREDENTIALS service principal..."
Write-Warn "If this fails due to permissions, ask your Azure admin to run:"
Write-Warn "  az ad sp create-for-rbac --name cip-rag-sp --role contributor --scopes /subscriptions/<sub-id>/resourceGroups/$ResourceGroup --sdk-auth"

$subscriptionId = az account show --query id --output tsv

Write-Host ""
Write-Host "  Secret Name         | Value" -ForegroundColor Cyan
Write-Host "  --------------------|------------------------------------------------" -ForegroundColor Cyan
Write-Host "  ACR_PASSWORD        | $AcrPassword" -ForegroundColor White
Write-Host "  GROQ_API_KEY        | $GroqApiKey" -ForegroundColor White
Write-Host "  CD_WEBHOOK_URL      | $cdWebhookUrl" -ForegroundColor White
Write-Host ""
Write-Host "  For AZURE_CREDENTIALS, run:" -ForegroundColor Cyan
Write-Host "  az ad sp create-for-rbac --name cip-rag-sp --role contributor \" -ForegroundColor White
Write-Host "    --scopes /subscriptions/$subscriptionId/resourceGroups/$ResourceGroup \" -ForegroundColor White
Write-Host "    --sdk-auth" -ForegroundColor White
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Next Steps:" -ForegroundColor Green
Write-Host "  1. Add the secrets above to GitHub" -ForegroundColor White
Write-Host "  2. Push a commit touching services/rag/ or docker/Dockerfile.rag-azure" -ForegroundColor White
Write-Host "  3. Watch the 'Deploy RAG to Azure' workflow in GitHub Actions" -ForegroundColor White
Write-Host "  4. Verify: curl https://$AppUrl/health" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green
