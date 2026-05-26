# Technology Comparative Analysis

To justify the architectural decisions for the Customer Intelligence Platform, we conducted a comparative analysis of the primary frameworks and models.

## 1. Machine Learning Classification (Conversion Prediction)

| Feature | XGBoost (Chosen) | LightGBM | Logistic Regression (Baseline) |
|---------|-----------------|----------|--------------------------------|
| **Accuracy / ROC-AUC** | Extremely High (0.90+) | Very High | Moderate (0.75+) |
| **Training Speed** | Moderate | Very Fast | Instant |
| **Handling Non-linear Data** | Excellent | Excellent | Poor |
| **Memory Footprint** | High | Low | Very Low |

**Justification:** We selected **XGBoost** because conversion likelihood heavily relies on complex, non-linear interactions between demographic (`age`, `income`) and behavioral (`num_clicks`, `recency`) features. While LightGBM is faster, XGBoost provided superior precision in our initial offline tests, and prediction latency via FastAPI is still under 50ms, which is well within our SLAs.

## 2. Vector Database (Complaint Retrieval)

| Feature | FAISS (Chosen) | Pinecone | ChromaDB |
|---------|---------------|----------|----------|
| **Hosting Model** | Local / In-memory | SaaS (Cloud API) | Local / Disk |
| **Latency** | < 10ms | ~ 100-200ms (Network) | ~ 20-50ms |
| **Cost** | Free (Open Source) | Pay-per-use / Subscription | Free |
| **Maintenance Overhead**| Requires manual index serialization | None | Low |

**Justification:** We selected **FAISS** to run in-memory within the FastAPI application state. For our dataset of ~25,000 complaints, holding the dense vectors in RAM is extremely lightweight (a few megabytes). This eliminates external network dependency and API costs associated with SaaS solutions like Pinecone, ensuring the RAG pipeline is fully self-contained and highly performant.
