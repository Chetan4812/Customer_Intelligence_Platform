# Customer Intelligence Platform Architecture

This diagram illustrates the dual-service architecture built for Week 13, including the Campaign Conversion Prediction (ML) service and the Complaint Intelligence (RAG) service.

```mermaid
graph TD
    Client[Client Application / API Consumer]
    
    subrange API Gateway
        Nginx[Nginx Reverse Proxy / Load Balancer]
    end

    subgraph "Conversion Service (FastAPI)"
        ConvAPI[POST /predict]
        XGB[XGBoost Model]
        ConvAPI --> XGB
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
    Nginx -->|Route: /conversion| ConvAPI
    Nginx -->|Route: /rag| RAGAPI
    
    ConvAPI -.->|Telemetry & Logs| AzureMonitor
    RAGAPI -.->|Telemetry & Logs| AzureMonitor

    classDef service fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef datastore fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef monitor fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    
    class ConvAPI,RAGAPI service;
    class FAISS,XGB datastore;
    class AzureMonitor monitor;
```

## Components
1. **Conversion Service**: Hosts the trained XGBoost model predicting marketing conversion probability.
2. **RAG Service**: Hosts the FAISS vector database and Llama2 inference engine for semantically searching complaints and generating conversational answers.
3. **Monitoring**: Azure Application Insights integrated for logging and performance metrics.
