# 🚀 Feast RAG Pipeline Deployment Guide

## 🏗️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                          Kubernetes/OpenShift Cluster                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        feast-rag-pipeline Namespace                 │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐  │   │
│  │  │   Ingress/Route │    │  Load Balancer  │    │    External     │  │   │
│  │  │  (External      │◄──►│   (Optional)    │◄──►│     Users       │  │   │
│  │  │   Access)       │    │                 │    │                 │  │   │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────┘  │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Feast RAG       │                                                │   │
│  │  │ Service         │                                                │   │
│  │  │ (ClusterIP)     │                                                │   │
│  │  │ Port: 8000      │                                                │   │
│  │  └─────────────────┘                                                │   │
│  │           │                                                         │   │
│  │           ▼                                                         │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                         │   │
│  │  │ Feast RAG       │    │ Ollama Service  │                         │   │
│  │  │ Deployment      │◄──►│ (ClusterIP)     │                         │   │
│  │  │                 │    │ Port: 11434     │                         │   │
│  │  │ ┌─────────────┐ │    │                 │                         │   │
│  │  │ │  FastAPI    │ │    │ ┌─────────────┐ │                         │   │
│  │  │ │  Feast      │ │    │ │   Ollama    │ │                         │   │
│  │  │ │  Milvus-lite│ │    │ │   LLM API   │ │                         │   │
│  │  │ │  RAG Logic  │ │    │ │  (llama3.2) │ │                         │   │
│  │  │ └─────────────┘ │    │ │(all-MiniLM)│ │                         │   │
│  │  │                 │    │ └─────────────┘ │                         │   │
│  │  │ Resources:      │    │                 │                         │   │
│  │  │ CPU: 1-2 cores  │    │ Resources:      │                         │   │
│  │  │ RAM: 2-4 GB     │    │ CPU: 2-4 cores  │                         │   │
│  │  │ Replicas: 1-N   │    │ RAM: 4-8 GB     │                         │   │
│  │  └─────────────────┘    │ Replicas: 1     │                         │   │
│  │           │             └─────────────────┘                         │   │
│  │           │                       │                                 │   │
│  │           ▼                       ▼                                 │   │
│  │  ┌─────────────────┐    ┌─────────────────┐                         │   │
│  │  │ Persistent      │    │ Persistent      │                         │   │
│  │  │ Volume          │    │ Volume          │                         │   │
│  │  │ (Feast+Milvus)  │    │ (Ollama Models) │                         │   │
│  │  │ Size: 10Gi      │    │ Size: 20Gi      │                         │   │
│  │  └─────────────────┘    └─────────────────┘                         │   │
│  │                                                                     │   │
│  │  ┌─────────────────┐                                                │   │
│  │  │ Model Init Job  │                                                │   │
│  │  │ (Completed)     │                                                │   │
│  │  │ - Downloads     │                                                │   │
│  │  │ - Initializes   │                                                │   │
│  │  └─────────────────┘                                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

## 🛠️ Deployment Options

### 1. **Kubernetes Deployment**
```bash
# Deploy all components
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n llama-stack-rag
kubectl get services -n llama-stack-rag
kubectl get ingress -n llama-stack-rag
```

### 2. **OpenShift Deployment**
```bash
# Deploy components
oc apply -f k8s-deployment.yaml

# Create route for external access
oc expose service/rag-pipeline-service --port=8000 -n llama-stack-rag

# Get route URL
oc get route -n llama-stack-rag
```

### 3. **Docker Compose (Local)**
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 🌐 Access Methods

### **Option 1: Port Forwarding (Development)**
```bash
kubectl port-forward -n llama-stack-rag service/rag-pipeline-service 8000:8000
# Access: http://localhost:8000
```

### **Option 2: Ingress (Kubernetes)**
```bash
# Add to /etc/hosts
echo "127.0.0.1 rag-pipeline.local" | sudo tee -a /etc/hosts
# Access: http://rag-pipeline.local
```

### **Option 3: Route (OpenShift)**
```bash
# Get route URL
oc get route/rag-pipeline-service -n llama-stack-rag -o jsonpath='{.spec.host}'
# Access: https://rag-pipeline-service-llama-stack-rag.apps.cluster.com
```

## 📊 Scalability Features

### **Horizontal Pod Autoscaling**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-pipeline-hpa
  namespace: llama-stack-rag
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-pipeline-deployment
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### **Resource Scaling Guidelines**
- **RAG Pipeline**: Scale replicas based on query load
- **Ollama Service**: Keep single replica (stateful model serving)
- **Storage**: Use dynamic provisioning for auto-scaling volumes

## 🔧 Configuration

### **Environment Variables (ConfigMap)**
```yaml
# Feast RAG Pipeline Configuration
RAG_API_HOST: "0.0.0.0"
RAG_API_PORT: "8000"
RAG_OLLAMA_HOST: "ollama-service"
RAG_OLLAMA_PORT: "11434"
RAG_FEAST_REPO_PATH: "/app/feast_feature_repo"
RAG_MILVUS_URI: "/app/feast_feature_repo/data/online_store.db"
RAG_EMBEDDING_MODEL: "all-MiniLM-L6-v2"
RAG_LLM_MODEL: "llama3.2:3b"
RAG_LOG_LEVEL: "INFO"
RAG_DEBUG_MODE: "false"
RAG_ENABLE_CORS: "true"
RAG_MAX_FILE_SIZE: "104857600"  # 100MB
```

### **Resource Requirements**
| Component      | CPU        | Memory  | Storage | Notes |
|----------------|------------|---------|---------|--------|
| Feast RAG API  | 1-2 cores  | 2-4 GB  | -       | Includes Feast + Milvus-lite |
| Ollama Service | 2-4 cores  | 4-8 GB  | -       | LLM inference |
| Feast Data     | -          | -       | 10 GB   | Milvus-lite database |
| Ollama Models  | -          | -       | 20 GB   | Model storage |

## 📋 Deployment Checklist

- [ ] Kubernetes/OpenShift cluster ready
- [ ] Sufficient cluster resources available
- [ ] Storage classes configured
- [ ] Ingress controller installed (for K8s)
- [ ] Image registry accessible
- [ ] Namespace created
- [ ] Persistent volumes available

## 🔍 Monitoring & Health Checks

### **Health Endpoints**
- RAG Pipeline: `http://service:8000/health`
- Ollama Service: `http://service:11434/api/tags`

### **Checking Deployment Status**
```bash
# Check all resources
kubectl get all -n llama-stack-rag

# Check pod logs
kubectl logs -f deployment/rag-pipeline-deployment -n llama-stack-rag
kubectl logs -f deployment/ollama-deployment -n llama-stack-rag

# Check persistent volumes
kubectl get pv,pvc -n llama-stack-rag
```

## 🚨 Troubleshooting

### **Common Issues**
1. **Model Download Timeout**: Increase job timeout in model-init-job
2. **Storage Issues**: Check PV/PVC status and storage class
3. **Network Issues**: Verify service discovery and DNS resolution
4. **Resource Limits**: Check CPU/Memory limits and requests

### **Debug Commands**
```bash
# Describe problematic pods
kubectl describe pod <pod-name> -n llama-stack-rag

# Check events
kubectl get events -n llama-stack-rag --sort-by='.lastTimestamp'

# Check resource usage
kubectl top pods -n llama-stack-rag
```

## 📦 Files Structure

```
deploy/
├── README.md                 # This file
├── docker-compose.yml        # Docker Compose setup
├── docker-compose.override.yml # Local overrides
├── k8s-deployment.yaml       # Kubernetes/OpenShift deployment
├── Dockerfile               # Container image
├── nginx.conf               # Nginx configuration
└── run.sh                   # Deployment script
```

## 🎯 Production Considerations

1. **Security**: Use non-root containers, network policies
2. **Monitoring**: Implement Prometheus/Grafana monitoring
3. **Backup**: Regular backup of persistent volumes
4. **Updates**: Rolling updates for zero-downtime deployments
5. **Load Testing**: Validate performance under expected load 