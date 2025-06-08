# Docker Guide - Multi-Dimensional Resume-Job Ranking System

This guide covers Docker deployment of the AI-powered resume-job matching system with advanced bias detection and explainable AI features.

## 🚀 Quick Start

### 1. **Setup Environment**
```bash
# Copy environment template
cp docker.env.example .env

# Edit with your OpenAI API key
nano .env  # or vim .env
```

### 2. **Basic Usage**
```bash
# Run basic ranking (5 jobs, 20 resumes)
docker-compose --profile basic up

# Run with custom parameters
NUM_JOBS=10 NUM_RESUMES=50 docker-compose --profile basic up
```

### 3. **Research Configuration**
```bash
# Full research setup with all advanced features
docker-compose --profile research up
```

## 📊 Available Profiles

### **Basic Profile** (`--profile basic`)
- **Use Case**: Quick testing and basic ranking
- **Features**: Standard 5-dimensional matching
- **Resources**: 2-4GB RAM, 1-2 CPUs
- **Command**: 
```bash
docker-compose --profile basic up
```

### **Research Profile** (`--profile research`)
- **Use Case**: Academic research and publication
- **Features**: All advanced features enabled
  - Model comparison (CareerBERT vs All-MPNet)
  - SHAP explainable AI
  - Gaucher et al. gender bias detection
  - Learning-to-rank ML models
  - Category analysis
- **Resources**: 4-6GB RAM, 2-4 CPUs
- **Command**:
```bash
docker-compose --profile research up
```

### **Bias Analysis Profile** (`--profile bias-analysis`)
- **Use Case**: Gender bias detection and diversity analysis
- **Features**: 
  - Gaucher et al. (2011) methodology
  - SHAP explanations
  - Comprehensive diversity metrics
- **Resources**: 2-4GB RAM, 1-2 CPUs
- **Command**:
```bash
docker-compose --profile bias-analysis up
```

### **Model Comparison Profile** (`--profile model-comparison`)
- **Use Case**: Comparing different AI models
- **Features**: CareerBERT vs All-MPNet comparison
- **Resources**: 3-5GB RAM, 2-3 CPUs
- **Command**:
```bash
docker-compose --profile model-comparison up
```

### **Production Profile** (`--profile production`)
- **Use Case**: Production deployment
- **Features**: Optimized for reliability and scale
- **Resources**: 3-4GB RAM, 1-2 CPUs
- **Command**:
```bash
docker-compose --profile production up
```

### **Development Profile** (`--profile dev`)
- **Use Case**: Development and debugging
- **Features**: Interactive shell, live code mounting
- **Resources**: 4-6GB RAM, 2-4 CPUs
- **Command**:
```bash
docker-compose --profile dev up -d
docker exec -it resume-ranking-dev bash
```

## 🔧 Configuration

### **Environment Variables**

Configure via `.env` file or environment variables:

```bash
# Core Configuration
OPENAI_SECRET=sk-your-api-key-here
NUM_JOBS=10
NUM_RESUMES=50

# Feature Toggles
ENABLE_DIVERSITY_ANALYSIS=true
ENABLE_EXPLAINABLE_AI=true
ENABLE_LEARNING_TO_RANK=false

# Scoring Weights
GENERAL_WEIGHT=8.0
SKILLS_WEIGHT=1.0
EXPERIENCE_WEIGHT=1.0
LOCATION_WEIGHT=1.0
EDUCATION_WEIGHT=1.0
```

### **Volume Mounts**

- **`./logs`**: Output files (rankings, analysis, explanations)
- **`./datasets`**: Input data (resumes, job descriptions)
- **`resume_models`**: Downloaded AI models (persistent volume)

## 🎯 Usage Examples

### **Basic Resume Ranking**
```bash
# Simple ranking with default settings
docker-compose --profile basic up

# Custom dataset size
NUM_JOBS=20 NUM_RESUMES=100 docker-compose --profile basic up
```

### **Gender Bias Analysis**
```bash
# Focus on bias detection
docker-compose --profile bias-analysis up

# Results will include:
# - Gender-coded language analysis
# - Bias classification reports
# - Mitigation recommendations
```

### **Research with All Features**
```bash
# Complete research configuration
docker-compose --profile research up

# Generates:
# - Model comparison results
# - SHAP explanations
# - Diversity analysis
# - Learning-to-rank improvements
# - Feature importance reports
```

### **Custom Model Comparison**
```bash
# Compare specific models
MODELS_TO_COMPARE="sentence-transformers/careerbert-jg sentence-transformers/all-mpnet-base-v2" \
docker-compose --profile model-comparison up
```

### **Production Deployment**
```bash
# Production-ready configuration
docker-compose --profile production up -d

# Check health
docker-compose ps
docker-compose logs resume-ranking-production
```

## 🔍 Development Workflow

### **Interactive Development**
```bash
# Start development container
docker-compose --profile dev up -d

# Access interactive shell
docker exec -it resume-ranking-dev bash

# Run custom commands
python runners/rank.py --num-jobs 5 --diversity-analysis --verbose

# View logs in real-time
docker-compose logs -f resume-ranking-dev
```

### **Testing Changes**
```bash
# Rebuild with changes
docker-compose build

# Test specific profile
docker-compose --profile basic up

# Clean restart
docker-compose down
docker-compose --profile research up
```

## 📊 Output Files

All outputs are saved to the `./logs` directory:

### **Standard Outputs**
- `ranking_results_YYYYMMDD_HHMMSS.csv` - Main ranking results
- `category_analysis_YYYYMMDD_HHMMSS.csv` - Category performance

### **Advanced Outputs** (Research Profile)
- `diversity_analysis_YYYYMMDD_HHMMSS.json` - Bias detection results
- `explanations_YYYYMMDD_HHMMSS.json` - SHAP explanations  
- `bias_report_YYYYMMDD_HHMMSS.txt` - Gender bias analysis
- `ml_ranking_results_YYYYMMDD_HHMMSS.csv` - Learning-to-rank results
- `feature_importance_YYYYMMDD_HHMMSS.txt` - Feature importance

## 🛠️ Management Commands

### **Container Management**
```bash
# Start specific service
docker-compose up resume-ranking-basic

# Run in background
docker-compose --profile production up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# View logs
docker-compose logs resume-ranking-basic

# Follow logs
docker-compose logs -f resume-ranking-research
```

### **Image Management**
```bash
# Build image
docker-compose build

# Build with no cache
docker-compose build --no-cache

# Pull latest base images
docker-compose pull

# View images
docker images | grep resume-ranking
```

### **Volume Management**
```bash
# List volumes
docker volume ls | grep resume

# Inspect model volume
docker volume inspect resume_models

# Backup models
docker run --rm -v resume_models:/data -v $(pwd):/backup alpine tar czf /backup/models-backup.tar.gz -C /data .

# Restore models
docker run --rm -v resume_models:/data -v $(pwd):/backup alpine tar xzf /backup/models-backup.tar.gz -C /data
```

## 🚨 Troubleshooting

### **Common Issues**

**Out of Memory**
```bash
# Check container memory usage
docker stats

# Reduce dataset size
NUM_JOBS=5 NUM_RESUMES=10 docker-compose --profile basic up

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 6GB+
```

**OpenAI API Errors**
```bash
# Check API key
docker-compose exec resume-ranking-basic env | grep OPENAI

# Test API key
docker-compose exec resume-ranking-basic python -c "
from openai import OpenAI
client = OpenAI()
print('API key valid')
"
```

**Model Download Failures**
```bash
# Clear model cache and rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose --profile basic up
```

**Permission Issues**
```bash
# Fix log directory permissions
sudo chown -R $USER:$USER ./logs

# Check volume permissions
docker-compose exec resume-ranking-basic ls -la /app/logs
```

### **Performance Optimization**

**Speed up builds**
```bash
# Use BuildKit for faster builds
DOCKER_BUILDKIT=1 docker-compose build

# Parallel service startup
docker-compose up --parallel
```

**Resource monitoring**
```bash
# Monitor resource usage
docker stats

# Check container health
docker-compose ps
```

## 🔒 Security Best Practices

### **API Key Security**
- Never commit `.env` file with real API keys
- Use Docker secrets for production:
```bash
echo "sk-your-api-key" | docker secret create openai_key -
```

### **Container Security**
- Images run as non-root user (`appuser`)
- Minimal attack surface with slim base image
- No unnecessary packages installed

### **Network Security**
```bash
# Run on custom network
docker-compose up --profile production

# No exposed ports by default (CLI tool)
# Expose only if needed for web interface
```

## 📈 Scaling for Production

### **Resource Requirements**
- **Minimum**: 2GB RAM, 1 CPU, 5GB disk
- **Recommended**: 4GB RAM, 2 CPUs, 10GB disk
- **Research**: 6GB RAM, 4 CPUs, 15GB disk

### **Docker Swarm Deployment**
```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  resume-ranking:
    image: resume-ranking:latest
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
          cpus: '2.0'
      restart_policy:
        condition: on-failure
        max_attempts: 3
```

### **Kubernetes Deployment**
```yaml
# k8s-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: resume-ranking
spec:
  replicas: 2
  selector:
    matchLabels:
      app: resume-ranking
  template:
    metadata:
      labels:
        app: resume-ranking
    spec:
      containers:
      - name: resume-ranking
        image: resume-ranking:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

## 🎓 Best Practices

### **Development**
1. Use `--profile dev` for interactive development
2. Mount source code as volume for live editing
3. Use health checks to ensure container readiness

### **Testing**
1. Start with `--profile basic` for quick validation
2. Use `--profile bias-analysis` for bias testing
3. Run `--profile research` for comprehensive evaluation

### **Production**
1. Use `--profile production` for production deployments
2. Implement proper logging and monitoring
3. Use external volume mounts for data persistence
4. Regular backup of models and results

### **Security**
1. Never expose OpenAI API keys in logs
2. Use Docker secrets for sensitive configuration
3. Regularly update base images for security patches
4. Monitor container resource usage

Your Docker setup is now ready for enterprise-grade deployment of the resume-job ranking system with comprehensive bias detection and explainable AI! 🚀 