# Multi-Dimensional Resume-Job Ranking System
# Production Docker Image with Advanced Features
FROM python:3.11-slim-buster

# Metadata
LABEL maintainer="Resume-Job Ranking System"
LABEL description="AI-powered resume-job matching with bias detection and explainable AI"
LABEL version="2.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    wget \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy language model (774MB - critical for NER and PII removal)
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p logs datasets models && \
    chown -R appuser:appuser /app

# Download sentence transformer models
RUN python -m core.preprocessors.download_sentence_transformer

# Set proper permissions
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create volume mount points
VOLUME ["/app/logs", "/app/datasets"]

# Environment variables for configuration
ENV PYTHONPATH=/app \
    OPENAI_SECRET="" \
    NUM_JOBS=5 \
    NUM_RESUMES=20 \
    ENABLE_DIVERSITY_ANALYSIS=false \
    ENABLE_EXPLAINABLE_AI=false \
    ENABLE_LEARNING_TO_RANK=false \
    ENABLE_MODEL_COMPARISON=false

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch, transformers, sentence_transformers, spacy, datasets, shap; print('Health check passed')" || exit 1

# Default command - run basic ranking
CMD ["python", "runners/rank.py", "--num-jobs", "5", "--num-resumes", "20"]

# Expose port for potential future web interface
EXPOSE 8000

# Add build information
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.version=$VERSION \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version="1.0" 