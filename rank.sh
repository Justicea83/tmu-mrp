#!/bin/bash

# =============================================================================
# RESUME-JOB RANKING SYSTEM - CONFIGURATION SCRIPT
# =============================================================================
# 
# This script runs the multi-dimensional resume-job ranking system with 
# configurable parameters for research and analysis purposes.
#
# BASIC USAGE:
#   ./rank.sh                    # Run with current configuration
#   chmod +x rank.sh             # Make executable if needed
#
# =============================================================================
# AVAILABLE PARAMETERS & EXAMPLES
# =============================================================================

# -----------------------------------------------------------------------------
# 📊 DATA CONFIGURATION
# -----------------------------------------------------------------------------
# --resumes-file PATH           # Path to resumes CSV file
#                               # Default: datasets/resumes_final.csv
#                               # Example: --resumes-file datasets/my_resumes.csv
#
# --jobs-file PATH              # Path to job descriptions CSV file  
#                               # Default: datasets/job_descriptions.csv
#                               # Example: --jobs-file datasets/my_jobs.csv
#
# --num-resumes N               # Number of resumes to process
#                               # Default: 1
#                               # Example: --num-resumes 100
#
# --num-jobs N                  # Number of job descriptions to process
#                               # Default: 1  
#                               # Example: --num-jobs 10

# -----------------------------------------------------------------------------
# 🎯 FILTERING & CATEGORY OPTIONS
# -----------------------------------------------------------------------------
# --resume-categories CAT1 CAT2 # Filter by specific resume categories
#                               # Available: INFORMATION-TECHNOLOGY AUTOMOBILE HR
#                               # Example: --resume-categories INFORMATION-TECHNOLOGY HR
#
# --exclude-resume-categories   # Exclude specific categories
#                               # Example: --exclude-resume-categories AUTOMOBILE
#
# --job-keywords WORD1 WORD2    # Filter jobs by keywords in position titles
#                               # Example: --job-keywords developer engineer manager
#
# --balanced-categories         # Enable balanced sampling across categories (default: True)
# --no-balanced-categories      # Disable balanced category sampling
#
# --category-analysis           # Enable detailed category performance analysis

# -----------------------------------------------------------------------------
# ⚖️ SCORING WEIGHTS CONFIGURATION
# -----------------------------------------------------------------------------
# --general-weight FLOAT        # Weight for general semantic matching
#                               # Default: 1.0
#                               # Example: --general-weight 2.5
#
# --skills-weight FLOAT         # Weight for skills matching
#                               # Default: 1.0
#                               # Example: --skills-weight 3.0
#
# --experience-weight FLOAT     # Weight for experience matching
#                               # Default: 1.0
#                               # Example: --experience-weight 1.5
#
# --location-weight FLOAT       # Weight for location matching
#                               # Default: 1.0
#                               # Example: --location-weight 0.5
#
# --education-weight FLOAT      # Weight for education matching
#                               # Default: 1.0
#                               # Example: --education-weight 2.0
#
# NOTE: Weights are automatically normalized to sum to 1.0

# -----------------------------------------------------------------------------
# 🤖 MODEL CONFIGURATION
# -----------------------------------------------------------------------------
# --general-model MODEL_NAME    # Sentence transformer for general matching
#                               # Default: careerbert
#                               # Examples:
#                               #   --general-model sentence-transformers/all-mpnet-base-v2
#                               #   --general-model sentence-transformers/all-MiniLM-L6-v2
#
# --skills-model MODEL_NAME     # Sentence transformer for skills matching
#                               # Default: careerbert
#                               # Example: --skills-model sentence-transformers/all-mpnet-base-v2
#
# --model-comparison            # Enable multi-model comparison mode
#                               # Runs ranking with multiple models for research
#
# --models-to-compare M1 M2     # List of models to compare
#                               # Default: careerbert sentence-transformers/all-mpnet-base-v2
#                               # Example: --models-to-compare careerbert sentence-transformers/all-MiniLM-L6-v2

# -----------------------------------------------------------------------------
# 📈 OUTPUT & ANALYSIS OPTIONS
# -----------------------------------------------------------------------------
# --output-file PATH            # Path to output CSV file
#                               # Default: logs/ranking_results_TIMESTAMP.csv
#                               # Example: --output-file my_results.csv
#
# --top-k N                     # Number of top matches to return per job
#                               # Default: 10
#                               # Example: --top-k 5
#
# --verbose                     # Enable verbose logging
# --verbose -v                  # Short form

# -----------------------------------------------------------------------------
# 🧠 ADVANCED FEATURES
# -----------------------------------------------------------------------------
# --explainable-ai              # Generate detailed explanations for rankings
#                               # Creates JSON files with match explanations
#
# --diversity-analysis          # Perform diversity and bias analysis
#                               # Generates bias reports and diversity metrics
#
# --learning-to-rank            # Use machine learning to improve rankings
#                               # Trains ML models and generates enhanced rankings
#
# --ltr-model-type TYPE         # Learning-to-rank model type
#                               # Options: linear, random_forest, gradient_boosting
#                               # Default: gradient_boosting

# -----------------------------------------------------------------------------
# 📝 EXAMPLE CONFIGURATIONS
# -----------------------------------------------------------------------------
#
# Basic ranking:
#   python runners/rank.py --num-jobs 5 --num-resumes 50
#
# Category comparison:
#   python runners/rank.py --resume-categories INFORMATION-TECHNOLOGY HR --category-analysis
#
# Custom weights (emphasize skills):
#   python runners/rank.py --skills-weight 3.0 --general-weight 1.0 --experience-weight 1.0 --location-weight 0.5
#
# Model comparison research:
#   python runners/rank.py --model-comparison --models-to-compare careerbert sentence-transformers/all-mpnet-base-v2
#
# Full research setup:
#   python runners/rank.py --num-jobs 10 --num-resumes 100 --resume-categories INFORMATION-TECHNOLOGY HR \
#     --category-analysis --model-comparison --top-k 10 --verbose

# =============================================================================
# CURRENT CONFIGURATION
# =============================================================================

python runners/rank.py \
  --num-jobs 1 --num-resumes 20 \
  --general-weight 8.0 \
  --skills-weight 1.0 \
  --experience-weight 1.0 \
  --location-weight 1.0 \
  --education-weight 1.0 \
  --resume-categories INFORMATION-TECHNOLOGY HR \
  --category-analysis \
  --top-k 10 \
  --model-comparison \
  --models-to-compare sentence-transformers/careerbert-jg sentence-transformers/all-mpnet-base-v2 \
  --explainable-ai \
  --diversity-analysis \
  --learning-to-rank