@echo off
REM =============================================================================
REM RESUME-JOB RANKING SYSTEM - CONFIGURATION SCRIPT (WINDOWS)
REM =============================================================================
REM 
REM This script runs the multi-dimensional resume-job ranking system with 
REM configurable parameters for research and analysis purposes.
REM
REM BASIC USAGE:
REM   rank.bat                    # Run with current configuration
REM   .\rank.bat                  # Run from PowerShell
REM
REM =============================================================================
REM AVAILABLE PARAMETERS & EXAMPLES
REM =============================================================================

REM -----------------------------------------------------------------------------
REM 📊 DATA CONFIGURATION
REM -----------------------------------------------------------------------------
REM --resumes-file PATH           # Path to resumes CSV file
REM                               # Default: datasets/resumes_final.csv
REM                               # Example: --resumes-file datasets\my_resumes.csv
REM
REM --jobs-file PATH              # Path to job descriptions CSV file  
REM                               # Default: datasets/job_descriptions.csv
REM                               # Example: --jobs-file datasets\my_jobs.csv
REM
REM --num-resumes N               # Number of resumes to process
REM                               # Default: 1
REM                               # Example: --num-resumes 100
REM
REM --num-jobs N                  # Number of job descriptions to process
REM                               # Default: 1  
REM                               # Example: --num-jobs 10

REM -----------------------------------------------------------------------------
REM 🎯 FILTERING & CATEGORY OPTIONS
REM -----------------------------------------------------------------------------
REM --resume-categories CAT1 CAT2 # Filter by specific resume categories
REM                               # Available: INFORMATION-TECHNOLOGY AUTOMOBILE HR
REM                               # Example: --resume-categories INFORMATION-TECHNOLOGY HR
REM
REM --exclude-resume-categories   # Exclude specific categories
REM                               # Example: --exclude-resume-categories AUTOMOBILE
REM
REM --job-keywords WORD1 WORD2    # Filter jobs by keywords in position titles
REM                               # Example: --job-keywords developer engineer manager
REM
REM --balanced-categories         # Enable balanced sampling across categories (default: True)
REM --no-balanced-categories      # Disable balanced category sampling
REM
REM --category-analysis           # Enable detailed category performance analysis

REM -----------------------------------------------------------------------------
REM ⚖️ SCORING WEIGHTS CONFIGURATION
REM -----------------------------------------------------------------------------
REM --general-weight FLOAT        # Weight for general semantic matching
REM                               # Default: 1.0
REM                               # Example: --general-weight 2.5
REM
REM --skills-weight FLOAT         # Weight for skills matching
REM                               # Default: 1.0
REM                               # Example: --skills-weight 3.0
REM
REM --experience-weight FLOAT     # Weight for experience matching
REM                               # Default: 1.0
REM                               # Example: --experience-weight 1.5
REM
REM --location-weight FLOAT       # Weight for location matching
REM                               # Default: 1.0
REM                               # Example: --location-weight 0.5
REM
REM --education-weight FLOAT      # Weight for education matching
REM                               # Default: 1.0
REM                               # Example: --education-weight 2.0
REM
REM NOTE: Weights are automatically normalized to sum to 1.0

REM -----------------------------------------------------------------------------
REM 🤖 MODEL CONFIGURATION
REM -----------------------------------------------------------------------------
REM --general-model MODEL_NAME    # Sentence transformer for general matching
REM                               # Default: careerbert
REM                               # Examples:
REM                               #   --general-model sentence-transformers/all-mpnet-base-v2
REM                               #   --general-model sentence-transformers/all-MiniLM-L6-v2
REM
REM --skills-model MODEL_NAME     # Sentence transformer for skills matching
REM                               # Default: careerbert
REM                               # Example: --skills-model sentence-transformers/all-mpnet-base-v2
REM
REM --model-comparison            # Enable multi-model comparison mode
REM                               # Runs ranking with multiple models for research
REM
REM --models-to-compare M1 M2     # List of models to compare
REM                               # Default: careerbert sentence-transformers/all-mpnet-base-v2
REM                               # Example: --models-to-compare careerbert sentence-transformers/all-MiniLM-L6-v2

REM -----------------------------------------------------------------------------
REM 📈 OUTPUT & ANALYSIS OPTIONS
REM -----------------------------------------------------------------------------
REM --output-file PATH            # Path to output CSV file
REM                               # Default: logs\ranking_results_TIMESTAMP.csv
REM                               # Example: --output-file my_results.csv
REM
REM --top-k N                     # Number of top matches to return per job
REM                               # Default: 10
REM                               # Example: --top-k 5
REM
REM --verbose                     # Enable verbose logging
REM --verbose -v                  # Short form

REM -----------------------------------------------------------------------------
REM 🧠 ADVANCED FEATURES
REM -----------------------------------------------------------------------------
REM --explainable-ai              # Generate detailed explanations for rankings
REM                               # Creates JSON files with match explanations
REM
REM --diversity-analysis          # Perform diversity and bias analysis
REM                               # Generates bias reports and diversity metrics
REM                               # Now includes Gaucher et al. (2011) gender bias detection
REM
REM --learning-to-rank            # Use machine learning to improve rankings
REM                               # Trains ML models and generates enhanced rankings
REM
REM --ltr-model-type TYPE         # Learning-to-rank model type
REM                               # Options: linear, random_forest, gradient_boosting
REM                               # Default: gradient_boosting

REM -----------------------------------------------------------------------------
REM 📝 EXAMPLE CONFIGURATIONS
REM -----------------------------------------------------------------------------
REM
REM Basic ranking:
REM   python runners\rank.py --num-jobs 5 --num-resumes 50
REM
REM Category comparison:
REM   python runners\rank.py --resume-categories INFORMATION-TECHNOLOGY HR --category-analysis
REM
REM Custom weights (emphasize skills):
REM   python runners\rank.py --skills-weight 3.0 --general-weight 1.0 --experience-weight 1.0 --location-weight 0.5
REM
REM Model comparison research:
REM   python runners\rank.py --model-comparison --models-to-compare careerbert sentence-transformers/all-mpnet-base-v2
REM
REM Full research setup:
REM   python runners\rank.py --num-jobs 10 --num-resumes 100 --resume-categories INFORMATION-TECHNOLOGY HR ^
REM     --category-analysis --model-comparison --top-k 10 --verbose

REM =============================================================================
REM WINDOWS SYSTEM REQUIREMENTS
REM =============================================================================
REM
REM Before running this script, ensure you have:
REM 1. Python 3.8+ installed and in PATH
REM 2. All dependencies installed: pip install -r requirements.txt
REM 3. spaCy model downloaded: python -m spacy download en_core_web_lg
REM 4. Required datasets in the datasets\ directory
REM
REM To check Python installation:
REM   python --version
REM
REM To verify dependencies:
REM   python -c "import torch, transformers, sentence_transformers; print('Dependencies OK')"

echo ===============================================================================
echo RESUME-JOB RANKING SYSTEM - WINDOWS VERSION
echo ===============================================================================
echo Starting multi-dimensional ranking analysis...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+ and add to PATH.
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Check if required directories exist
if not exist "datasets" (
    echo ERROR: datasets directory not found. Please ensure datasets are available.
    pause
    exit /b 1
)

if not exist "logs" mkdir logs

echo Running ranking analysis with current configuration...
echo.

REM =============================================================================
REM CURRENT CONFIGURATION
REM =============================================================================

python runners\rank.py ^
  --num-jobs 1 --num-resumes 3 ^
  --general-weight 8.0 ^
  --skills-weight 1.0 ^
  --experience-weight 1.0 ^
  --location-weight 1.0 ^
  --education-weight 1.0 ^
  --resume-categories INFORMATION-TECHNOLOGY HR ^
  --category-analysis ^
  --top-k 10 ^
  --model-comparison ^
  --models-to-compare sentence-transformers/careerbert-jg sentence-transformers/all-mpnet-base-v2 ^
  --explainable-ai ^
  --diversity-analysis ^
  --learning-to-rank

REM Check if the Python script completed successfully
if errorlevel 1 (
    echo.
    echo ERROR: Ranking analysis failed. Please check the error messages above.
    echo.
    echo Common solutions:
    echo 1. Install missing dependencies: pip install -r requirements.txt
    echo 2. Download spaCy model: python -m spacy download en_core_web_lg
    echo 3. Check that datasets exist in the datasets\ directory
    echo 4. Ensure you have sufficient disk space and memory
    echo.
    pause
    exit /b 1
) else (
    echo.
    echo ===============================================================================
    echo RANKING ANALYSIS COMPLETED SUCCESSFULLY
    echo ===============================================================================
    echo.
    echo Results have been saved to the logs\ directory:
    echo - Ranking results: logs\ranking_results_*.csv
    echo - Diversity analysis: logs\diversity_analysis_*.json
    echo - Bias reports: logs\bias_report_*.txt
    echo - ML rankings: logs\ml_ranking_results_*.csv
    echo - Feature importance: logs\feature_importance_*.txt
    echo - Explanations: logs\explanations_*.json
    echo.
    echo Enhanced Features Included:
    echo ✓ 5-dimensional matching (General, Skills, Experience, Location, Education)
    echo ✓ Education Matcher with 177 field mappings
    echo ✓ Model comparison (CareerBERT vs All-MPNet)
    echo ✓ SHAP-enhanced Explainable AI
    echo ✓ Learning-to-Rank with adaptive cross-validation
    echo ✓ Comprehensive diversity analytics
    echo ✓ Gaucher et al. (2011) gender bias detection
    echo.
)

echo Press any key to view the logs directory...
pause >nul
start explorer logs

exit /b 0 