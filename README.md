# Multi-Dimensional Resume-Job Ranking System

A comprehensive AI-powered system for matching resumes to job descriptions using advanced natural language processing, multi-dimensional scoring, and machine learning models. This project provides complete pipelines for resume parsing, job analysis, and intelligent ranking with research-grade analysis capabilities.

## 🚀 Key Features

### 🤖 **AI-Powered Resume Processing**
- **GPT-4-mini Integration**: Intelligent parsing of resumes into structured JSON format
- **PII Removal**: Comprehensive removal of personally identifiable information
- **Multi-format Support**: Handles both text and HTML resume formats
- **Structured Data Extraction**: Extracts experience, education, skills, and personal information

### 📊 **Multi-Dimensional Matching Engine**
- **5-Dimensional Scoring**: General, Skills, Experience, Location, Education matching
- **Advanced Education Matcher**: 177 curated field mappings + 64k Hugging Face academic subjects
- **Skills Matching**: Specialized matching for technical and professional skills
- **Experience Matching**: Analyzes work experience and career progression
- **Location Matching**: Geographic compatibility assessment with semantic similarity
- **Weighted Scoring**: Configurable weights for different matching dimensions

### 🔬 **Research & Analysis Tools**
- **Model Comparison**: Compare CareerBERT vs general models (all-mpnet-base-v2)
- **SHAP-Enhanced Explainable AI**: Feature contribution analysis and what-if scenarios
- **Learning-to-Rank**: Machine learning models with adaptive cross-validation
- **Category Analysis**: Comprehensive performance analysis by resume categories
- **Diversity Analytics**: Bias detection and gender representation analysis
- **🆕 Gaucher et al. (2011) Gender Bias Detection**: Research-grade analysis of gender-coded language in job descriptions

### 🌍 **Cross-Platform Support**
- **Linux/macOS**: `rank.sh` shell script
- **Windows**: `rank.bat` batch file + `rank.ps1` PowerShell script
- **Enhanced Windows Integration**: Parameter validation, dependency checking, colorized output

### 🎯 **Advanced Features**
- **Statistical Analysis**: Confidence intervals, significance testing, correlation analysis
- **Bias Mitigation**: Gender bias detection using Gaucher et al. methodology
- **Export Capabilities**: Multiple output formats for research documentation
- **Batch Processing**: Efficient processing of large datasets

## 📋 Prerequisites

- **Python 3.8+** (Python 3.11+ recommended)
- **OpenAI API key** (for resume parsing)
- **Git**
- **4GB+ RAM** recommended for model processing
- **2GB+ disk space** for models and data

### Platform-Specific Requirements

#### **Windows**
- Windows 10+ or Windows Server 2019+
- PowerShell 5.1+ (for `rank.ps1`)
- Command Prompt (for `rank.bat`)

#### **Linux/macOS**
- Bash shell
- Standard UNIX utilities

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd model-training
```

### 2. Create Virtual Environment

#### **Linux/macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

#### **Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### **Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

⚠️ **Important**: This downloads the large English language model (774MB) required for:
- Named Entity Recognition (NER) for PII removal
- Advanced text processing and entity extraction
- Resume parsing and analysis

### 5. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env  # Linux/macOS
copy .env.example .env  # Windows
```

Edit the `.env` file and add your OpenAI API key:
```env
OPENAI_SECRET=sk-your-actual-openai-api-key-here
```

### 6. Download Required Models
Download the sentence transformer models:
```bash
python -m core.preprocessors.download_sentence_transformer
```

This downloads:
- `sentence-transformers/all-mpnet-base-v2` - General purpose model
- `sentence-transformers/careerbert-jg` - Job/career specialized model

## 🎯 Quick Start

### **Linux/macOS**
```bash
# Make executable and run
chmod +x rank.sh
./rank.sh
```

### **Windows - Batch File (Simple)**
```cmd
# Double-click rank.bat or run from Command Prompt
rank.bat
```

### **Windows - PowerShell (Recommended)**
```powershell
# Basic run
.\rank.ps1

# Environment check only
.\rank.ps1 -CheckDependencies

# Custom configuration with auto-open results
.\rank.ps1 -NumJobs 5 -NumResumes 20 -OpenResults

# Full research setup
.\rank.ps1 -NumJobs 10 -NumResumes 100 -ModelComparison -ExplainableAI -DiversityAnalysis -LearningToRank -Verbose -OpenResults
```

### **Direct Python Execution (All Platforms)**
```bash
# Basic resume-job ranking
python runners/rank.py --num-jobs 5 --num-resumes 20

# Category comparison analysis
python runners/rank.py --resume-categories INFORMATION-TECHNOLOGY HR --category-analysis

# Model comparison research
python runners/rank.py --model-comparison --explainable-ai --diversity-analysis
```

## 📁 Project Structure

```
model-training/
├── core/
│   ├── matching_engine/          # Multi-dimensional matching system
│   │   ├── base.py              # Base matching engine
│   │   ├── general.py           # General semantic matching
│   │   ├── skills.py            # Skills-specific matching
│   │   ├── experience.py        # Experience matching
│   │   ├── location.py          # Location matching with semantic similarity
│   │   ├── education.py         # 🆕 Advanced education matching with 177 field mappings
│   │   └── engine.py            # Main engine coordination
│   ├── explainable_ai.py        # 🆕 SHAP-enhanced explainable AI
│   ├── learning_to_rank.py      # 🆕 Learning-to-rank ML models
│   ├── diversity_analytics.py   # 🆕 Comprehensive bias analysis + Gaucher et al.
│   ├── models.py                # Pydantic data models
│   ├── resume_parser.py         # AI-powered resume parsing
│   ├── openai/                  # OpenAI integration
│   └── utils.py                 # Utility functions
├── runners/
│   └── rank.py                  # Main ranking script with advanced features
├── datasets/                    # Data files
│   ├── resumes_final.csv        # Processed resume data
│   └── job_descriptions.csv     # Job descriptions data
├── logs/                        # Generated output files
│   ├── ranking_results_*.csv    # Ranking results
│   ├── diversity_analysis_*.json # 🆕 Comprehensive diversity analysis
│   ├── bias_report_*.txt        # 🆕 Bias reports with gender-coded language analysis
│   ├── explanations_*.json      # 🆕 SHAP-enhanced explanations
│   ├── ml_ranking_results_*.csv # 🆕 Learning-to-rank results
│   └── feature_importance_*.txt # 🆕 Feature importance reports
├── rank.sh                      # Linux/macOS configuration script
├── rank.bat                     # 🆕 Windows batch file
├── rank.ps1                     # 🆕 Windows PowerShell script (recommended)
├── METHODOLOGY.md               # Detailed methodology
└── README.md                    # This file
```

## 🔧 Configuration Options

### 📊 **Data Configuration**
- `--resumes-file`: Path to resumes CSV file
- `--jobs-file`: Path to job descriptions CSV file  
- `--num-resumes`: Number of resumes to process
- `--num-jobs`: Number of jobs to process

### 🎯 **Filtering Options**
- `--resume-categories`: Filter by categories (INFORMATION-TECHNOLOGY, HR, AUTOMOBILE)
- `--exclude-resume-categories`: Exclude specific categories
- `--job-keywords`: Filter jobs by keywords
- `--balanced-categories`: Enable balanced category sampling
- `--category-analysis`: Enable detailed category analysis

### ⚖️ **Scoring Configuration**
- `--general-weight`: Weight for general semantic matching (default: 8.0)
- `--skills-weight`: Weight for skills matching (default: 1.0)
- `--experience-weight`: Weight for experience matching (default: 1.0)
- `--location-weight`: Weight for location matching (default: 1.0)
- `--education-weight`: Weight for education matching (default: 1.0)

### 🤖 **Model Configuration**
- `--general-model`: Model for general matching
- `--skills-model`: Model for skills matching
- `--model-comparison`: Enable model comparison mode
- `--models-to-compare`: List of models to compare

### 🧠 **Advanced Features**
- `--explainable-ai`: Generate SHAP-enhanced explanations
- `--diversity-analysis`: Perform comprehensive bias analysis (includes Gaucher et al.)
- `--learning-to-rank`: Use ML models for ranking improvement
- `--ltr-model-type`: Learning-to-rank model (linear, random_forest, gradient_boosting)

### 📈 **Output Options**
- `--output-file`: Custom output file path
- `--top-k`: Number of top matches per job
- `--verbose`: Enable detailed logging

## 📊 Understanding the Output

### **Main Results File**
```csv
job_id,job_position,job_company,rank,resume_id,resume_category,total_score,general_score,skills_score,experience_score,location_score,education_score
```

### **🆕 Diversity Analysis (JSON)**
```json
{
  "summary": {
    "total_candidates": 100,
    "gender_diversity_index": 0.87,
    "diversity_assessment": "High Diversity"
  },
  "gender_coded_language": {
    "methodology": "Gaucher et al. (2011)",
    "overall_statistics": {
      "masculine_bias_percentage": 15.0,
      "feminine_bias_percentage": 8.0,
      "neutral_percentage": 77.0
    },
    "job_analyses": [
      {
        "job_title": "Software Engineer",
        "gender_polarity": 2,
        "bias_classification": "Moderate Masculine Bias",
        "masculine_words_found": ["competitive", "dominant"],
        "recommendations": ["Replace 'competitive' with 'collaborative'"]
      }
    ],
    "bias_assessment": "Low Bias Risk"
  }
}
```

### **🆕 SHAP Explanations (JSON)**
```json
{
  "rank": 1,
  "resume_id": "12345",
  "job_position": "Data Scientist",
  "explanation": {
    "feature_contributions": {
      "general_score": 0.35,
      "skills_score": 0.40,
      "experience_score": 0.15
    },
    "what_if_analysis": {
      "if_skills_improved_10%": {
        "new_total_score": 87.5,
        "rank_change": "+2 positions"
      }
    }
  }
}
```

## 🧪 Research Applications

### **🆕 Gender Bias Analysis**
Analyze job descriptions for gender-coded language using Gaucher et al. (2011) methodology:

```bash
# Basic diversity analysis
python runners/rank.py --diversity-analysis

# Focus on bias detection
python runners/rank.py --diversity-analysis --num-jobs 20
```

**Sample Output:**
- **Masculine Bias Detected**: 15% of jobs contain masculine-coded words
- **Bias Classification**: "Competitive", "dominant", "aggressive" language detected
- **Recommendations**: Replace biased language with neutral alternatives

### **Model Performance Evaluation**
Compare fine-tuned models (CareerBERT) against general models:

```bash
python runners/rank.py --model-comparison --category-analysis
```

**Sample Results:**
- **CareerBERT**: 59.00 ± 2.37 average score
- **All-MPNet-Base-v2**: 31.34 ± 0.89 average score
- **88% improvement** with domain-specific model

### **🆕 Explainable AI Analysis**
Generate detailed explanations with SHAP:

```bash
python runners/rank.py --explainable-ai --num-jobs 5
```

**Features:**
- **Feature Contribution Analysis**: Which dimensions drive rankings
- **What-if Scenarios**: Impact of changing candidate profiles
- **Global Feature Importance**: Overall system behavior insights

### **🆕 Learning-to-Rank Enhancement**
Use machine learning to improve ranking quality:

```bash
python runners/rank.py --learning-to-rank --ltr-model-type gradient_boosting
```

**Capabilities:**
- **Adaptive Cross-Validation**: Works with any dataset size
- **Multiple ML Models**: Linear, Random Forest, Gradient Boosting
- **Feature Importance**: Identifies most important ranking factors

## 🔍 Enhanced Features Deep Dive

### **🆕 Gaucher et al. (2011) Gender Bias Detection**

Based on the landmark research: *"Evidence That Gendered Wording in Job Advertisements Exists and Sustains Gender Inequality"*

#### **Methodology:**
- **42 Masculine-Coded Words**: competitive, aggressive, dominant, decisive, etc.
- **39 Feminine-Coded Words**: collaborative, supportive, nurturing, empathetic, etc.
- **Gender Polarity Score**: `masculine_score - feminine_score`

#### **Classification:**
- **+3 or higher**: Strong Masculine Bias ⚠️
- **+1 to +2**: Moderate Masculine Bias
- **-1 to +1**: Gender Neutral ✅
- **-2 to -1**: Moderate Feminine Bias
- **-3 or lower**: Strong Feminine Bias ⚠️

#### **Real-World Impact:**
Used by LinkedIn, Indeed, Glassdoor, and Fortune 500 companies for bias-free job postings.

### **🆕 Advanced Education Matching**

#### **Comprehensive Field Coverage:**
- **64k+ Academic Subjects**: Automatically loaded from Hugging Face datasets
- **177 Local Mappings**: Technology, Business, Engineering, Science, Healthcare, etc.
- **Hierarchical Matching**: Field categories and subcategories
- **Degree Level Analysis**: Certificate → PhD progression

#### **Data Sources:**
- **Primary**: WikiAcademicSubjects (Hugging Face)
- **Fallback**: Comprehensive local mappings
- **Production Safeguards**: Automatic fallback if datasets unavailable

## 💻 Platform-Specific Usage

### **Windows PowerShell Examples**

```powershell
# Quick environment check
.\rank.ps1 -CheckDependencies

# Basic analysis with results viewing
.\rank.ps1 -NumJobs 5 -NumResumes 20 -OpenResults

# Research configuration
.\rank.ps1 -NumJobs 10 -NumResumes 100 -ResumeCategories "INFORMATION-TECHNOLOGY","HR" -CategoryAnalysis -ModelComparison -ExplainableAI -DiversityAnalysis -LearningToRank -Verbose

# Custom weights (emphasize skills)
.\rank.ps1 -SkillsWeight 3.0 -GeneralWeight 1.0 -ExperienceWeight 1.0 -LocationWeight 0.5

# Gender bias analysis focus
.\rank.ps1 -DiversityAnalysis -NumJobs 20 -Verbose
```

### **Linux/macOS Examples**

```bash
# Basic run with current configuration
./rank.sh

# Custom configuration (edit rank.sh or run directly)
python runners/rank.py --num-jobs 10 --num-resumes 100 --diversity-analysis --explainable-ai --learning-to-rank

# Model comparison
python runners/rank.py --model-comparison --models-to-compare sentence-transformers/careerbert-jg sentence-transformers/all-mpnet-base-v2
```

## 📝 Key Dependencies

```python
# Core ML & NLP
sentence-transformers>=2.2.2
torch>=2.0.0
spacy>=3.7.0
datasets>=2.14.0  # 🆕 For Hugging Face integration

# Advanced Features
shap==0.43.0      # 🆕 For explainable AI
scikit-learn>=1.3.0  # 🆕 For learning-to-rank

# Data Processing  
pandas>=2.0.0
numpy>=1.24.0

# AI Integration
openai>=1.0.0
pydantic>=2.0.0

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
```

**Critical Installation Steps:**

1. **Download spaCy model:**
```bash
python -m spacy download en_core_web_lg
```

2. **Verify installation (Windows PowerShell):**
```powershell
.\rank.ps1 -CheckDependencies
```

3. **Test basic functionality:**
```bash
python runners/rank.py --num-jobs 1 --num-resumes 2
```

## 🚨 Important Considerations

### **Performance & Costs**
- **Processing Time**: ~30 seconds per resume for AI parsing
- **API Costs**: GPT-4-mini usage (~$0.01 per resume)
- **Memory Usage**: 2-4GB RAM for model processing
- **Storage**: Models require ~2GB disk space

### **Privacy & Ethics**
- **PII Removal**: Comprehensive removal of personal identifiers
- **🆕 Gender Bias Detection**: Research-grade analysis prevents discriminatory language
- **Fair Sampling**: Balanced category sampling prevents algorithmic bias
- **Transparency**: Full explainability with SHAP analysis

### **Research Validity**
- **Statistical Rigor**: Confidence intervals and significance testing
- **Reproducibility**: Deterministic processing with comprehensive logging
- **Peer-Reviewed Methods**: Implements Gaucher et al. (2011) methodology
- **Academic Standards**: Suitable for research publication

## 🔍 Troubleshooting

### **Windows-Specific Issues**

**PowerShell Execution Policy**
```powershell
# If PowerShell blocks script execution
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Python Not Found (Windows)**
```cmd
# Check if Python is in PATH
python --version

# If not found, reinstall Python with "Add to PATH" option
# Or add manually: System Properties > Environment Variables
```

### **Cross-Platform Issues**

**OpenAI API Errors**
```bash
# Check API key configuration
cat .env | grep OPENAI_SECRET  # Linux/macOS
type .env | findstr OPENAI_SECRET  # Windows

# Verify account credits and rate limits at platform.openai.com
```

**Model Loading Failures**
```bash
# Re-download models
python -m core.preprocessors.download_sentence_transformer

# Check disk space (need 2GB+)
df -h          # Linux/macOS
dir C:\ /s     # Windows
```

**Memory Issues**
```bash
# Reduce dataset size
python runners/rank.py --num-resumes 10 --num-jobs 5

# Monitor memory usage
htop           # Linux/macOS
taskmgr        # Windows
```

**🆕 Dependency Issues**
```powershell
# Windows: Comprehensive environment check
.\rank.ps1 -CheckDependencies

# Linux/macOS: Manual check
python -c "import torch, transformers, sentence_transformers, spacy, datasets, shap; print('All dependencies OK')"
```

## 📄 Citation

If you use this system in research, please cite:

```bibtex
@software{resume_job_ranking_system,
  title={Multi-Dimensional Resume-Job Ranking System with Gender Bias Detection},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]},
  note={Implements Gaucher et al. (2011) gender bias detection methodology}
}

@article{gaucher2011evidence,
  title={Evidence that gendered wording in job advertisements exists and sustains gender inequality},
  author={Gaucher, Danielle and Friesen, Justin and Kay, Aaron C},
  journal={Journal of personality and social psychology},
  volume={101},
  number={1},
  pages={109},
  year={2011},
  publisher={American Psychological Association}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push branch (`git push origin feature/new-feature`)
5. Create Pull Request

**Areas for contribution:**
- Additional bias detection methods
- New matching dimensions
- Enhanced explainability features
- Performance optimizations
- Cross-platform improvements

## 📞 Support

- **Documentation**: See [METHODOLOGY.md](METHODOLOGY.md) for detailed technical information
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Configuration**: 
  - See `rank.sh` (Linux/macOS)
  - See `rank.bat` and `rank.ps1` (Windows)
  - Use `.\rank.ps1 -CheckDependencies` for Windows diagnostics

## 📄 License

MIT License

---

## 🎉 What's New in v2.0

### **🆕 Enhanced Features**
- ✅ **Cross-Platform Support**: Windows batch + PowerShell scripts
- ✅ **Gaucher et al. Gender Bias Detection**: Research-grade bias analysis
- ✅ **SHAP Explainable AI**: Feature contribution analysis
- ✅ **Learning-to-Rank**: Machine learning ranking enhancement
- ✅ **Advanced Education Matching**: 64k+ academic subjects + 177 local mappings
- ✅ **Comprehensive Diversity Analytics**: Gender representation + bias detection
- ✅ **Enhanced Error Handling**: Null-safe processing throughout
- ✅ **Adaptive Cross-Validation**: Works with any dataset size
- ✅ **Production Safeguards**: Robust fallback mechanisms

### **🔧 Technical Improvements**
- Better memory management for large datasets
- Improved error messages and diagnostics
- Enhanced logging and debugging capabilities
- Optimized model loading and caching
- Cross-platform path handling

### **📊 Research Capabilities**
- Publication-ready statistical analysis
- Peer-reviewed bias detection methodology
- Comprehensive explainability features
- Advanced ML model comparison
- Industry-standard diversity metrics
