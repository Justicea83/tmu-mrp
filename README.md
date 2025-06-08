# Multi-Dimensional Resume-Job Ranking System

A comprehensive AI-powered system for matching resumes to job descriptions using advanced natural language processing, multi-dimensional scoring, and machine learning models. This project provides complete pipelines for resume parsing, job analysis, and intelligent ranking with research-grade analysis capabilities.

## 🚀 Key Features

### 🤖 **AI-Powered Resume Processing**
- **GPT-4-mini Integration**: Intelligent parsing of resumes into structured JSON format
- **PII Removal**: Comprehensive removal of personally identifiable information
- **Multi-format Support**: Handles both text and HTML resume formats
- **Structured Data Extraction**: Extracts experience, education, skills, and personal information

### 📊 **Multi-Dimensional Matching Engine**
- **General Semantic Matching**: Uses sentence transformers for content similarity
- **Skills Matching**: Specialized matching for technical and professional skills
- **Experience Matching**: Analyzes work experience and career progression
- **Location Matching**: Geographic compatibility assessment
- **Weighted Scoring**: Configurable weights for different matching dimensions

### 🔬 **Research & Analysis Tools**
- **Model Comparison**: Compare CareerBERT vs general models (all-mpnet-base-v2)
- **Category Analysis**: Comprehensive performance analysis by resume categories
- **Balanced Sampling**: Ensure fair representation across resume categories
- **Statistical Analysis**: Detailed performance metrics and confidence intervals
- **Export Capabilities**: Multiple output formats for research documentation

### 🎯 **Advanced Filtering & Configuration**
- **Category Filtering**: Filter by resume categories (IT, HR, Automobile)
- **Job Keyword Filtering**: Filter jobs by position keywords
- **Flexible Weighting**: Customize importance of different matching dimensions
- **Model Selection**: Choose between different sentence transformer models
- **Batch Processing**: Efficient processing of large datasets

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key (for resume parsing)
- Git
- 4GB+ RAM recommended for model processing
- 2GB+ disk space for models and data

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd model-training
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy Language Model
```bash
python -m spacy download en_core_web_lg
```

This downloads the large English language model required for:
- Named Entity Recognition (NER) for PII removal
- Advanced text processing and entity extraction
- Resume parsing and analysis

### 5. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env
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
- `lwolfrum2/careerbert-jg` - Job/career specialized model

## 📁 Project Structure

```
model-training/
├── core/
│   ├── matching_engine/          # Multi-dimensional matching system
│   │   ├── base.py              # Base matching engine
│   │   ├── general.py           # General semantic matching
│   │   ├── skills.py            # Skills-specific matching
│   │   ├── experience.py        # Experience matching
│   │   ├── location.py          # Location matching
│   │   └── engine.py            # Main engine coordination
│   ├── models.py                # Pydantic data models
│   ├── resume_parser.py         # AI-powered resume parsing
│   ├── openai/                  # OpenAI integration
│   └── utils.py                 # Utility functions
├── runners/
│   └── rank.py                  # Main ranking script
├── datasets/                    # Data files
│   ├── resumes_final.csv        # Processed resume data
│   └── job_descriptions.csv     # Job descriptions data
├── logs/                        # Generated output files
│   ├── ranking_results_*.csv    # Ranking results
│   ├── *_analysis.csv           # Analysis files
│   └── *.log                    # Log files
├── rank.sh                      # Configuration script
├── METHODOLOGY.md               # Detailed methodology
└── README.md                    # This file
```

## 🎯 Quick Start

### Basic Resume-Job Ranking
```bash
# Run with pre-configured settings
./rank.sh

# Or run directly
python runners/rank.py --num-jobs 5 --num-resumes 20
```

### Category Comparison Analysis
```bash
python runners/rank.py \
  --resume-categories INFORMATION-TECHNOLOGY HR \
  --category-analysis \
  --num-jobs 10 --num-resumes 50
```

### Model Comparison Research
```bash
python runners/rank.py \
  --model-comparison \
  --models-to-compare careerbert sentence-transformers/all-mpnet-base-v2 \
  --num-jobs 5 --num-resumes 20
```

### Custom Weight Configuration
```bash
python runners/rank.py \
  --general-weight 2.0 \
  --skills-weight 3.0 \
  --experience-weight 1.5 \
  --location-weight 0.5 \
  --top-k 5
```

## 🔧 Configuration Options

### 📊 **Data Configuration**
- `--resumes-file`: Path to resumes CSV file
- `--jobs-file`: Path to job descriptions CSV file  
- `--num-resumes`: Number of resumes to process
- `--num-jobs`: Number of jobs to process

### 🎯 **Filtering Options**
- `--resume-categories`: Filter by categories (INFORMATION-TECHNOLOGY, HR, AUTOMOBILE)
- `--job-keywords`: Filter jobs by keywords
- `--balanced-categories`: Enable balanced category sampling
- `--category-analysis`: Enable detailed category analysis

### ⚖️ **Scoring Configuration**
- `--general-weight`: Weight for general semantic matching
- `--skills-weight`: Weight for skills matching  
- `--experience-weight`: Weight for experience matching
- `--location-weight`: Weight for location matching

### 🤖 **Model Configuration**
- `--general-model`: Model for general matching
- `--skills-model`: Model for skills matching
- `--model-comparison`: Enable model comparison mode
- `--models-to-compare`: List of models to compare

### 📈 **Output Options**
- `--output-file`: Custom output file path
- `--top-k`: Number of top matches per job
- `--verbose`: Enable detailed logging

## 📊 Understanding the Output

### Main Results File
```csv
job_id,job_position,job_company,rank,resume_id,resume_category,total_score,general_score,skills_score,experience_score,location_score
```

### Category Analysis File
Comprehensive statistics showing:
- Performance by resume category
- Score distributions and confidence intervals
- Category representation in top-k results
- Statistical significance testing

### Model Comparison Files
- **Detailed Comparison**: Side-by-side model results
- **Model Statistics**: Performance metrics per model
- **Individual Results**: Separate files per model

## 🔬 Research Applications

### Model Performance Evaluation
Compare fine-tuned models (CareerBERT) against general models:
```bash
python runners/rank.py --model-comparison --category-analysis
```

**Sample Results:**
- CareerBERT: 54.50 ± 2.37 average score
- All-MPNet-Base-v2: 42.39 ± 0.89 average score
- 28% improvement with domain-specific model

### Category Bias Analysis
Analyze how different resume categories perform:
```bash
python runners/rank.py --resume-categories INFORMATION-TECHNOLOGY HR AUTOMOBILE --balanced-categories --category-analysis
```

### Weight Sensitivity Analysis
Test different weight configurations:
```bash
# Skills-focused
python runners/rank.py --skills-weight 5.0 --general-weight 1.0

# Experience-focused  
python runners/rank.py --experience-weight 3.0 --skills-weight 1.0
```

## 🧪 Development

### Adding New Matchers
1. Create new matcher in `core/matching_engine/`
2. Inherit from `BaseMatcher`
3. Implement `compute_score()` method
4. Register in `engine.py`

### Custom Models
1. Add model to `download_sentence_transformer.py`
2. Use `--general-model` or `--skills-model` parameters
3. Models must be compatible with sentence-transformers

### Extending Analysis
1. Modify `save_results()` in `runners/rank.py`
2. Add new statistical computations
3. Export additional CSV files as needed

## 📊 Methodology

The system implements a sophisticated multi-dimensional matching approach:

### 1. **Resume Processing Pipeline**
- **GPT-4-mini Parsing**: Structured data extraction
- **PII Removal**: Privacy-preserving text cleaning
- **Normalization**: Standardized format conversion

### 2. **Multi-Dimensional Scoring**
- **General Matching**: Semantic similarity via sentence transformers
- **Skills Matching**: Exact + semantic skills comparison  
- **Experience Matching**: Career level and duration analysis
- **Location Matching**: Geographic compatibility

### 3. **Advanced Features**
- **Weighted Aggregation**: Customizable importance weighting
- **Model Comparison**: CareerBERT vs general models
- **Category Analysis**: Bias detection and performance analysis
- **Statistical Validation**: Confidence intervals and significance testing

For detailed methodology, see [METHODOLOGY.md](METHODOLOGY.md).

## 📝 Key Dependencies

```python
# Core ML & NLP
sentence-transformers>=2.2.2
torch>=2.0.0
spacy>=3.7.0

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

**Important**: After installing dependencies:
1. Download the English language model:
```bash
python -m spacy download en_core_web_lg
```

2. For advanced explainable AI features with SHAP:
```bash
pip install shap==0.43.0
```

## 🚨 Important Considerations

### **Performance & Costs**
- **Processing Time**: ~30 seconds per resume for AI parsing
- **API Costs**: GPT-4-mini usage for resume parsing (~$0.01 per resume)
- **Memory Usage**: 2-4GB RAM for model processing
- **Storage**: Models require ~2GB disk space

### **Privacy & Ethics**
- **PII Removal**: Comprehensive removal of personal identifiers
- **Bias Testing**: Category analysis helps detect algorithmic bias
- **Fair Sampling**: Balanced category sampling prevents discrimination

### **Research Validity**
- **Statistical Rigor**: Confidence intervals and significance testing
- **Reproducibility**: Deterministic processing with logging
- **Transparency**: Full methodology documentation

## 🔍 Troubleshooting

### **Common Issues**

**OpenAI API Errors**
```bash
# Check API key
cat .env | grep OPENAI_SECRET

# Verify account credits and rate limits
```

**Model Loading Failures**
```bash
# Re-download models
python -m core.preprocessors.download_sentence_transformer

# Check disk space (need 2GB+)
df -h
```

**Memory Issues**
```bash
# Reduce batch size
python runners/rank.py --num-resumes 10 --num-jobs 5

# Monitor memory usage
htop
```

**Poor Matching Results**
```bash
# Try different models
python runners/rank.py --general-model sentence-transformers/all-mpnet-base-v2

# Adjust weights
python runners/rank.py --skills-weight 2.0 --general-weight 1.0
```

## 📄 Citation

If you use this system in research, please cite:

```bibtex
@software{resume_job_ranking_system,
  title={Multi-Dimensional Resume-Job Ranking System},
  author={[Your Name]},
  year={2024},
  url={[Repository URL]}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-matcher`)
3. Commit changes (`git commit -am 'Add new matching dimension'`)
4. Push branch (`git push origin feature/new-matcher`)
5. Create Pull Request

## 📞 Support

- **Documentation**: See [METHODOLOGY.md](METHODOLOGY.md) for detailed technical information
- **Issues**: Use GitHub Issues for bug reports and feature requests
- **Configuration**: See `rank.sh` for comprehensive parameter documentation

## 📄 License

[Add your license information here]

## 🎓 Enhanced Education Matching with Hugging Face Integration

The system now includes **comprehensive education field mappings** powered by Hugging Face datasets and advanced local mappings:

### 🔥 **NEW: Automatic Hugging Face Dataset Integration**
The education matcher now **automatically loads** from Hugging Face datasets by default:
- **[meliascosta/wiki_academic_subjects](https://huggingface.co/datasets/meliascosta/wiki_academic_subjects)**: 64k academic subject hierarchies from Wikipedia
- **Hierarchical field mappings** (e.g., `["Humanities", "Philosophy", "Ethics"]`)
- **Automatic fallback** to comprehensive local mappings if datasets unavailable

### Field Categories Covered

**From Hugging Face Wiki Academic Subjects (loaded by default):**
- **Humanities**: Philosophy, History, Literature, Linguistics, etc.
- **Natural Sciences**: Physics, Chemistry, Biology, Earth Sciences, etc.
- **Social Sciences**: Psychology, Sociology, Anthropology, Economics, etc.
- **Applied Sciences**: Engineering, Medicine, Agriculture, etc.
- **Formal Sciences**: Mathematics, Computer Science, Logic, etc.
- **Interdisciplinary**: Environmental Studies, Cognitive Science, etc.

**Plus comprehensive local mappings:**
- **Technology** (23 fields): Computer Science, AI, Cybersecurity, etc.
- **Business** (23 fields): Management, Finance, Marketing, etc.
- **Engineering** (18 fields): Mechanical, Electrical, Civil, etc.
- **Science** (20 fields): Mathematics, Biology, Chemistry, etc.
- **Healthcare** (22 fields): Medicine, Nursing, Public Health, etc.
- **Education** (14 fields): Teaching, Pedagogy, Curriculum, etc.
- **Social Sciences** (15 fields): Psychology, Sociology, etc.
- **Arts & Design** (15 fields): Graphic Design, Architecture, etc.
- **Legal** (12 fields): Law, Constitutional, Corporate, etc.
- **Agriculture** (11 fields): Agricultural Science, Forestry, etc.

**Total**: **64k+ Wikipedia academic subjects + 177 comprehensive local mappings**

### Degree Level Hierarchy
- **PhD/Doctorate** (Level 5): PhD, Doctor, Doctoral
- **Master's** (Level 4): Master's, MBA, MS, MA, MSc
- **Bachelor's** (Level 3): Bachelor's, BS, BA, BSc
- **Associate** (Level 2): Associate, AA, AS
- **Certificate** (Level 1): Diploma, Certificate, Certification

### Advanced Dataset Integration

The system automatically attempts to load from multiple datasets:

#### Primary: Wiki Academic Subjects (loaded by default) 
```python
# Automatically loaded when system starts
# No configuration required!
```

#### Optional: Additional Dataset Updates
```python
from core.matching_engine.education import EducationMatcher

# System auto-loads wiki_academic_subjects
matcher = EducationMatcher(match)

# Optionally load additional datasets
matcher.update_field_mappings_from_dataset("millawell/wikipedia_field_of_science")
```

#### Available Additional Datasets
- **[millawell/wikipedia_field_of_science](https://huggingface.co/datasets/millawell/wikipedia_field_of_science)**: 304k scientific field taxonomies
  - Structure: `token` (list), `label` (hierarchical list like `["Humanities", "Philosophy", "Social philosophy"]`)
  - Perfect complement to the wiki_academic_subjects dataset
- **[jacob-hugging-face/job-descriptions](https://huggingface.co/datasets/jacob-hugging-face/job-descriptions)**: 853 job descriptions with field classifications
- **Custom datasets**: Extensible framework for additional academic datasets with auto-detection of column formats

### Education Matching Features
1. **Degree Level Alignment**: Matches candidate education level to job requirements
2. **Field Relevance Scoring**: Uses comprehensive mappings to match study fields
3. **Experience-Education Balance**: Higher degrees can compensate for less experience
4. **Keyword Matching**: Enhanced with specialized field keywords
5. **Comprehensive Coverage**: 177 field mappings across 10 major categories 