# Model Training for Resume and Job Matching

This project provides a complete pipeline for processing resumes and job descriptions using AI-powered parsing and machine learning models. It includes entity definitions, OpenAI integration for structured data extraction, and sentence transformers for semantic matching.

## 🚀 Features

- **Resume Processing**: AI-powered parsing of resume text into structured JSON format
- **Job Description Processing**: Structured entities for job postings with comprehensive metadata
- **Machine Learning Models**: Pre-configured sentence transformers for semantic analysis
- **OpenAI Integration**: Leverages GPT models for intelligent text parsing
- **Data Pipeline**: Complete preprocessing pipeline from raw CSV to structured data
- **Pydantic Entities**: Type-safe data models for resumes and job posts

## 📋 Prerequisites

- Python 3.11 or higher
- OpenAI API key
- Git

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

### 4. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Edit the `.env` file and add your OpenAI API key:
```env
OPENAI_SECRET=sk-your-actual-openai-api-key-here
```

**Important**: Never commit your actual `.env` file to version control.

### 5. Download Required Models
Download the sentence transformer models:
```bash
python -m core.preprocessors.download_sentence_transformer
```

This will download:
- `sentence-transformers/all-mpnet-base-v2` - General purpose sentence transformer
- `lwolfrum2/careerbert-jg` - Specialized model for job/career matching

## 📁 Project Structure

```
model-training/
├── core/
│   ├── entities/              # Pydantic data models
│   │   ├── resume.py         # Resume entity definition
│   │   ├── jobpost.py        # Job post entity definition
│   │   └── __init__.py
│   ├── openai/               # OpenAI integration
│   │   ├── wrapper.py        # OpenAI API wrapper
│   │   ├── post_processor.py # Response processing
│   │   └── ...
│   ├── prompt_schema/        # AI prompt schemas
│   │   ├── resume_schema.py  # Resume parsing schema
│   │   └── ...
│   ├── preprocessors/        # Data processing scripts
│   │   ├── parse_resumes_to_json.py  # Main resume parser
│   │   └── download_sentence_transformer.py
│   └── utils.py             # Utility functions
├── datasets/                 # Data files
│   ├── resumes.csv          # Raw resume data
│   ├── job_descriptions.csv # Raw job data
│   └── resumes_final.csv    # Processed resume data (generated)
├── sentence-transformers/   # Downloaded ML models
├── .env                     # Environment variables (create this)
├── .env.example            # Environment template
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🎯 Usage

### Resume Processing Pipeline

The main pipeline processes resumes from the dataset, selects specific categories, and parses them using AI:

```bash
# Process resumes (selects 80 IT, 20 Automobile, 20 HR resumes)
python -m core.preprocessors.parse_resumes_to_json

# Or use the runner script
python run_resume_parser.py
```

This will:
1. Load resumes from `datasets/resumes.csv`
2. Select 120 resumes (80 IT, 20 Automobile, 20 HR)
3. Parse each resume using OpenAI to extract structured data
4. Save results to `datasets/resumes_final.csv`

### Using the Entities

```python
from core.entities import Resume, JobPost

# Create a resume instance
resume = Resume(
    ID="123",
    Resume_str="Software Engineer with 5 years experience...",
    Resume_html="<html>...</html>",
    Category="INFORMATION-TECHNOLOGY",
    # ... other fields
)

# Create a job post instance
jobpost = JobPost(
    Position="Senior Software Engineer",
    Long_Description="We are looking for...",
    Company_Name="Tech Corp",
    # ... other fields
)
```

### Direct AI Parsing

```python
from core.preprocessors.parse_resumes_to_json import ResumeParser

parser = ResumeParser()
parsed_data = parser.parse_resume_with_ai("Your resume text here...")
print(parsed_data)  # Structured JSON output
```

## 📊 Data Schema

### Resume Entity Fields
- `ID`: Unique identifier
- `Resume_str`: Raw resume text
- `Resume_html`: HTML version of resume
- `Category`: Resume category (IT, HR, etc.)
- `hash`: Content hash
- `char_len`: Character count
- `sent_len`: Sentence count
- `type_token_ratio`: Linguistic metric
- `gender_term_count`: Gender-related terms
- `html_len`: HTML length
- `text_from_html`: Extracted text
- `html_strip_diff`: HTML processing difference

### Job Post Entity Fields
- `Position`: Job title
- `Long_Description`: Full job description
- `Company_Name`: Hiring company
- `Exp_Years`: Required experience
- `Primary_Keyword`: Main job keyword
- `English_Level`: Required English proficiency
- `Published`: Publication status
- `Long_Description_lang`: Description language
- `id`: Unique identifier
- `__index_level_0__`: Index reference
- `char_len`: Description length

## 🔧 Configuration

### Environment Variables
- `OPENAI_SECRET`: Your OpenAI API key (required)

### Model Configuration
Models are configured in `core/preprocessors/download_sentence_transformer.py`:
- **Primary Model**: `sentence-transformers/all-mpnet-base-v2`
- **Career Model**: `lwolfrum2/careerbert-jg`

### Processing Configuration
Resume selection is configured in `parse_resumes_to_json.py`:
```python
CATEGORY_COUNTS = {
    "INFORMATION-TECHNOLOGY": 80,
    "AUTOMOBILE": 20,
    "HR": 20
}
```

## 🧪 Development

### Adding New Entities
1. Create new entity file in `core/entities/`
2. Define Pydantic model with appropriate fields
3. Add import to `core/entities/__init__.py`

### Extending AI Parsing
1. Create new schema in `core/prompt_schema/`
2. Follow the pattern in `resume_schema.py`
3. Use with the `QueryEngine` class

### Custom Preprocessing
1. Add new scripts to `core/preprocessors/`
2. Follow the pattern in existing scripts
3. Use the logging utilities from `core.utils`

## 📝 Dependencies

Key dependencies include:
- `pydantic`: Data validation and parsing
- `openai`: OpenAI API integration
- `pandas`: Data manipulation
- `sentence-transformers`: ML models for text embeddings
- `torch`: PyTorch for ML operations
- `python-dotenv`: Environment variable management

## 🚨 Important Notes

1. **API Costs**: The resume parsing uses OpenAI API calls, which incur costs
2. **Rate Limits**: Be aware of OpenAI rate limits when processing large datasets
3. **Model Storage**: Downloaded models require ~1GB of disk space
4. **Processing Time**: AI parsing is time-intensive (expect several minutes for 120 resumes)

## 🆘 Troubleshooting

### Common Issues

**Import Errors**: Ensure virtual environment is activated and all dependencies installed
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**OpenAI API Errors**: Check your API key and account credits
```bash
# Verify your .env file has the correct API key
cat .env
```

**Model Download Failures**: Ensure internet connection and retry
```bash
python -m core.preprocessors.download_sentence_transformer
```

**Memory Issues**: For large datasets, consider processing in smaller batches

## 📄 License

[Add your license information here]

## 🤝 Contributing

[Add contribution guidelines here]

## 📞 Support

[Add support/contact information here] 