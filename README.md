# Resume Compatibility Checker Backend

A powerful FastAPI-based backend service for analyzing resume compatibility with job descriptions. This tool provides multiple analysis methods, from basic keyword matching to advanced AI-powered analysis, helping job seekers understand how well their resume matches job requirements.

## Features

### 🔍 Multiple Analysis Methods

- **Basic Keyword Matching**: Fast, simple keyword-based analysis
- **Semantic Analysis**: TF-IDF based semantic similarity matching
- **Hybrid Analysis**: Combines explicit skill matching, experience validation, and semantic understanding for the most accurate results
- **AI-Powered Analysis**: Leverages OpenRouter API with multiple AI models (including free tiers)

### 📄 Document Parsing

- Supports PDF and DOCX resume formats
- Automatic text extraction and processing
- Word count and content analysis

### 🤖 AI Features

- **Resume Analysis**: Get detailed match scores, strengths, gaps, and recommendations
- **Section Enhancement**: AI-powered suggestions to improve specific resume sections
- **Cover Letter Generation**: Generate tailored cover letters based on resume and job description
- **Interview Preparation**: Get likely interview questions and preparation tips

### 🎯 Analysis Capabilities

- Skill matching (technical and soft skills)
- Experience requirement validation
- Requirement-by-requirement analysis
- Semantic similarity scoring
- Actionable insights and recommendations

## Tech Stack

- **FastAPI**: Modern, fast web framework for building APIs
- **Sentence Transformers**: For semantic similarity analysis
- **scikit-learn**: TF-IDF and machine learning utilities
- **PyPDF2**: PDF parsing
- **python-docx**: DOCX parsing
- **OpenRouter API**: Access to multiple AI models (GPT-4, Claude, Llama, etc.)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd resume-compatibility-checker-backend
```

2. Create a virtual environment:
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows (PowerShell)**:
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     venv\Scripts\activate.bat
     ```
   - **Linux/Mac**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

   > **Note**: Get your API key from [OpenRouter](https://openrouter.ai/). Free tier models are available without credit card.

## Usage

### Starting the Server

Run the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative docs**: `http://localhost:8000/redoc`

## API Endpoints

### Core Endpoints

#### `POST /api/analyze`
Main endpoint for resume analysis with multiple analysis types.

**Parameters:**
- `resume` (file): PDF or DOCX resume file
- `job_description` (form): Job description text
- `analysis_type` (form, optional): `"basic"`, `"semantic"`, or `"advanced"` (default: `"advanced"`)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "resume=@resume.pdf" \
  -F "job_description=We are looking for a Python developer with 3+ years of experience..." \
  -F "analysis_type=advanced"
```

#### `POST /api/analyze-basic`
Fast keyword-only analysis.

#### `POST /api/parse-resume`
Extract and return text from resume file.

#### `GET /api/models`
Get list of available AI models (free and paid).

### AI-Powered Endpoints

#### `POST /api/analyze-ai`
AI-powered resume analysis using OpenRouter.

**Parameters:**
- `resume` (file): PDF or DOCX resume file
- `job_description` (form): Job description text
- `model` (form, optional): AI model ID (default: `"llama-free"`)

**Available Models:**
- Free: `llama-free`, `gemma-free`, `mistral-free`
- Paid: `gpt-4o-mini`, `claude-haiku`, `gpt-4o`, `claude-sonnet`

#### `POST /api/enhance-section`
Enhance a specific resume section with AI suggestions.

**Parameters:**
- `section_text` (form): The section content to enhance
- `job_description` (form): Job description for context
- `section_name` (form, optional): Name of the section (default: `"Work Experience"`)
- `model` (form, optional): AI model ID

#### `POST /api/generate-cover-letter`
Generate a tailored cover letter.

**Parameters:**
- `resume` (file): PDF or DOCX resume file
- `job_description` (form): Job description text
- `company_name` (form): Company name
- `model` (form, optional): AI model ID (default: `"gpt-4o-mini"`)

#### `POST /api/interview-prep`
Generate interview questions and preparation tips.

**Parameters:**
- `resume` (file): PDF or DOCX resume file
- `job_description` (form): Job description text
- `model` (form, optional): AI model ID

## Response Format

### Analysis Response Example

```json
{
  "success": true,
  "filename": "resume.pdf",
  "analysis_type": "Hybrid AI (Best Accuracy)",
  "result": {
    "final_score": 78.5,
    "skill_analysis": {
      "match_percentage": 85.0,
      "matched_skills": ["python", "fastapi", "docker"],
      "missing_skills": ["kubernetes"],
      "total_required": 10,
      "total_matched": 9
    },
    "experience_analysis": {
      "meets_requirement": true,
      "required_years": 3,
      "resume_years": 4.5,
      "message": "✓ 4.5 years meets 3+ requirement"
    },
    "requirement_analysis": {
      "matches": [...],
      "total_requirements": 8,
      "strong_matches": 6,
      "average_score": 72.3
    },
    "semantic_similarity": 81.2,
    "insights": [
      "🎯 Excellent match! You're highly qualified for this role.",
      "💪 Strong technical skills match (85%)"
    ],
    "recommendation": "HIGHLY RECOMMENDED - Apply with confidence!"
  },
  "final_recommendation": "HIGHLY RECOMMENDED - Apply with confidence!"
}
```

## CORS Configuration

The API is configured to allow requests from `http://localhost:3000` (typical Next.js/React dev server). To modify CORS settings, edit `app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Modify as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Project Structure

```
resume-compatibility-checker-backend/
├── app/
│   └── main.py              # FastAPI application and endpoints
├── analyzers/
│   ├── hybrid_analyzer.py   # Hybrid analysis (best accuracy)
│   ├── keyword_matcher.py   # Basic keyword matching
│   ├── semantic_analyzer.py # TF-IDF semantic analysis
│   ├── transformer_analyzer.py # Transformer-based analysis
│   └── openrouter_analyzer.py # AI-powered analysis via OpenRouter
├── parsers/
│   ├── pdf_parser.py        # PDF text extraction
│   └── docx_parser.py       # DOCX text extraction
├── utils/
│   └── text_processing.py   # Text processing utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Development

### Running Tests

Test the OpenRouter API connection:
```bash
python test_openrouter_api.py
```

### Adding New Analysis Methods

1. Create a new analyzer in `analyzers/`
2. Import and add endpoint in `app/main.py`
3. Follow the existing response format for consistency

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENROUTER_API_KEY` | OpenRouter API key for AI features | Yes (for AI endpoints) |

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]

## Support

For issues, questions, or contributions, please open an issue on the repository.

