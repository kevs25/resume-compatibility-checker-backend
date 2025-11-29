import traceback
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import os
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from analyzers.keyword_matcher import analyze_resume_match
from analyzers.semantic_analyzer import analyze_semantic_match
from analyzers.transformer_analyzer import analyze_with_transformers
from analyzers.hybrid_analyzer import analyze_hybrid
from analyzers.openrouter_analyzer import (
    analyze_resume_with_ai,
    enhance_resume_section,
    generate_cover_letter,
    get_interview_questions,
    MODELS
)

app = FastAPI(title="Resume Analyzer API")

# CORS middleware for React/Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your Next.js dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Resume Analyzer API is running"}

@app.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    analysis_type: str = Form("advanced")  # "basic", "semantic", "advanced"
):
    """
    Main endpoint to analyze resume against job description
    
    Args:
        resume: PDF or DOCX file
        job_description: Job description text
        analysis_type: "basic" (keyword), "semantic" (TF-IDF), or "advanced" (transformers)
    
    Returns:
        Analysis results based on selected method
    """
    try:
        # Check file extension
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        # Parse resume based on file type
        if file_extension == '.pdf':
            resume_text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            resume_text = await parse_docx(resume)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or DOCX file."
            )
        
        if analysis_type == "basic":
            # Keyword-based analysis only
            keyword_analysis = analyze_resume_match(resume_text, job_description)
            return {
                "success": True,
                "filename": resume.filename,
                "analysis_type": "Basic Keyword Matching",
                "result": keyword_analysis
            }
        
        elif analysis_type == "semantic":
            # TF-IDF semantic analysis
            semantic_analysis = analyze_semantic_match(resume_text, job_description)
            return {
                "success": True,
                "filename": resume.filename,
                "analysis_type": "TF-IDF Semantic Analysis",
                "result": semantic_analysis
            }
        
        else:  # "advanced" - default
            # Hybrid analysis (BEST) - Combines everything
            hybrid_result = analyze_hybrid(resume_text, job_description)
            
            return {
                "success": True,
                "filename": resume.filename,
                "analysis_type": "Hybrid AI (Best Accuracy)",
                "result": hybrid_result,
                "final_recommendation": hybrid_result["recommendation"]
            }
    
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-basic")
async def analyze_basic(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    """
    Basic keyword-only analysis (faster, simpler)
    """
    try:
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension == '.pdf':
            resume_text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            resume_text = await parse_docx(resume)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        analysis_result = analyze_resume_match(resume_text, job_description)
        
        return {
            "success": True,
            "filename": resume.filename,
            "result": analysis_result
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/parse-resume")
async def parse_resume_only(resume: UploadFile = File(...)):
    """
    Endpoint to parse and extract text from resume file
    """
    try:
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension == '.pdf':
            text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            text = await parse_docx(resume)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format"
            )
        
        return {
            "success": True,
            "filename": resume.filename,
            "text": text,
            "word_count": len(text.split())
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def get_available_models():
    """
    Get list of available AI models
    """
    return {
        "success": True,
        "models": {
            "free": [
                {"id": "llama-free", "name": "Llama 3.1 8B (Free)", "provider": "Meta"},
                {"id": "gemma-free", "name": "Gemma 2 9B (Free)", "provider": "Google"},
                {"id": "mistral-free", "name": "Mistral 7B (Free)", "provider": "Mistral"}
            ],
            "paid": [
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini", "provider": "OpenAI", "cost": "Low"},
                {"id": "claude-haiku", "name": "Claude 3 Haiku", "provider": "Anthropic", "cost": "Low"},
                {"id": "gpt-4o", "name": "GPT-4o", "provider": "OpenAI", "cost": "Medium"},
                {"id": "claude-sonnet", "name": "Claude 3.5 Sonnet", "provider": "Anthropic", "cost": "High"}
            ]
        }
    }

@app.post("/api/analyze-ai")
async def analyze_with_openrouter(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    model: str = Form("llama-free")
):
    """
    AI-powered resume analysis using OpenRouter
    
    Args:
        resume: PDF or DOCX file
        job_description: Job description text
        model: AI model to use (default: llama-free)
    
    Returns:
        AI analysis with match score, strengths, gaps, and recommendations
    """
    try:
        # Parse resume
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension == '.pdf':
            resume_text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            resume_text = await parse_docx(resume)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format. Please upload PDF or DOCX file."
            )
        
        # Analyze with AI
        result = analyze_resume_with_ai(resume_text, job_description, model)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return {
            "success": True,
            "filename": resume.filename,
            "analysis_type": f"AI Analysis ({model})",
            "result": result
        }
    
    except HTTPException:
        traceback.print_exc()
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/enhance-section")
async def enhance_section(
    section_text: str = Form(...),
    job_description: str = Form(...),
    section_name: str = Form("Work Experience"),
    model: str = Form("llama-free")
):
    """
    Enhance a resume section using AI
    
    Args:
        section_text: The section content to enhance
        job_description: Job description for context
        section_name: Name of the section (e.g., "Work Experience", "Skills")
        model: AI model to use
    
    Returns:
        Enhanced section with suggestions
    """
    try:
        result = enhance_resume_section(section_text, job_description, section_name, model)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return {
            "success": True,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate-cover-letter")
async def create_cover_letter(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    company_name: str = Form(...),
    model: str = Form("gpt-4o-mini")
):
    """
    Generate a tailored cover letter
    
    Args:
        resume: PDF or DOCX file
        job_description: Job description text
        company_name: Company name
        model: AI model to use (default: gpt-4o-mini for quality)
    
    Returns:
        Generated cover letter
    """
    try:
        # Parse resume
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension == '.pdf':
            resume_text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            resume_text = await parse_docx(resume)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format"
            )
        
        # Generate cover letter
        result = generate_cover_letter(resume_text, job_description, company_name, model)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return {
            "success": True,
            "filename": resume.filename,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/interview-prep")
async def prepare_interview(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
    model: str = Form("llama-free")
):
    """
    Generate likely interview questions and preparation tips
    
    Args:
        resume: PDF or DOCX file
        job_description: Job description text
        model: AI model to use
    
    Returns:
        Interview questions with preparation tips
    """
    try:
        # Parse resume
        file_extension = os.path.splitext(resume.filename)[1].lower()
        
        if file_extension == '.pdf':
            resume_text = await parse_pdf(resume)
        elif file_extension in ['.docx', '.doc']:
            resume_text = await parse_docx(resume)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format"
            )
        
        # Get interview questions
        result = get_interview_questions(resume_text, job_description, model)
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result.get("message"))
        
        return {
            "success": True,
            "filename": resume.filename,
            "result": result
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)