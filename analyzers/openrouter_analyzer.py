"""
OpenRouter AI-powered resume analysis
Supports multiple models including free tiers
"""

import requests
import json
from typing import Dict, Optional, List
import os
from dotenv import load_dotenv
load_dotenv()
# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Available models
MODELS = {
    # Free models
    "llama-free": "meta-llama/llama-3.2-3b-instruct:free",
    "gemma-free": "google/gemma-2-9b-it:free",
    "mistral-free": "mistralai/mistral-7b-instruct:free",
    
    # Paid models (cheap to expensive)
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "claude-haiku": "anthropic/claude-3-haiku",
    "gpt-4o": "openai/gpt-4o",
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
}

def get_api_key() -> str:
    """Get OpenRouter API key from environment variable"""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")
    return api_key


def _call_openrouter_single(
    prompt: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict:
    """
    Low-level helper: call OpenRouter for a single model.
    Returns either {"success": True, "model_used": ..., "response": ...}
    or {"error": True, "message": ..., "status_code": ..., "detail": ...}.
    """
    api_key = get_api_key()
    model_id = MODELS.get(model, model)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/yourusername/resume-analyzer",
        "X-Title": "Resume Analyzer",
    }

    data = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": "You MUST respond with ONLY valid JSON. No explanations. No markdown.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=data,
            timeout=60,
        )

        if response.status_code in (401, 404, 429, 500, 503):
            return {
                "error": True,
                "message": f"HTTP error from OpenRouter (status {response.status_code})",
                "status_code": response.status_code,
                "detail": response.text,
                "model_used": model,
            }

        response.raise_for_status()
        result = response.json()

        return {
            "success": True,
            "model_used": model,
            "response": result,
        }

    except requests.exceptions.Timeout:
        return {
            "error": True,
            "message": "Request timed out. The model may be slow or unavailable.",
            "model_used": model,
        }
    except requests.exceptions.RequestException as e:
        detail = ""
        if getattr(e, "response", None) is not None:
            try:
                detail = e.response.text
            except Exception:
                detail = str(e)
        else:
            detail = str(e)

        return {
            "error": True,
            "message": f"API request failed: {str(e)}",
            "detail": detail,
            "model_used": model,
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Unexpected error: {str(e)}",
            "model_used": model,
        }


# def call_openrouter(
#     prompt: str,
#     model: str = "llama-free",
#     temperature: float = 0.7,
#     max_tokens: int = 2000
# ) -> Dict:
    """
    Call OpenRouter API
    
    Args:
        prompt: The prompt to send
        model: Model key from MODELS dict
        temperature: Creativity (0-1)
        max_tokens: Max response length
    
    Returns:
        API response dict
    """
    try:
        api_key = get_api_key()
        model_id = MODELS.get(model, model)  # Allow direct model IDs too
        
        # Required headers for OpenRouter
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/yourusername/resume-analyzer",  # Optional but recommended
            "X-Title": "Resume Analyzer"  # Optional but recommended
        }
        
        # Request body - exactly matching OpenRouter spec
        data = {
            "model": model_id,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Make the request
        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=data,
            timeout=60  # Increased timeout for slower models
        )
        
        # Check for HTTP errors
        if response.status_code == 404:
            return {
                "error": True,
                "message": "API endpoint not found. Check your OpenRouter API key and endpoint URL.",
                "status_code": 404
            }
        elif response.status_code == 401:
            return {
                "error": True,
                "message": "Invalid API key. Please check your OPENROUTER_API_KEY environment variable.",
                "status_code": 401
            }
        elif response.status_code == 429:
            return {
                "error": True,
                "message": "Rate limit exceeded. Please wait a moment and try again.",
                "status_code": 429
            }
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.Timeout:
        return {
            "error": True,
            "message": "Request timed out. The model may be slow or unavailable."
        }
    except requests.exceptions.RequestException as e:
        # Try to get error details from response
        error_detail = str(e)
        if hasattr(e.response, 'text'):
            try:
                error_json = e.response.json()
                error_detail = error_json.get('error', {}).get('message', str(e))
            except:
                error_detail = e.response.text if e.response else str(e)
        
        return {
            "error": True,
            "message": f"API request failed: {error_detail}",
            "full_error": str(e)
        }
    except Exception as e:
        return {
            "error": True,
            "message": f"Unexpected error: {str(e)}"
        }


def call_openrouter(
    prompt: str,
    model: str = "llama-free",
    temperature: float = 0.7,
    max_tokens: int = 2000
) -> Dict:
    """
    Call OpenRouter with automatic fallback models (HTTP-level only).
    This is kept for compatibility with helper functions that don't do
    their own JSON validation.
    """

    # Main model + fallbacks (without duplicates)
    models_to_try = [model] + [m for m in MODELS if m != model]

    errors: List[Dict] = []

    for current_model in models_to_try:
        result = _call_openrouter_single(
            prompt=prompt,
            model=current_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if result.get("success"):
            return result

        errors.append(result)

    # If all models fail:
    return {
        "error": True,
        "message": "All models failed.",
        "errors": errors,
    }


# def analyze_resume_with_ai(
#     resume_text: str,
#     job_description: str,
#     model: str = "llama-free"
# ) -> Dict:
#     """
#     Analyze resume against JD using AI
    
#     Args:
#         resume_text: Full resume text
#         job_description: Job description text
#         model: Model to use (default: free Llama)
    
#     Returns:
#         Analysis results with match score and insights
#     """
    
#     prompt = f"""You are an expert resume analyzer and career coach. Analyze this resume against the job description and provide detailed feedback.

# JOB DESCRIPTION:
# {job_description}

# RESUME:
# {resume_text}

# Provide your analysis in the following JSON format (respond ONLY with valid JSON, no other text):

# {{
#     "match_score": <number 0-100>,
#     "overall_assessment": "<one sentence overall assessment>",
#     "strengths": [
#         "<strength 1>",
#         "<strength 2>",
#         "<strength 3>"
#     ],
#     "gaps": [
#         "<gap 1>",
#         "<gap 2>",
#         "<gap 3>"
#     ],
#     "matched_requirements": [
#         "<requirement 1 that is met>",
#         "<requirement 2 that is met>"
#     ],
#     "missing_requirements": [
#         "<requirement 1 that is missing>",
#         "<requirement 2 that is missing>"
#     ],
#     "recommendations": [
#         "<actionable recommendation 1>",
#         "<actionable recommendation 2>",
#         "<actionable recommendation 3>"
#     ],
#     "experience_match": "<brief assessment of experience level match>",
#     "skills_match": "<brief assessment of technical skills match>",
#     "application_recommendation": "<APPLY/CONSIDER/NOT_RECOMMENDED>"
# }}

# Be honest and specific in your assessment."""

#     response = call_openrouter(prompt, model=model, temperature=0.3, max_tokens=2000)
    
#     if response.get("error"):
#         return response
    
#     try:
#         # Extract content from response
#         content = response["choices"][0]["message"]["content"]
        
#         # Try to parse JSON (handle markdown code blocks)
#         content = content.strip()
#         if content.startswith("```json"):
#             content = content[7:]  # Remove ```json
#         if content.startswith("```"):
#             content = content[3:]  # Remove ```
#         if content.endswith("```"):
#             content = content[:-3]  # Remove ```
        
#         analysis = json.loads(content.strip())
        
#         return {
#             "success": True,
#             "model_used": model,
#             "analysis": analysis,
#             "tokens_used": response.get("usage", {})
#         }
    
#     except json.JSONDecodeError as e:
#         return {
#             "error": True,
#             "message": f"Failed to parse AI response as JSON: {str(e)}",
#             "raw_response": content
#         }
#     except Exception as e:
#         return {
#             "error": True,
#             "message": f"Error processing response: {str(e)}"
#         }

def analyze_resume_with_ai(resume_text: str,job_description: str,model: str = "llama-free") -> Dict:
    """
    Analyze resume against JD using AI with strong JSON extraction.
    """

    # -------------------------
    # 1. Strong system message
    # -------------------------
    system_prompt = (
        "You are an expert resume analyzer. "
        "You MUST respond with ONLY valid JSON. "
        "Do not add explanations, comments, or markdown. "
        "Your entire response MUST be a single JSON object."
    )

    # -------------------------
    # 2. User prompt
    # -------------------------
    user_prompt = f"""
Analyze this resume against the job description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Respond ONLY with valid JSON in the following format:

{{
    "match_score": <number 0-100>,
    "overall_assessment": "<summary>",
    "strengths": ["<s1>", "<s2>", "<s3>"],
    "gaps": ["<g1>", "<g2>", "<g3>"],
    "matched_requirements": ["<req1>", "<req2>"],
    "missing_requirements": ["<req1>", "<req2>"],
    "recommendations": ["<r1>", "<r2>", "<r3>"],
    "experience_match": "<assessment>",
    "skills_match": "<assessment>",
    "application_recommendation": "<APPLY/CONSIDER/NOT_RECOMMENDED>"
}}
"""
    # -------------------------
    # 3. Call OpenRouter with JSON-aware fallbacks
    # -------------------------
    models_to_try = [model] + [m for m in MODELS if m != model]
    errors: List[Dict] = []

    for current_model in models_to_try:
        print(f"[OpenRouter] analyze_resume_with_ai trying model: {current_model}")

        api_result = _call_openrouter_single(
            prompt=user_prompt,
            model=current_model,
            temperature=0.2,
            max_tokens=2000,
        )

        # HTTP / network / low-level error -> record and try next model
        if api_result.get("error"):
            errors.append(api_result)
            continue

        inner = api_result.get("response") or {}
        content = ""

        try:
            choices = inner.get("choices")
            if not choices:
                errors.append({
                    "error": True,
                    "message": "OpenRouter response did not contain 'choices'.",
                    "raw_response": inner,
                    "model_used": current_model,
                })
                continue

            message = choices[0].get("message", {})
            content = (message.get("content") or "").strip()

            if not content:
                errors.append({
                    "error": True,
                    "message": "OpenRouter response did not contain any content.",
                    "raw_response": inner,
                    "model_used": current_model,
                })
                continue

            # Remove code-block markers
            content = content.replace("```json", "").replace("```", "").strip()

            # Extract JSON using braces detection
            first_brace = content.find("{")
            last_brace = content.rfind("}")

            if first_brace == -1 or last_brace == -1:
                errors.append({
                    "error": True,
                    "message": "No valid JSON object found in the AI response.",
                    "raw_response": content,
                    "model_used": current_model,
                })
                continue

            json_str = content[first_brace:last_brace + 1]

            # Parse JSON
            analysis = json.loads(json_str)

            return {
                "success": True,
                "model_used": current_model,
                "analysis": analysis,
                "tokens_used": inner.get("usage", {})
            }

        except json.JSONDecodeError as e:
            errors.append({
                "error": True,
                "message": f"Failed to parse AI response as JSON: {str(e)}",
                "raw_response": content,
                "model_used": current_model,
            })
            continue
        except Exception as e:
            errors.append({
                "error": True,
                "message": f"Unexpected error: {str(e)}",
                "raw_response": content,
                "model_used": current_model,
            })
            continue

    # If we reach here, all models failed in some way (HTTP or JSON)
    return {
        "error": True,
        "message": "All models failed to return valid JSON.",
        "errors": errors,
    }


def enhance_resume_section(
    section_text: str,
    job_description: str,
    section_name: str,
    model: str = "llama-free"
) -> Dict:
    """
    Get AI suggestions to improve a specific resume section
    
    Args:
        section_text: The section to enhance (e.g., experience bullet points)
        job_description: Job description for context
        section_name: Name of section (e.g., "Work Experience", "Skills")
        model: Model to use
    
    Returns:
        Enhanced version with suggestions
    """
    
    prompt = f"""You are an expert resume writer. Enhance this resume section to better match the job description while keeping it truthful.

JOB DESCRIPTION:
{job_description}

CURRENT {section_name.upper()} SECTION:
{section_text}

Provide your suggestions in JSON format (respond ONLY with valid JSON):

{{
    "enhanced_version": "<improved version of the section>",
    "key_changes": [
        "<change 1>",
        "<change 2>",
        "<change 3>"
    ],
    "keywords_added": ["<keyword1>", "<keyword2>"],
    "explanation": "<brief explanation of improvements>"
}}

Focus on:
1. Using keywords from the job description
2. Quantifying achievements where possible
3. Using strong action verbs
4. Highlighting relevant experience
5. Keeping it concise and impactful"""

    response = call_openrouter(prompt, model=model, temperature=0.5, max_tokens=1500)
    
    if response.get("error"):
        return response
    
    try:
        inner = response.get("response") or {}
        choices = inner.get("choices")
        if not choices:
            return {
                "error": True,
                "message": "OpenRouter response did not contain 'choices' for enhance_resume_section.",
                "raw_response": inner
            }

        content = choices[0].get("message", {}).get("content", "")
        
        # Clean markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        suggestions = json.loads(content.strip())
        
        return {
            "success": True,
            "model_used": response.get("model_used", model),
            "section_name": section_name,
            "suggestions": suggestions,
            "tokens_used": inner.get("usage", {})
        }
    
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Failed to parse AI response: {str(e)}",
            "raw_response": content
        }

def generate_cover_letter(
    resume_text: str,
    job_description: str,
    company_name: str,
    model: str = "gpt-4o-mini"  # Use better model for cover letters
) -> Dict:
    """
    Generate a tailored cover letter
    
    Args:
        resume_text: Full resume text
        job_description: Job description
        company_name: Company name
        model: Model to use (default: GPT-4o mini for quality)
    
    Returns:
        Generated cover letter
    """
    
    prompt = f"""You are an expert cover letter writer. Create a compelling, professional cover letter based on this resume and job description.

COMPANY: {company_name}

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Generate a cover letter that:
1. Is 3-4 paragraphs long
2. Highlights the most relevant experience from the resume
3. Shows enthusiasm for the role and company
4. Addresses key requirements from the job description
5. Maintains a professional yet personable tone
6. Is specific and avoids generic statements

Provide response in JSON format:

{{
    "cover_letter": "<full cover letter text>",
    "key_highlights": [
        "<highlight 1 from resume that matches JD>",
        "<highlight 2 from resume that matches JD>",
        "<highlight 3 from resume that matches JD>"
    ],
    "tone": "<brief description of tone used>"
}}"""

    response = call_openrouter(prompt, model=model, temperature=0.7, max_tokens=2000)
    
    if response.get("error"):
        return response
    
    try:
        inner = response.get("response") or {}
        choices = inner.get("choices")
        if not choices:
            return {
                "error": True,
                "message": "OpenRouter response did not contain 'choices' for generate_cover_letter.",
                "raw_response": inner
            }

        content = choices[0].get("message", {}).get("content", "")
        
        # Clean markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        cover_letter_data = json.loads(content.strip())
        
        return {
            "success": True,
            "model_used": response.get("model_used", model),
            "cover_letter_data": cover_letter_data,
            "tokens_used": inner.get("usage", {})
        }
    
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Failed to parse AI response: {str(e)}",
            "raw_response": content
        }

def get_interview_questions(
    resume_text: str,
    job_description: str,
    model: str = "llama-free"
) -> Dict:
    """
    Generate likely interview questions based on resume and JD
    
    Args:
        resume_text: Full resume text
        job_description: Job description
        model: Model to use
    
    Returns:
        List of interview questions with preparation tips
    """
    
    prompt = f"""You are an experienced technical interviewer. Based on this resume and job description, generate the most likely interview questions.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Provide response in JSON format:

{{
    "technical_questions": [
        {{"question": "<technical question 1>", "hint": "<how to prepare>"}},
        {{"question": "<technical question 2>", "hint": "<how to prepare>"}},
        {{"question": "<technical question 3>", "hint": "<how to prepare>"}}
    ],
    "behavioral_questions": [
        {{"question": "<behavioral question 1>", "hint": "<how to answer>"}},
        {{"question": "<behavioral question 2>", "hint": "<how to answer>"}}
    ],
    "experience_questions": [
        {{"question": "<experience question 1>", "hint": "<what to highlight>"}},
        {{"question": "<experience question 2>", "hint": "<what to highlight>"}}
    ],
    "preparation_tips": [
        "<tip 1>",
        "<tip 2>",
        "<tip 3>"
    ]
}}"""

    response = call_openrouter(prompt, model=model, temperature=0.6, max_tokens=2000)
    
    if response.get("error"):
        return response
    
    try:
        inner = response.get("response") or {}
        choices = inner.get("choices")
        if not choices:
            return {
                "error": True,
                "message": "OpenRouter response did not contain 'choices' for get_interview_questions.",
                "raw_response": inner
            }

        content = choices[0].get("message", {}).get("content", "")
        
        # Clean markdown
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        
        questions_data = json.loads(content.strip())
        
        return {
            "success": True,
            "model_used": response.get("model_used", model),
            "questions": questions_data,
            "tokens_used": inner.get("usage", {})
        }
    
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Failed to parse AI response: {str(e)}",
            "raw_response": content
        }