"""
Advanced semantic analysis using Sentence Transformers
This provides much better semantic understanding than TF-IDF
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict, List, Tuple
import re

# Global model instance (loaded once)
_model = None

def get_model():
    """Lazy load the sentence transformer model"""
    global _model
    if _model is None:
        # Using all-MiniLM-L6-v2: Fast, accurate, only 80MB
        # Alternative: 'all-mpnet-base-v2' (420MB, more accurate but slower)
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

def clean_text_for_embedding(text: str) -> str:
    """Clean text while preserving important information"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basics
    text = re.sub(r'[^\w\s\+\#\.\-,]', ' ', text)
    return text.strip()

def extract_resume_sections(resume_text: str) -> Dict[str, str]:
    """Extract key sections from resume"""
    sections = {
        'full_text': resume_text,
        'skills': '',
        'experience': '',
        'projects': '',
        'education': ''
    }
    
    text_lower = resume_text.lower()
    
    # Skills section
    skills_match = re.search(r'(skills?.*?tools?)(.*?)(?=\n[A-Z]{2,}|work\s+experience|experience|$)', 
                            text_lower, re.DOTALL)
    if skills_match:
        sections['skills'] = skills_match.group(2).strip()
    
    # Experience section
    exp_match = re.search(r'(work\s+experience|experience)(.*?)(?=\nprojects?|\neducation|$)', 
                         text_lower, re.DOTALL)
    if exp_match:
        sections['experience'] = exp_match.group(2).strip()
    
    # Projects section
    proj_match = re.search(r'(projects?)(.*?)(?=\neducation|$)', 
                          text_lower, re.DOTALL)
    if proj_match:
        sections['projects'] = proj_match.group(2).strip()
    
    # Education section
    edu_match = re.search(r'(education)(.*?)$', text_lower, re.DOTALL)
    if edu_match:
        sections['education'] = edu_match.group(2).strip()
    
    return sections

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts using sentence transformers"""
    model = get_model()
    
    # Clean texts
    text1_clean = clean_text_for_embedding(text1)
    text2_clean = clean_text_for_embedding(text2)
    
    # Generate embeddings
    embeddings = model.encode([text1_clean, text2_clean])
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return float(similarity)

def analyze_section_similarity(resume_sections: Dict[str, str], job_description: str) -> Dict[str, float]:
    """Analyze similarity for each resume section"""
    section_scores = {}
    
    important_sections = ['skills', 'experience', 'projects']
    
    for section_name in important_sections:
        section_text = resume_sections.get(section_name, '')
        if section_text and len(section_text) > 20:  # Minimum length check
            similarity = calculate_semantic_similarity(section_text, job_description)
            section_scores[section_name] = round(similarity * 100, 2)
    
    return section_scores

def extract_key_requirements(job_description: str) -> List[str]:
    """Extract individual requirements from JD"""
    # Split by bullet points, newlines, or periods
    requirements = []
    
    # Split by common delimiters
    lines = re.split(r'[•\n\*\-]|\d+\.', job_description)
    
    for line in lines:
        line = line.strip()
        if len(line) > 10:  # Meaningful requirement
            requirements.append(line)
    
    return requirements

def calculate_requirement_matches(resume_text: str, requirements: List[str]) -> List[Dict]:
    """Calculate how well resume matches each individual requirement"""
    model = get_model()
    
    if not requirements:
        return []
    
    # Get embeddings for resume and all requirements
    resume_embedding = model.encode(clean_text_for_embedding(resume_text))
    requirement_embeddings = model.encode([clean_text_for_embedding(req) for req in requirements])
    
    matches = []
    for i, req in enumerate(requirements):
        similarity = cosine_similarity([resume_embedding], [requirement_embeddings[i]])[0][0]
        matches.append({
            'requirement': req.strip(),
            'match_score': round(similarity * 100, 2),
            'status': 'Strong Match' if similarity > 0.7 else 'Partial Match' if similarity > 0.5 else 'Weak Match'
        })
    
    # Sort by match score
    matches.sort(key=lambda x: x['match_score'], reverse=True)
    
    return matches

def calculate_skill_relevance(resume_text: str, job_description: str) -> Dict:
    """Calculate overall skill and experience relevance"""
    
    # Overall similarity
    overall_similarity = calculate_semantic_similarity(resume_text, job_description)
    
    # Extract sections
    sections = extract_resume_sections(resume_text)
    
    # Section-wise similarity
    section_scores = analyze_section_similarity(sections, job_description)
    
    # Requirement-level matching
    requirements = extract_key_requirements(job_description)
    requirement_matches = calculate_requirement_matches(resume_text, requirements)
    
    # Calculate weighted score
    # Skills section: 30%, Experience: 40%, Projects: 20%, Overall: 10%
    weighted_score = 0
    weights_used = 0
    
    if 'skills' in section_scores:
        weighted_score += section_scores['skills'] * 0.30
        weights_used += 0.30
    
    if 'experience' in section_scores:
        weighted_score += section_scores['experience'] * 0.40
        weights_used += 0.40
    
    if 'projects' in section_scores:
        weighted_score += section_scores['projects'] * 0.20
        weights_used += 0.20
    
    # Add overall similarity for remaining weight
    if weights_used < 1.0:
        weighted_score += (overall_similarity * 100) * (1.0 - weights_used)
    
    # Calculate average requirement match
    avg_requirement_match = 0
    if requirement_matches:
        avg_requirement_match = sum(r['match_score'] for r in requirement_matches) / len(requirement_matches)
    
    # Final combined score (70% weighted sections + 30% requirement matching)
    final_score = (weighted_score * 0.7) + (avg_requirement_match * 0.3)
    
    return {
        'overall_similarity': round(overall_similarity * 100, 2),
        'section_scores': section_scores,
        'weighted_score': round(weighted_score, 2),
        'requirement_matches': requirement_matches[:10],  # Top 10 requirements
        'average_requirement_match': round(avg_requirement_match, 2),
        'final_match_score': round(final_score, 2)
    }

def generate_insights(analysis_result: Dict) -> List[str]:
    """Generate human-readable insights"""
    insights = []
    final_score = analysis_result['final_match_score']
    
    # Overall assessment
    if final_score >= 75:
        insights.append("🎯 Excellent match! Your profile strongly aligns with this role.")
    elif final_score >= 60:
        insights.append("✅ Strong match! You're a good fit for this position.")
    elif final_score >= 45:
        insights.append("⚠️ Moderate match. You meet some requirements but may need to highlight relevant experience.")
    else:
        insights.append("❌ Limited match detected. Consider whether this role aligns with your background.")
    
    # Section-specific insights
    section_scores = analysis_result['section_scores']
    
    if section_scores:
        best_section = max(section_scores.items(), key=lambda x: x[1])
        if best_section[1] >= 60:
            insights.append(f"💪 Strong {best_section[0]} alignment ({best_section[1]}%)")
        
        weak_sections = [s for s, score in section_scores.items() if score < 50]
        if weak_sections:
            insights.append(f"⚡ Consider strengthening: {', '.join(weak_sections)}")
    
    # Requirement matching insights
    req_matches = analysis_result['requirement_matches']
    if req_matches:
        strong_matches = [r for r in req_matches if r['match_score'] >= 70]
        weak_matches = [r for r in req_matches if r['match_score'] < 50]
        
        if strong_matches:
            insights.append(f"✨ {len(strong_matches)} requirements strongly matched")
        if weak_matches and len(weak_matches) > len(strong_matches):
            insights.append(f"🔍 {len(weak_matches)} requirements need attention")
    
    # Overall similarity insight
    overall = analysis_result['overall_similarity']
    if overall >= 70:
        insights.append(f"📊 Excellent semantic alignment ({overall}%)")
    elif overall >= 50:
        insights.append(f"📊 Good contextual match ({overall}%)")
    else:
        insights.append(f"📊 Consider using terminology from the job description ({overall}%)")
    
    return insights

def get_recommendation(score: float) -> str:
    """Get application recommendation"""
    if score >= 75:
        return "HIGHLY RECOMMENDED - Apply with confidence! Strong alignment across all areas."
    elif score >= 60:
        return "RECOMMENDED - Good fit. Highlight your matching skills and experience."
    elif score >= 45:
        return "CONSIDER - Moderate fit. Tailor your resume and emphasize relevant experience."
    else:
        return "REVIEW CAREFULLY - Significant gaps. Consider if this role matches your career goals."

def analyze_with_transformers(resume_text: str, job_description: str) -> Dict:
    """
    Main function for transformer-based semantic analysis
    This is much more accurate than TF-IDF for understanding meaning
    """
    
    # Calculate skill relevance
    relevance = calculate_skill_relevance(resume_text, job_description)
    
    # Generate insights
    insights = generate_insights(relevance)
    
    # Get recommendation
    recommendation = get_recommendation(relevance['final_match_score'])
    
    return {
        'final_match_score': relevance['final_match_score'],
        'overall_semantic_similarity': relevance['overall_similarity'],
        'section_scores': relevance['section_scores'],
        'weighted_section_score': relevance['weighted_score'],
        'requirement_analysis': {
            'average_match': relevance['average_requirement_match'],
            'top_matches': relevance['requirement_matches'][:5],
            'total_requirements': len(relevance['requirement_matches'])
        },
        'insights': insights,
        'recommendation': recommendation,
        'analysis_method': 'Sentence Transformers (all-MiniLM-L6-v2)'
    }