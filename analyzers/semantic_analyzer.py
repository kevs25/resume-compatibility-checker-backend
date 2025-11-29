from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import Dict, List, Tuple
import numpy as np

def preprocess_text(text: str) -> str:
    """Preprocess text for semantic analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace special characters with spaces but keep important ones
    text = re.sub(r'[^\w\s\+\#\.]', ' ', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_sections(resume_text: str) -> Dict[str, str]:
    """Extract different sections from resume"""
    text_lower = resume_text.lower()
    sections = {}
    
    # Common section headers
    section_patterns = {
        'skills': r'(skills?.*?tools?)(.*?)(?=\n[A-Z]|\n\n|$)',
        'experience': r'(work\s+experience|experience|professional\s+experience)(.*?)(?=\n[A-Z]{2,}|\neducation|$)',
        'education': r'(education)(.*?)(?=\n[A-Z]{2,}|$)',
        'projects': r'(projects?)(.*?)(?=\n[A-Z]{2,}|\neducation|$)',
    }
    
    for section_name, pattern in section_patterns.items():
        match = re.search(pattern, text_lower, re.DOTALL | re.IGNORECASE)
        if match:
            sections[section_name] = match.group(2).strip()
    
    # If sections not found, use full text
    if not sections:
        sections['full_text'] = resume_text
    
    return sections

def calculate_tfidf_similarity(resume_text: str, job_description: str) -> Dict:
    """
    Calculate semantic similarity using TF-IDF vectorization
    
    Returns overall and section-wise similarity scores
    """
    # Preprocess texts
    resume_processed = preprocess_text(resume_text)
    jd_processed = preprocess_text(job_description)
    
    # Create TF-IDF vectorizer with optimized parameters
    vectorizer = TfidfVectorizer(
        max_features=500,  # Top 500 features
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        stop_words='english',
        min_df=1,  # Minimum document frequency
    )
    
    # Fit and transform
    try:
        tfidf_matrix = vectorizer.fit_transform([resume_processed, jd_processed])
        
        # Calculate cosine similarity
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        overall_percentage = round(similarity * 100, 2)
        
    except Exception as e:
        print(f"Error in TF-IDF calculation: {e}")
        overall_percentage = 0.0
    
    # Get top matching terms
    try:
        feature_names = vectorizer.get_feature_names_out()
        resume_vector = tfidf_matrix[0].toarray()[0]
        jd_vector = tfidf_matrix[1].toarray()[0]
        
        # Find common important terms
        common_terms = []
        for idx, (res_score, jd_score) in enumerate(zip(resume_vector, jd_vector)):
            if res_score > 0 and jd_score > 0:
                # Both documents have this term
                importance = min(res_score, jd_score)
                common_terms.append((feature_names[idx], importance))
        
        # Sort by importance
        common_terms.sort(key=lambda x: x[1], reverse=True)
        top_matching_terms = [term for term, _ in common_terms[:15]]
        
    except Exception as e:
        print(f"Error extracting terms: {e}")
        top_matching_terms = []
    
    return {
        "overall_similarity": overall_percentage,
        "top_matching_terms": top_matching_terms,
    }

def calculate_section_wise_similarity(resume_text: str, job_description: str) -> Dict[str, float]:
    """Calculate similarity for each resume section against JD"""
    sections = extract_sections(resume_text)
    jd_processed = preprocess_text(job_description)
    
    section_scores = {}
    
    vectorizer = TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        stop_words='english',
    )
    
    for section_name, section_text in sections.items():
        if not section_text.strip():
            continue
            
        section_processed = preprocess_text(section_text)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([section_processed, jd_processed])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            section_scores[section_name] = round(similarity * 100, 2)
        except:
            section_scores[section_name] = 0.0
    
    return section_scores

def calculate_keyword_density(resume_text: str, job_description: str) -> Dict:
    """Calculate how well important JD keywords appear in resume"""
    jd_words = preprocess_text(job_description).split()
    resume_words = preprocess_text(resume_text).split()
    
    # Count JD word frequencies
    jd_word_freq = {}
    for word in jd_words:
        if len(word) > 2:  # Skip very short words
            jd_word_freq[word] = jd_word_freq.get(word, 0) + 1
    
    # Sort by frequency (top keywords in JD)
    top_jd_keywords = sorted(jd_word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
    
    # Check presence in resume
    keyword_coverage = []
    for keyword, jd_count in top_jd_keywords:
        resume_count = resume_words.count(keyword)
        coverage = {
            "keyword": keyword,
            "jd_frequency": jd_count,
            "resume_frequency": resume_count,
            "present": resume_count > 0
        }
        keyword_coverage.append(coverage)
    
    # Calculate coverage percentage
    present_count = sum(1 for kw in keyword_coverage if kw["present"])
    coverage_percentage = round((present_count / len(keyword_coverage)) * 100, 2) if keyword_coverage else 0
    
    return {
        "keyword_coverage_percentage": coverage_percentage,
        "detailed_coverage": keyword_coverage
    }

def analyze_semantic_match(resume_text: str, job_description: str) -> Dict:
    """
    Main function for semantic analysis
    Combines TF-IDF similarity with keyword density analysis
    """
    
    # Overall semantic similarity
    tfidf_result = calculate_tfidf_similarity(resume_text, job_description)
    
    # Section-wise analysis
    section_scores = calculate_section_wise_similarity(resume_text, job_description)
    
    # Keyword density
    keyword_analysis = calculate_keyword_density(resume_text, job_description)
    
    # Combined score (weighted average)
    # 60% semantic similarity + 40% keyword coverage
    combined_score = round(
        (tfidf_result["overall_similarity"] * 0.6) + 
        (keyword_analysis["keyword_coverage_percentage"] * 0.4),
        2
    )
    
    # Generate insights
    insights = generate_semantic_insights(
        tfidf_result["overall_similarity"],
        section_scores,
        keyword_analysis,
        combined_score
    )
    
    return {
        "combined_match_score": combined_score,
        "semantic_similarity": tfidf_result["overall_similarity"],
        "keyword_coverage": keyword_analysis["keyword_coverage_percentage"],
        "top_matching_terms": tfidf_result["top_matching_terms"],
        "section_scores": section_scores,
        "keyword_details": keyword_analysis["detailed_coverage"][:10],  # Top 10 keywords
        "insights": insights,
        "recommendation": get_recommendation(combined_score)
    }

def generate_semantic_insights(
    semantic_score: float,
    section_scores: Dict[str, float],
    keyword_analysis: Dict,
    combined_score: float
) -> List[str]:
    """Generate human-readable insights from analysis"""
    insights = []
    
    # Overall assessment
    if combined_score >= 75:
        insights.append("🎯 Strong alignment! Your resume closely matches this job description.")
    elif combined_score >= 60:
        insights.append("✅ Good match overall. Your profile is relevant to this role.")
    elif combined_score >= 45:
        insights.append("⚠️ Moderate alignment. Some improvements could strengthen your application.")
    else:
        insights.append("❌ Limited alignment. Consider significant tailoring or skill development.")
    
    # Semantic similarity insight
    if semantic_score >= 70:
        insights.append(f"📊 Excellent semantic similarity ({semantic_score}%) - your experience aligns well contextually.")
    elif semantic_score >= 50:
        insights.append(f"📊 Decent semantic match ({semantic_score}%) - consider using more JD terminology.")
    else:
        insights.append(f"📊 Low semantic similarity ({semantic_score}%) - your resume language differs significantly from the JD.")
    
    # Keyword coverage insight
    coverage = keyword_analysis["keyword_coverage_percentage"]
    if coverage >= 70:
        insights.append(f"🔑 Strong keyword coverage ({coverage}%) - good keyword optimization!")
    elif coverage >= 50:
        insights.append(f"🔑 Moderate keyword coverage ({coverage}%) - add more relevant keywords from the JD.")
    else:
        insights.append(f"🔑 Low keyword coverage ({coverage}%) - many important JD keywords are missing.")
    
    # Section-specific insights
    if section_scores:
        best_section = max(section_scores.items(), key=lambda x: x[1])
        worst_section = min(section_scores.items(), key=lambda x: x[1])
        
        if best_section[1] >= 60:
            insights.append(f"💪 Strong {best_section[0]} section ({best_section[1]}%)")
        
        if worst_section[1] < 40 and len(section_scores) > 1:
            insights.append(f"⚡ Consider strengthening your {worst_section[0]} section ({worst_section[1]}%)")
    
    return insights

def get_recommendation(score: float) -> str:
    """Get application recommendation based on score"""
    if score >= 75:
        return "HIGHLY RECOMMENDED - Apply with confidence!"
    elif score >= 60:
        return "RECOMMENDED - Good fit, apply and highlight your strengths"
    elif score >= 45:
        return "CONSIDER - Tailor your resume before applying"
    else:
        return "NOT RECOMMENDED - Significant skill gap, consider upskilling first"