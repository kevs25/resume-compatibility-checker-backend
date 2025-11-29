# """
# Hybrid analyzer combining:
# 1. Explicit skill matching
# 2. Experience validation
# 3. Semantic understanding
# 4. Smart weighting

# This gives more accurate results than pure semantic similarity
# """

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import re
# from typing import Dict, List, Set, Tuple
# from datetime import datetime

# # Technical skills database
# TECH_SKILLS = {
#     'python', 'javascript', 'typescript', 'java', 'go', 'golang', 'c++', 'c#',
#     'react', 'reactjs', 'vue', 'angular', 'nextjs', 'next.js',
#     'django', 'flask', 'fastapi', 'express', 'nodejs', 'spring',
#     'mongodb', 'mysql', 'postgresql', 'postgres', 'redis', 'dynamodb',
#     'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'lambda', 's3', 'ec2',
#     'git', 'linux', 'ci/cd', 'jenkins', 'terraform',
# }

# SOFT_SKILLS = {
#     'architecture', 'design patterns', 'scalability', 'abstractions',
#     'system design', 'optimization', 'leadership', 'mentoring',
# }

# # Global model
# _model = None

# def get_model():
#     global _model
#     if _model is None:
#         _model = SentenceTransformer('all-MiniLM-L6-v2')
#     return _model

# def normalize_skill(skill: str) -> str:
#     """Normalize skill names"""
#     skill = skill.lower().strip()
#     mapping = {
#         'react.js': 'react', 'reactjs': 'react',
#         'next.js': 'nextjs', 'vue.js': 'vue',
#         'node.js': 'nodejs', 'postgresql': 'postgres',
#         'golang': 'go', 'kubernetes': 'k8s',
#     }
#     return mapping.get(skill, skill)

# def extract_skills_from_text(text: str) -> Set[str]:
#     """Extract technical skills from text"""
#     text_lower = text.lower()
#     found_skills = set()
    
#     # Check multi-word skills first
#     all_skills = TECH_SKILLS | SOFT_SKILLS
#     multi_word = [s for s in all_skills if ' ' in s]
    
#     for skill in sorted(multi_word, key=len, reverse=True):
#         if skill in text_lower:
#             found_skills.add(normalize_skill(skill))
    
#     # Check single words
#     words = re.findall(r'\b\w+\b', text_lower)
#     for word in words:
#         normalized = normalize_skill(word)
#         if normalized in TECH_SKILLS or normalized in SOFT_SKILLS:
#             found_skills.add(normalized)
    
#     return found_skills

# def calculate_years_of_experience(resume_text: str) -> float:
#     """Calculate years of experience from work history"""
#     text_lower = resume_text.lower()
    
#     # Look for date patterns in work experience
#     # Pattern: "Month Year – Present" or "Month Year – Month Year"
#     date_patterns = [
#         r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\s*[-–—]\s*(present|current)',
#         r'(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})\s*[-–—]\s*(january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})',
#     ]
    
#     total_years = 0
#     current_year = datetime.now().year
    
#     for pattern in date_patterns:
#         matches = re.findall(pattern, text_lower, re.IGNORECASE)
#         for match in matches:
#             try:
#                 start_year = int(match[1])
                
#                 if 'present' in str(match[-1]).lower() or 'current' in str(match[-1]).lower():
#                     # Still working
#                     years = current_year - start_year
#                 else:
#                     # Extract end year
#                     end_year = int(match[-1]) if match[-1].isdigit() else current_year
#                     years = end_year - start_year
                
#                 total_years += years
#             except:
#                 continue
    
#     return total_years

# def extract_required_years(jd_text: str) -> int:
#     """Extract required years from JD"""
#     text = jd_text.lower()
    
#     # Patterns for years requirement
#     patterns = [
#         r'(\d+)\+?\s*(?:to|-)\s*(\d+)\s*(?:years?|yrs?)',  # "2-5 years"
#         r'(\d+)\+\s*(?:years?|yrs?)',  # "2+ years"
#         r'(\d+)\s*(?:years?|yrs?)',  # "2 years"
#     ]
    
#     for pattern in patterns:
#         match = re.search(pattern, text)
#         if match:
#             # Return minimum requirement
#             return int(match.group(1))
    
#     return 0

# def check_experience_requirement(resume_text: str, jd_text: str) -> Tuple[bool, float, float, str]:
#     """Check if experience requirement is met"""
#     required_years = extract_required_years(jd_text)
#     resume_years = calculate_years_of_experience(resume_text)
    
#     # Be lenient: if within 1 year, consider it met
#     meets_requirement = (resume_years >= required_years - 0.5) if required_years > 0 else True
    
#     if required_years == 0:
#         message = "No specific experience requirement found"
#     elif meets_requirement:
#         message = f"✓ {resume_years} years meets {required_years}+ requirement"
#     else:
#         message = f"✗ {resume_years} years (needs {required_years}+)"
    
#     return meets_requirement, required_years, resume_years, message

# def extract_project_experience(resume_text: str) -> List[str]:
#     """Extract project descriptions from resume"""
#     projects = []
#     text_lower = resume_text.lower()
    
#     # Find projects section
#     project_section = re.search(r'(projects?)(.*?)(?=\n[A-Z]{2,}|\neducation|$)', 
#                                text_lower, re.DOTALL)
    
#     if project_section:
#         project_text = project_section.group(2)
#         # Split by bullet points or project titles
#         project_items = re.split(r'[•\n\-\*]', project_text)
#         projects = [p.strip() for p in project_items if len(p.strip()) > 30]
    
#     return projects

# def calculate_skill_match_score(resume_skills: Set[str], jd_skills: Set[str]) -> Dict:
#     """Calculate detailed skill matching score"""
#     matched = resume_skills & jd_skills
#     missing = jd_skills - resume_skills
#     extra = resume_skills - jd_skills
    
#     if len(jd_skills) == 0:
#         match_percentage = 100 if len(resume_skills) > 0 else 0
#     else:
#         match_percentage = (len(matched) / len(jd_skills)) * 100
    
#     # Categorize matched skills
#     matched_tech = matched & TECH_SKILLS
#     matched_soft = matched & SOFT_SKILLS
    
#     return {
#         'match_percentage': round(match_percentage, 2),
#         'matched_skills': sorted(list(matched)),
#         'matched_tech_skills': sorted(list(matched_tech)),
#         'matched_soft_skills': sorted(list(matched_soft)),
#         'missing_skills': sorted(list(missing)),
#         'total_required': len(jd_skills),
#         'total_matched': len(matched),
#     }

# def semantic_similarity_with_context(text1: str, text2: str) -> float:
#     """Calculate semantic similarity with better context handling"""
#     model = get_model()
    
#     # Clean and prepare texts
#     text1 = ' '.join(text1.split())
#     text2 = ' '.join(text2.split())
    
#     # Get embeddings
#     embeddings = model.encode([text1, text2])
    
#     # Calculate similarity
#     similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
#     return float(similarity)

# def analyze_requirement_matches(resume_text: str, jd_requirements: List[str]) -> List[Dict]:
#     """Analyze each JD requirement separately with hybrid approach"""
#     model = get_model()
#     results = []
    
#     # Get resume skills
#     resume_skills = extract_skills_from_text(resume_text)
    
#     for req in jd_requirements:
#         if len(req.strip()) < 10:
#             continue
        
#         # Extract skills from this requirement
#         req_skills = extract_skills_from_text(req)
        
#         # Semantic similarity
#         semantic_score = semantic_similarity_with_context(resume_text, req) * 100
        
#         # Skill overlap
#         matched_skills = resume_skills & req_skills
#         skill_overlap = (len(matched_skills) / len(req_skills) * 100) if req_skills else 0
        
#         # Combined score (60% semantic + 40% skill match)
#         combined_score = (semantic_score * 0.6) + (skill_overlap * 0.4)
        
#         # Determine status
#         if combined_score >= 70:
#             status = 'Strong Match'
#         elif combined_score >= 50:
#             status = 'Partial Match'
#         else:
#             status = 'Weak Match'
        
#         results.append({
#             'requirement': req.strip(),
#             'combined_score': round(combined_score, 2),
#             'semantic_score': round(semantic_score, 2),
#             'skill_overlap': round(skill_overlap, 2),
#             'matched_skills': sorted(list(matched_skills)),
#             'status': status
#         })
    
#     return sorted(results, key=lambda x: x['combined_score'], reverse=True)

# def extract_jd_requirements(jd_text: str) -> List[str]:
#     """Extract individual requirements from JD"""
#     # Split by newlines, bullets, or numbers
#     lines = re.split(r'[\n•\*\-]|\d+\.', jd_text)
#     requirements = [line.strip() for line in lines if len(line.strip()) > 10]
#     return requirements

# def calculate_hybrid_score(
#     skill_match: Dict,
#     experience_met: bool,
#     requirement_matches: List[Dict],
#     overall_semantic: float
# ) -> float:
#     """Calculate final hybrid score with smart weighting"""
    
#     # Base components
#     skill_score = skill_match['match_percentage']
    
#     # Experience bonus/penalty
#     experience_score = 100 if experience_met else 50
    
#     # Average requirement match
#     avg_req_match = sum(r['combined_score'] for r in requirement_matches) / len(requirement_matches) if requirement_matches else 0
    
#     # Overall semantic understanding
#     semantic_score = overall_semantic * 100
    
#     # Weighted combination
#     # 40% skills, 25% requirements, 20% semantic, 15% experience
#     final_score = (
#         (skill_score * 0.40) +
#         (avg_req_match * 0.25) +
#         (semantic_score * 0.20) +
#         (experience_score * 0.15)
#     )
    
#     return round(final_score, 2)

# def generate_insights(
#     final_score: float,
#     skill_match: Dict,
#     experience_met: bool,
#     requirement_matches: List[Dict]
# ) -> List[str]:
#     """Generate actionable insights"""
#     insights = []
    
#     # Overall assessment
#     if final_score >= 75:
#         insights.append("🎯 Excellent match! You're highly qualified for this role.")
#     elif final_score >= 60:
#         insights.append("✅ Strong candidate! You meet most key requirements.")
#     elif final_score >= 45:
#         insights.append("⚠️ Moderate fit. Emphasize your relevant experience and skills.")
#     else:
#         insights.append("❌ Limited alignment. Consider skill development or different roles.")
    
#     # Skill insights
#     if skill_match['match_percentage'] >= 70:
#         insights.append(f"💪 Strong technical skills match ({skill_match['match_percentage']:.0f}%)")
#     elif skill_match['match_percentage'] >= 50:
#         insights.append(f"📚 Good skill foundation ({skill_match['match_percentage']:.0f}%). Consider adding: {', '.join(skill_match['missing_skills'][:3])}")
#     else:
#         insights.append(f"⚡ Skill gap detected. Focus on: {', '.join(skill_match['missing_skills'][:3])}")
    
#     # Experience insight
#     if not experience_met:
#         insights.append("⏰ Experience requirement not fully met. Highlight relevant projects and achievements.")
    
#     # Requirement insights
#     strong_reqs = [r for r in requirement_matches if r['status'] == 'Strong Match']
#     if strong_reqs:
#         insights.append(f"✨ {len(strong_reqs)}/{len(requirement_matches)} requirements strongly matched")
    
#     # Matched skills insight
#     if skill_match['matched_tech_skills']:
#         top_skills = ', '.join(skill_match['matched_tech_skills'][:4])
#         insights.append(f"🔑 Key matching skills: {top_skills}")
    
#     return insights

# def get_recommendation(score: float) -> str:
#     """Get application recommendation"""
#     if score >= 75:
#         return "HIGHLY RECOMMENDED - Apply with confidence!"
#     elif score >= 60:
#         return "RECOMMENDED - Strong fit, tailor your application"
#     elif score >= 45:
#         return "CONSIDER - Address gaps in application materials"
#     else:
#         return "REVIEW - Significant gaps exist, consider upskilling"

# def analyze_hybrid(resume_text: str, job_description: str) -> Dict:
#     """
#     Main hybrid analysis function
#     Combines explicit skill matching, experience validation, and semantic understanding
#     """
    
#     # 1. Extract skills
#     resume_skills = extract_skills_from_text(resume_text)
#     jd_skills = extract_skills_from_text(job_description)
#     skill_match = calculate_skill_match_score(resume_skills, jd_skills)
    
#     # 2. Check experience
#     exp_met, req_years, resume_years, exp_message = check_experience_requirement(
#         resume_text, job_description
#     )
    
#     # 3. Extract and analyze requirements
#     jd_requirements = extract_jd_requirements(job_description)
#     requirement_matches = analyze_requirement_matches(resume_text, jd_requirements)
    
#     # 4. Overall semantic similarity
#     overall_semantic = semantic_similarity_with_context(resume_text, job_description)
    
#     # 5. Calculate hybrid score
#     final_score = calculate_hybrid_score(
#         skill_match, exp_met, requirement_matches, overall_semantic
#     )
    
#     # 6. Generate insights
#     insights = generate_insights(final_score, skill_match, exp_met, requirement_matches)
    
#     # 7. Get recommendation
#     recommendation = get_recommendation(final_score)
    
#     return {
#         'final_score': final_score,
#         'skill_analysis': skill_match,
#         'experience_analysis': {
#             'meets_requirement': exp_met,
#             'required_years': req_years,
#             'resume_years': resume_years,
#             'message': exp_message
#         },
#         'requirement_analysis': {
#             'matches': requirement_matches,
#             'total_requirements': len(requirement_matches),
#             'strong_matches': len([r for r in requirement_matches if r['status'] == 'Strong Match']),
#             'average_score': round(sum(r['combined_score'] for r in requirement_matches) / len(requirement_matches), 2) if requirement_matches else 0
#         },
#         'semantic_similarity': round(overall_semantic * 100, 2),
#         'insights': insights,
#         'recommendation': recommendation,
#         'analysis_method': 'Hybrid (Skills + Experience + Semantic)'
#     }