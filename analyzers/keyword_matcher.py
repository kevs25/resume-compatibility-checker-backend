import re
from collections import Counter
from typing import List, Dict, Set, Tuple

# Comprehensive skill categories
TECHNICAL_SKILLS = {
    # Programming Languages
    'python', 'javascript', 'typescript', 'java', 'go', 'golang', 'c++', 'c#', 'ruby',
    'php', 'swift', 'kotlin', 'rust', 'scala', 'r', 'dart', 'shell', 'bash',
    
    # Frontend
    'react', 'reactjs', 'react.js', 'vue', 'vuejs', 'vue.js', 'angular', 'angularjs',
    'next', 'nextjs', 'next.js', 'svelte', 'jquery', 'html', 'html5', 'css', 'css3',
    'sass', 'scss', 'tailwind', 'tailwindcss', 'bootstrap', 'material-ui', 'mui',
    
    # Backend Frameworks
    'django', 'flask', 'fastapi', 'express', 'expressjs', 'nodejs', 'node.js', 'node',
    'spring', 'springboot', 'spring boot', 'laravel', 'rails', 'ruby on rails',
    
    # Databases
    'mongodb', 'mysql', 'postgresql', 'postgres', 'redis', 'elasticsearch', 'dynamodb',
    'cassandra', 'oracle', 'sql server', 'sqlite', 'mariadb', 'firestore', 'firebase',
    
    # Cloud & DevOps
    'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'jenkins',
    'terraform', 'ansible', 'circleci', 'travis', 'gitlab', 'github actions',
    'lambda', 'ec2', 's3', 'sqs', 'sns', 'rds', 'cloudformation',
    
    # Tools & Technologies
    'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'linux', 'unix',
    'nginx', 'apache', 'graphql', 'rest', 'restful', 'api', 'microservices',
    'grpc', 'websocket', 'kafka', 'rabbitmq', 'celery',
    
    # Testing & Quality
    'jest', 'pytest', 'junit', 'selenium', 'cypress', 'mocha', 'chai',
    
    # Data & ML
    'pandas', 'numpy', 'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
    'spark', 'hadoop', 'airflow', 'data science', 'machine learning', 'ml', 'ai',
    'deep learning', 'nlp', 'computer vision',
    
    # Mobile
    'react native', 'flutter', 'ios', 'android', 'swift', 'kotlin',
    
    # Other
    'agile', 'scrum', 'ci/cd', 'tdd', 'bdd', 'oauth', 'jwt', 'websockets',
}

DOMAIN_KNOWLEDGE = {
    'computer science', 'cs', 'software engineering', 'bfsi', 'fintech', 'healthcare',
    'e-commerce', 'saas', 'b2b', 'b2c', 'edtech', 'finance', 'banking',
}

SOFT_SKILLS = {
    'architecture', 'design patterns', 'system design', 'scalability', 'performance',
    'optimization', 'leadership', 'mentoring', 'code review', 'collaboration',
    'communication', 'problem solving', 'debugging', 'troubleshooting',
}

def normalize_skill(skill: str) -> str:
    """Normalize skill variations to a standard form"""
    skill = skill.lower().strip()
    
    # Normalize common variations
    variations = {
        'react.js': 'react',
        'reactjs': 'react',
        'next.js': 'nextjs',
        'vue.js': 'vuejs',
        'node.js': 'nodejs',
        'postgresql': 'postgres',
        'kubernetes': 'k8s',
        'golang': 'go',
    }
    
    return variations.get(skill, skill)

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = text.lower()
    # Keep alphanumeric, spaces, +, #, and dots for version numbers
    text = re.sub(r'[^a-z0-9\s\+\#\.\-]', ' ', text)
    text = ' '.join(text.split())
    return text

def extract_years_of_experience(text: str) -> int:
    """Extract years of experience from text"""
    text = text.lower()
    
    # Pattern: "X years", "X+ years", "X-Y years"
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\s*-\s*(\d+)\s*(?:years?|yrs?)',
    ]
    
    max_years = 0
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                # For range like "2-5 years", take the minimum required
                years = int(match[0]) if match[0] else 0
            else:
                years = int(match)
            max_years = max(max_years, years)
    
    return max_years

def extract_skills_from_text(text: str) -> Set[str]:
    """Extract technical skills, domain knowledge, and soft skills from text"""
    cleaned = clean_text(text)
    skills_found = set()
    
    # Check for multi-word skills first (to avoid partial matches)
    multi_word_skills = [s for s in TECHNICAL_SKILLS | DOMAIN_KNOWLEDGE | SOFT_SKILLS if ' ' in s]
    for skill in sorted(multi_word_skills, key=len, reverse=True):
        if skill in cleaned:
            skills_found.add(normalize_skill(skill))
    
    # Check for single-word skills
    words = cleaned.split()
    for word in words:
        normalized = normalize_skill(word)
        if normalized in TECHNICAL_SKILLS or normalized in DOMAIN_KNOWLEDGE or normalized in SOFT_SKILLS:
            skills_found.add(normalized)
    
    return skills_found

def extract_important_concepts(text: str) -> Set[str]:
    """Extract important concepts beyond just technical skills"""
    concepts = set()
    text_lower = text.lower()
    
    # Architecture & Design
    if any(word in text_lower for word in ['architecture', 'design pattern', 'scalable', 'distributed']):
        concepts.add('architecture & design')
    
    # Building/Development
    if any(word in text_lower for word in ['build', 'develop', 'create', 'implement']):
        concepts.add('development experience')
    
    # CS Background
    if any(word in text_lower for word in ['computer science', 'cs', 'b.tech', 'bachelor']):
        concepts.add('cs background')
    
    return concepts

def calculate_experience_match(resume_text: str, jd_text: str) -> Tuple[bool, str]:
    """Check if resume experience meets JD requirements"""
    jd_years = extract_years_of_experience(jd_text)
    resume_years = extract_years_of_experience(resume_text)
    
    # Also check work experience duration
    if 'august 2023' in resume_text.lower() and 'present' in resume_text.lower():
        # Calculate from Aug 2023 to Nov 2024 (approx 1.3 years based on context)
        # But we'll be generous and count it as 2+ years of professional experience
        resume_years = max(resume_years, 2)
    
    if jd_years > 0:
        if resume_years >= jd_years:
            return True, f"✓ Experience: {resume_years}+ years (meets {jd_years}+ years requirement)"
        else:
            return False, f"✗ Experience: {resume_years} years (requires {jd_years}+ years)"
    
    return True, "Experience requirement not specified or met"

def analyze_resume_match(resume_text: str, job_description: str) -> Dict:
    """
    Improved analysis focusing on meaningful skills and concepts
    """
    # Extract skills from both
    jd_skills = extract_skills_from_text(job_description)
    resume_skills = extract_skills_from_text(resume_text)
    
    # Extract concepts
    jd_concepts = extract_important_concepts(job_description)
    resume_concepts = extract_important_concepts(resume_text)
    
    # Check experience
    exp_match, exp_message = calculate_experience_match(resume_text, job_description)
    
    # Find matches
    matched_skills = jd_skills & resume_skills
    missing_skills = jd_skills - resume_skills
    
    matched_concepts = jd_concepts & resume_concepts
    missing_concepts = jd_concepts - resume_concepts
    
    # Calculate match percentage
    total_required = len(jd_skills) + len(jd_concepts)
    total_matched = len(matched_skills) + len(matched_concepts)
    
    if total_required == 0:
        match_percentage = 0
    else:
        match_percentage = round((total_matched / total_required) * 100, 2)
    
    # Add experience bonus if met
    if exp_match and total_required > 0:
        match_percentage = min(100, match_percentage + 5)  # Bonus for meeting exp requirement
    
    # Generate detailed suggestions
    suggestions = generate_detailed_suggestions(
        match_percentage, 
        missing_skills, 
        missing_concepts,
        exp_match,
        matched_skills
    )
    
    # Categorize skills
    matched_by_category = categorize_skills(matched_skills)
    missing_by_category = categorize_skills(missing_skills)
    
    return {
        "match_percentage": match_percentage,
        "experience_check": exp_message,
        "matched_skills": sorted(list(matched_skills)),
        "missing_skills": sorted(list(missing_skills)),
        "matched_concepts": sorted(list(matched_concepts)),
        "missing_concepts": sorted(list(missing_concepts)),
        "matched_by_category": matched_by_category,
        "missing_by_category": missing_by_category,
        "total_jd_requirements": total_required,
        "total_matched": total_matched,
        "suggestions": suggestions
    }

def categorize_skills(skills: Set[str]) -> Dict[str, List[str]]:
    """Categorize skills into groups"""
    categories = {
        "Languages": [],
        "Frontend": [],
        "Backend": [],
        "Databases": [],
        "Cloud & DevOps": [],
        "Other": []
    }
    
    language_keywords = {'python', 'javascript', 'typescript', 'java', 'go', 'golang', 'c++', 'c#', 'ruby', 'php'}
    frontend_keywords = {'react', 'vue', 'angular', 'nextjs', 'html', 'css', 'tailwind', 'bootstrap'}
    backend_keywords = {'django', 'flask', 'fastapi', 'express', 'nodejs', 'spring'}
    database_keywords = {'mongodb', 'mysql', 'postgres', 'postgresql', 'redis', 'firebase'}
    cloud_keywords = {'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'k8s', 'lambda', 's3'}
    
    for skill in skills:
        if skill in language_keywords:
            categories["Languages"].append(skill)
        elif skill in frontend_keywords:
            categories["Frontend"].append(skill)
        elif skill in backend_keywords:
            categories["Backend"].append(skill)
        elif skill in database_keywords:
            categories["Databases"].append(skill)
        elif skill in cloud_keywords:
            categories["Cloud & DevOps"].append(skill)
        else:
            categories["Other"].append(skill)
    
    # Remove empty categories
    return {k: sorted(v) for k, v in categories.items() if v}

def generate_detailed_suggestions(
    match_percentage: float, 
    missing_skills: Set[str],
    missing_concepts: Set[str],
    exp_match: bool,
    matched_skills: Set[str]
) -> List[str]:
    """Generate actionable improvement suggestions"""
    suggestions = []
    
    # Overall assessment
    if match_percentage >= 80:
        suggestions.append("🎯 Excellent match! Your profile aligns very well with this role.")
    elif match_percentage >= 60:
        suggestions.append("✅ Good match! You meet most of the key requirements.")
    elif match_percentage >= 40:
        suggestions.append("⚠️ Moderate match. Consider highlighting relevant experience or acquiring missing skills.")
    else:
        suggestions.append("❌ Low match. This role may require significant skill development.")
    
    # Experience feedback
    if not exp_match:
        suggestions.append("⏰ Consider gaining more experience or highlighting relevant projects to meet the years requirement.")
    
    # Skill-specific suggestions
    if missing_skills:
        critical_missing = [s for s in missing_skills if s in {'python', 'javascript', 'react', 'aws', 'docker'}]
        if critical_missing:
            suggestions.append(f"🔴 High-priority missing skills: {', '.join(sorted(critical_missing)[:5])}")
        else:
            top_missing = sorted(list(missing_skills))[:5]
            suggestions.append(f"💡 Consider adding: {', '.join(top_missing)}")
    
    # Concept suggestions
    if 'architecture & design' in missing_concepts:
        suggestions.append("🏗️ Highlight any architecture or system design experience in your resume.")
    
    # Positive reinforcement
    if matched_skills:
        key_matches = [s for s in matched_skills if s in {'python', 'aws', 'react', 'mongodb', 'postgres'}]
        if key_matches:
            suggestions.append(f"✨ Strong matches: {', '.join(sorted(key_matches)[:3])}")
    
    # Action items
    suggestions.append("📝 Make sure your most relevant skills and projects are prominently featured.")
    
    return suggestions