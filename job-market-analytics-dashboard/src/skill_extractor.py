from collections import Counter

COMMON_SKILLS = [
    'python',
    'sql',
    'power bi',
    'tableau',
    'aws',
    'azure',
    'docker',
    'kubernetes',
    'excel',
    'tensorflow',
    'pandas',
    'machine learning',
    'spark',
    'java',
    'javascript',
    'react'
]


def extract_skills(job_descriptions):
    skills = []

    for description in job_descriptions:
        if not isinstance(description, str):
            continue

        text = description.lower()

        for skill in COMMON_SKILLS:
            if skill in text:
                skills.append(skill)

    return Counter(skills)
