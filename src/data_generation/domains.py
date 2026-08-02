"""
Domain taxonomy for memory state transition dataset generation.

Defines 10 life domains to ensure broad coverage across the dataset.
Each generated triplet can be tagged with a domain for auditing.
"""

DOMAINS = [
    "health",
    "finance",
    "career",
    "education",
    "relationships",
    "travel",
    "hobbies",
    "beliefs_values",
    "technology",
    "daily_life",
]

DOMAIN_DESCRIPTIONS = {
    "health": "Medical conditions, fitness, diet, mental health, sleep, wellness habits",
    "finance": "Income, savings, investments, debt, budgeting, financial goals",
    "career": "Jobs, promotions, workplace dynamics, professional development, entrepreneurship",
    "education": "Degrees, courses, certifications, learning goals, academic activities",
    "relationships": "Family, friends, romantic partners, social connections, community",
    "travel": "Trips, relocation, exploration, cultural experiences, travel planning",
    "hobbies": "Sports, arts, music, gaming, reading, entertainment, creative pursuits",
    "beliefs_values": "Politics, religion, ethics, worldview, personal philosophy, activism",
    "technology": "Devices, software, AI, gadgets, digital tools, social media",
    "daily_life": "Routines, errands, home, pets, transportation, daily habits",
}

DOMAIN_KEYWORDS = {
    "health": ["doctor", "hospital", "diet", "exercise", "symptom", "diagnosis", "allergy",
               "medication", "therapy", "fitness", "workout", "nutrition", "sleep", "stress",
               "mental health", "blood pressure", "surgery", "recovery", "weight", "yoga"],
    "finance": ["salary", "savings", "investment", "debt", "loan", "mortgage", "budget",
                "401k", "credit card", "bank", "income", "tax", "retirement", "stock",
                "crypto", "payment", "insurance", "raise", "bonus", "spending"],
    "career": ["job", "promotion", "manager", "startup", "freelance", "client", "meeting",
               "deadline", "project", "colleague", "boss", "interview", "resume", "office",
               "remote work", "team", "leadership", "career change", "promotion", "hire"],
    "education": ["university", "college", "degree", "course", "major", "professor",
                  "semester", "exam", "study", "thesis", "PhD", "master", "bachelor",
                  "certification", "bootcamp", "online course", "homework", "graduate"],
    "relationships": ["married", "wife", "husband", "girlfriend", "boyfriend", "partner",
                      "friend", "family", "parent", "child", "sibling", "divorce", "dating",
                      "wedding", "breakup", "mom", "dad", "son", "daughter", "relative"],
    "travel": ["trip", "flight", "hotel", "vacation", "backpacking", "visa", "passport",
               "destination", "itinerary", "tour", "airbnb", "travel", "visit", "abroad",
               "sightseeing", "beach", "mountain", "city", "country", "culture"],
    "hobbies": ["guitar", "piano", "painting", "photography", "hiking", "cooking", "gaming",
                "reading", "movie", "music", "sport", "running", "cycling", "swimming",
                "chess", "board game", "craft", "instrument", "dance", "singing"],
    "beliefs_values": ["religion", "political", "vote", "belief", "value", "ethics",
                       "philosophy", "activism", "protest", "volunteer", "charity", "donate",
                       "spiritual", "worldview", "principle", "rights", "justice", "moral"],
    "technology": ["computer", "phone", "app", "software", "AI", "laptop", "device",
                   "coding", "programming", "website", "data", "algorithm", "tech",
                   "startup", "cloud", "security", "privacy", "internet", "digital"],
    "daily_life": ["morning", "routine", "commute", "errand", "grocery", "cooking", "cleaning",
                   "laundry", "pet", "dog", "cat", "apartment", "house", "neighbor",
                   "sleep", "wake", "meal", "breakfast", "dinner", "weekend"],
}


def get_domain_prompt_section(domain: str) -> str:
    """
    Generate the domain-specific prompt section to inject into templates.

    Args:
        domain: One of the DOMAINS keys.

    Returns:
        A string to append to the prompt template.
    """
    if domain not in DOMAIN_DESCRIPTIONS:
        return ""

    desc = DOMAIN_DESCRIPTIONS[domain]
    keywords = DOMAIN_KEYWORDS.get(domain, [])
    kw_str = ", ".join(keywords[:15])

    return f"""
## Domain Focus: {domain.replace('_', ' ').title()}

**All examples in this batch MUST be set in the "{domain}" domain.**

Domain scope: {desc}

Suggested themes and keywords: {kw_str}

Make sure every example's M, delta_M, and M_prime are clearly about {domain.replace('_', ' ')} topics.
Do NOT generate examples from other domains in this batch.
"""