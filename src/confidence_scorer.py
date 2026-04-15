import re

def extract_confidence(response_text: str) -> tuple:
    match = re.search(r'Confidence:\s*(\d{1,3})%', response_text)

    if match:
        confidence = int(match.group(1))
        confidence = min(confidence, 100)
        response_text = re.sub(r'Confidence:\s*\d{1,3}%', '', response_text)
    else:
        confidence = estimate_confidence(response_text)

    return response_text.strip(), confidence


def estimate_confidence(text: str) -> int:
    text_lower = text.lower()

    if any(w in text_lower for w in ['definitely', 'certainly']):
        return 90
    if any(w in text_lower for w in ['likely', 'generally']):
        return 75
    if any(w in text_lower for w in ['maybe', 'possibly']):
        return 50

    return 70


def get_confidence_label(score: int) -> str:
    if score >= 85:
        return "High"
    elif score >= 60:
        return "Medium"
    return "Low"


def get_confidence_color(score: int) -> str:
    if score >= 85:
        return "green"
    elif score >= 60:
        return "orange"
    return "red"