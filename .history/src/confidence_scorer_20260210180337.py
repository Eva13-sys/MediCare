import re

def extract_confidence(response_text: str) -> tuple:
    confidence_patterns = [
        r'Confidence:\s*(\d+)%',
        r'(\d+)%\s*confident',
        r'confidence.*?(\d+)%'
    ]
    
    confidence = None
    for pattern in confidence_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            confidence = int(match.group(1))
            response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)
            break
    
    # If no explicit confidence found, estimate based on language
    if confidence is None:
        confidence = estimate_confidence(response_text)
    
    return response_text.strip(), confidence


def estimate_confidence(text: str) -> int:
    """
    Estimate confidence based on language used in response
    """
    text_lower = text.lower()
    
    # High confidence indicators
    if any(phrase in text_lower for phrase in ['definitely', 'certainly', 'clearly', 'absolutely']):
        return 95
    
    # Medium-high confidence
    if any(phrase in text_lower for phrase in ['likely', 'probably', 'typically', 'generally']):
        return 75
    
    # Low confidence indicators
    if any(phrase in text_lower for phrase in ['might', 'may', 'possibly', 'perhaps', 'unsure', "don't know"]):
        return 40
    
    # Default medium confidence
    return 70


def get_confidence_label(score: int) -> str:
    """Get human-readable confidence label"""
    if score >= 85:
        return "High"
    elif score >= 60:
        return "Medium"
    else:
        return "Low"


def get_confidence_color(score: int) -> str:
    """Get color for UI display"""
    if score >= 85:
        return "green"
    elif score >= 60:
        return "orange"
    else:
        return "red"