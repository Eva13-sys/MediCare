from src.confidence_scorer import extract_confidence, get_confidence_label

def validate(answer):
    clean, score = extract_confidence(answer)
    label = get_confidence_label(score)

    return {
        "clean_answer": clean,
        "confidence": score,
        "label": label
    }