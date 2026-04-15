def get_advice(chatModel, symptoms):
    prompt = f"""
    Symptoms: {symptoms}

    Provide:
    - Possible conditions
    - Precautions
    - Whether doctor visit is needed
    """
    return chatModel.invoke(prompt).content