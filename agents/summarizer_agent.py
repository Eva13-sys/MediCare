def summarize(chatModel, text):
    prompt = f"Summarize this medical text:\n{text}"
    return chatModel.invoke(prompt).content