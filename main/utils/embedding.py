import openai


def get_embedding(text: str, model: str) -> list[float]:
    """Get embedding from text using OpenAI API

    Args:
        text (str): Text to get embedding from
        model (str): Model to use for embedding

    Returns:
        list[float]: Embedding
    """

    result = openai.Embedding.create(model=model, input=text)
    return result["data"][0]["embedding"]
