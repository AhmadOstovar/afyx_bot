import logging
import azure.functions as func

import os
import openai
import json
from ..main.chatbot import Chatbot


def main(
    req: func.HttpRequest,
    embeddingsbinary: bytes,
    recordsbinary: bytes,
) -> func.HttpResponse:
    logging.info("Python HTTP triggered function 'run_chatbot' processed a request.")

    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        query = req_body.get("prompt")
        history = req_body.get("history")

    if query:
        logging.info("Loading information.")
        # Set OpenAI key
        key = os.environ["OPENAI_KEY"]
        openai.api_key = key

        # Load settings
        with open("./settings/settings.json", "r") as f:
            settings = json.load(f)

        # Get record embeddings
        embeddings = json.loads(embeddingsbinary.decode("utf-8"))

        # Get record data
        records = json.loads(recordsbinary.decode("utf-8"))

        # Initialize the Chatbot with settings, the additional information (records) and the embeddings of those records
        Bot = Chatbot(
            settings=settings, records=records, embeddings=embeddings
        )

        # Ask the bot for an answer to the user query
        answer, refs = Bot.answer(
            messages=history,
            query=query,
        )

        logging.info("Finished responding.")

        # Return both answer and refs
        return func.HttpResponse(json.dumps({"answer": answer, "refs": refs}))

    else:
        return func.HttpResponse(
            "Python HTTP triggered function 'run_chatbot' executed successfully. Pass a query or a request body for a tailored response.",
            status_code=200,
        )
