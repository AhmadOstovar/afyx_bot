import logging
import azure.functions as func

import os
import openai
import json
from ..main.completer import Completer


def main(
    req: func.HttpRequest,
    embeddingsbinary: bytes,
    recordsbinary: bytes,
) -> func.HttpResponse:

    logging.info("Python HTTP triggered function 'run_completer' processed a request.")

    query = req.params.get("query")
    if not query:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            query = req_body.get("query")

    if query:

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

        # Create bot
        completer = Completer(settings, records, embeddings)

        logging.info("Finished preparing Completer.")

        # Get answer and refs
        answer, refs = completer.answer(query)
        logging.info("Finished querying Completer.")

        # Return both answer and refs
        return func.HttpResponse(json.dumps({"answer": answer, "refs": refs}))

    else:
        return func.HttpResponse(
            "Python HTTP triggered function 'run_completer' executed successfully. Pass a query or a request body for a tailored response.",
            status_code=200,
        )
