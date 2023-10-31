import logging
import azure.functions as func

import os
import openai
import json
from main.curator import Curator


def main(
    recordsclean: func.InputStream,
    records: func.Out[bytes],
    embeddings: func.Out[bytes],
):
    logging.info(
        f"Python blob trigger function processing blob \n"
        f"Name: {recordsclean.name}\n"
        f"Blob Size: {recordsclean.length} bytes"
    )

    # Set OpenAI key
    key = os.environ["OPENAI_KEY"]
    openai.api_key = key

    # Load settings
    with open("./settings/settings.json", "r") as f:
        settings = json.load(f)

    # Get cleaned data
    cleaned_data = json.loads(recordsclean.read().decode("utf-8"))

    logging.info("Finished preparing Curator.")

    # Create and run preprocessor
    cur = Curator(settings)
    recs, embs = cur.run(cleaned_data)
    logging.info("Finished curating data.")

    # Convert dictionaries "recs" and "embs" to json bytes
    recs_bytes = json.dumps(recs).encode("utf-8")
    embs_bytes = json.dumps(embs).encode("utf-8")

    # Write json bytes to output blobs
    records.set(recs_bytes)
    embeddings.set(embs_bytes)

    logging.info("Python blob trigger function finished processing blob")
