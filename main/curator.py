from transformers import GPT2TokenizerFast
from nltk.tokenize import sent_tokenize
from main.utils.embedding import get_embedding


class Curator:
    """Curator class to parse and curate records"""

    def __init__(self, settings: dict) -> None:
        """Initialize curator

        Args:
            settings (dict): Settings dictionary
        """

        self.embedding_model = settings["EMBEDDING_MODEL"]
        self.default_maximum_content_length = settings[
            "DEFAULT_MAXIMUM_CONTENT_LENGTH"
        ]
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string

        Args:
            text (str): Text to count tokens in

        Returns:
            int: Number of tokens
        """

        return len(self.tokenizer.encode(text))

    def reduce_long_text(self, text: str, max_len: int) -> str:
        """Reduce a long text to a maximum of `max_len` tokens

        Reducing may potentially cut at a sentence end.

        Args:
            text (str): Text to reduce
            max_len (int, optional): Maximum number of tokens. Defaults to
                self.default_maximum_content_length.

        Returns:
            str: Reduced text
        """

        n_tokens = self.count_tokens(text)
        if n_tokens > max_len:
            sentences = sent_tokenize(text.replace("\n", " "))
            ntokens = 0
            for i, sentence in enumerate(sentences):
                ntokens += 1 + self.count_tokens(sentence)
                if ntokens > max_len:
                    return ". ".join(sentences[:i][:-1]) + "."

        return text

    def parse_record(self, record: str) -> str:
        """Parse single record

        Args:
            record (str): Record to parse

        Returns:
            str: Parsed record
        """

        record["content"] = self.reduce_long_text(
            record["content"], self.default_maximum_content_length
        )
        record["tokens"] = self.count_tokens(record["content"])

        return record

    def run(self, all_records: dict) -> None:
        """Parse all records and save result to files

        records structure:
            {
                "title": {
                    "ref": "https://...",
                    "content": "..."
                }
                ...
            }

        Args:
            all_records (dict): All records to parse

        Returns:
            dict: Parsed records
        """

        # Parse all records
        all_records = {
            title: self.parse_record(record)
            for title, record in all_records.items()
        }

        # Calculate embeddings for all records and store as a dictionary
        embeddings = {
            title: get_embedding(record["content"], self.embedding_model)
            for title, record in all_records.items()
        }

        return all_records, embeddings
