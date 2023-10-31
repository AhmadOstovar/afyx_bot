import openai
import numpy as np
import tiktoken
from main.utils.embedding import get_embedding


class Completer:
    """A ChatGPT based bot that uses prompts for context"""

    def __init__(
        self, settings: dict, records: dict, embeddings: dict
    ) -> None:
        """Initialize Completer

        Args:
            settings (dict): Settings dictionary
            records (dict): Records dictionary
            embeddings (dict): Embeddings dictionary
        """

        self.embeddings = embeddings
        self.records = records

        self.embedding_model = settings["EMBEDDING_MODEL"]
        self.max_section_len = settings["MAX_SECTION_LEN"]
        self.separator = settings["SEPARATOR"]
        self.completions_api_params = settings["COMPLETIONS_API_PARAMS"]

        self.encoding = tiktoken.get_encoding(settings["ENCODING"])
        self.separator_len = len(self.encoding.encode(settings["SEPARATOR"]))

    def vector_similarity(self, x: list[float], y: list[float]) -> float:
        """Returns the similarity between two vectors

        Because OpenAI Embeddings are normalized to length 1, the
        cosine similarity is the same as the dot product.

        Args:
            x (list[float]): Vector 1
            y (list[float]): Vector 2

        Returns:
            float: Similarity
        """

        return np.dot(np.array(x), np.array(y))

    def order_record_sections_by_query_similarity(
        self, query: str, contexts: dict[(str, str), np.array]
    ) -> list[(float, (str, str))]:
        """Find the most relevant sections for the supplied query

        Find the query embedding for the supplied query, and compare
        it against all of the pre-calculated record embeddings to find
        the most relevant sections.

        Return the list of record sections, sorted by relevance in
        descending order.

        Args:
            query (str): Query to find relevant sections for
            contexts (dict[(str, str), np.array]): Record embeddings

        Returns:
            list[(float, (str, str))]: List of record sections, sorted
                by relevance
        """

        query_embedding = get_embedding(query, self.embedding_model)

        record_similarities = sorted(
            [
                (
                    self.vector_similarity(query_embedding, doc_embedding),
                    doc_index,
                )
                for doc_index, doc_embedding in contexts.items()
            ],
            reverse=True,
        )

        return record_similarities

    def construct_prompt(self, question: str) -> tuple[str, list[str]]:
        """Construct a prompt for the supplied question

        Args:
            question (str): Question to construct a prompt for

        Returns:
            tuple[str, list[str]]: Prompt and list of record section
                indexes
        """

        most_relevant_record_sections = (
            self.order_record_sections_by_query_similarity(
                question, self.embeddings
            )
        )

        chosen_sections = []
        chosen_sections_len = 0
        chosen_sections_indexes = []

        for _, section_index in most_relevant_record_sections:
            # Add contexts until we run out of space.
            record_section = self.records[section_index]

            chosen_sections_len += (
                record_section["tokens"] + self.separator_len
            )
            if chosen_sections_len > self.max_section_len:
                break

            chosen_sections.append(
                self.separator + record_section["content"].replace("\n", " ")
            )
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        refs = [self.records[key]["ref"] for key in chosen_sections_indexes]

        header_start = (
            "Answer the question as truthfully as possible using the "
            "provided context, and if the answer is not contained within the "
            "text below, say 'Can you please rephrase? I am not sure what to "
            "answer.'\n\nContext:\n"
        )

        header = (
            header_start
            + "".join(chosen_sections)
            + "\n\n Q: "
            + question
            + "\n A:"
        )

        return header, refs

    def answer(self, query: str) -> str:
        """Answer the supplied query

        Args:
            query (str): Query to answer

        Returns:
            str: Answer
        """

        prompt, refs = self.construct_prompt(query)

        response = openai.Completion.create(
            prompt=prompt,
            **self.completions_api_params,
        )

        return response["choices"][0]["text"].strip(" \n"), refs
