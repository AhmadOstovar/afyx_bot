import openai
import numpy as np
import tiktoken
from main.utils.embedding import get_embedding


class Chatbot:
    def __init__(self, settings: dict, records: dict, embeddings: dict) -> None:
        """Initialize the Chatbot
        Args:
            settings (dict): Settings for the chatbot.
            records (dict): The additional information supplied.
            embeddings (dict): The embeddings of the records.
        """

        # Settings
        self.embedding_model = settings["EMBEDDING_MODEL"]
        self.max_section_len = settings["MAX_SECTION_LEN"]
        self.query_record_similarity_cutoff = settings["QUERY_RECORD_SIMILARITY_CUTOFF"]
        self.references_limit = settings["REFERENCES_LIMIT"]
        self.separator = settings["SEPARATOR"]
        self.completions_api_params = settings["CHATBOT_API_PARAMS"]
        self.encoding = tiktoken.get_encoding(settings["ENCODING"])
        self.separator_len = len(self.encoding.encode(settings["SEPARATOR"]))

        self.embeddings = embeddings
        self.records = records

        self.pre_prompt = settings["PRE_PROMPT"]

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
                (self.vector_similarity(query_embedding, doc_embedding), doc_index)
                for doc_index, doc_embedding in contexts.items()
            ],
            reverse=True,
        )

        return record_similarities

    def add_context_to_query(self, query: str, record_sections: list[int, str]):
        """Construct a prompt for the supplied query

        Args:
            query (str): Query to construct a prompt for

        Returns:
            messages (list[dict]): A list of the history formatted for the openai.ChatCompletion endpoint
            refs (list[str]): A list of relevant references
        """

        chosen_sections = []
        chosen_sections_indexes = []

        for _, section_index in record_sections:
            # Add contexts until we run out of space.
            record_section = self.records[section_index]

            """chosen_sections_len += (
                record_section["tokens"] + self.separator_len
            )
            if chosen_sections_len > self.max_section_len:
                break"""

            chosen_sections.append(
                self.separator + record_section["content"].replace("\n", " ")
            )
            chosen_sections_indexes.append(str(section_index))

        # Useful diagnostic information
        refs = [self.records[key]["ref"] for key in chosen_sections_indexes]

        query_instructions = (
            "Answer the query as truthfully as possible using the "
            "provided context, and if the answer is not contained within the "
            "text below, say 'Can you please rephrase? I am not sure what to "
            "answer.'\n\nContext:\n"
        )

        query_with_context = (
            query_instructions + "".join(chosen_sections) + "\n\n Q: " + query + "\n A:"
        )

        return query_with_context, refs

    def answer(self, messages: list, query: str) -> str:
        """Answer the supplied query

        Args:
            messages (list): Previous messages, synonym: history
            query (str): Query to answer

        Returns:
            str: Answer
        """

        # Finds records that may be related to the query
        most_relevant_record_sections = self.order_record_sections_by_query_similarity(
            query, self.embeddings
        )

        # Filters the records according to the similarity cutoff and reference limits defined in settings
        filtered_record_sections = [
            tup
            for tup in most_relevant_record_sections
            if tup[0] >= self.query_record_similarity_cutoff
        ][: self.references_limit]

        # Adds context to query if there are relevant records
        if len(filtered_record_sections) > 0:
            query, refs = self.add_context_to_query(query, filtered_record_sections)
        else:
            refs = []

        # Append the query to the messages
        messages.insert(0, {"role": "system", "content": self.pre_prompt})
        messages.append({"role": "user", "content": query})

        # Requests a response from OpenAIs API
        response = openai.ChatCompletion.create(
            messages=messages,
            **self.completions_api_params,
        )["choices"][0]["message"]["content"]

        return response, refs
