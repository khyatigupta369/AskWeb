from config import Config
import tiktoken  # tokenizer
import openai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AI:
    def __init__(self, config: Config):
        self._config = config
        self.setup()

    def setup(self):
        openai.api_key = self._config['open_ai_key']
        openai.proxy = self._config['open_ai_proxy']
        self._chat_model = self._config['open_ai_chat_model']
        self._use_stream = self._config['use_stream']
        self._encoding = tiktoken.encoding_for_model(self._config['gpt_version'])
        self._language = self._config['language']
        self._temperature = self._config['temperature']

    def _chat_stream(self, messages: list[dict], use_stream: bool = None) -> str:
        use_stream = use_stream if use_stream else self._use_stream
        response = openai.ChatCompletion.create(
            temperature=self._temperature,
            stream=use_stream,
            model=self._chat_model,
            messages=messages,
        )
        if use_stream:
            data = ''
            for chunk in response:
                if chunk.choice[0].delta.get('content', None):
                    data += chunk.choice[0].delta.content
                    print(data, end='')
            print()
            return data.strip()
        else:
            chunk = response.choice[0].message.content.strip()
            print(chunk)
            print(f"Total tokens used: {response.usage.total_tokens}, "
                  f"cost: ${response.usage.total_tokens / 1000 * 0.002}")

            return chunk

    def _num_of_tokens_in_string(self, string) -> int:
        return len(self._encoding.encode(string))

    def completion(self, query: str, context: list[str]):
        context = self.trim_texts(context)
        print(f"Number of query fragments:{len(context)}")

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(context))
        result = self._chat_stream([
            {'role': 'system',
             'content': f'You are a helpful AI article assistant. '
                        f'The following are the relevant article content fragments found from the article. '
                        f'The relevance is sorted from high to low. '
                        f'You can only answer according to the following content:\n```\n{text}\n```\n'
                        f'You need to carefully consider your answer to ensure that it is based on the context. '
                        f'If the context does not mention the content or it is uncertain whether it is correct, '
                        f'please answer "Current context cannot provide effective information."'
                        f'You must use {self._language} to respond.'},
            {'role': 'user', 'content': query},
        ])
        return result

    def trim_texts(self, context):
        maximum = 4096 - 1024
        for index, text in enumerate(context):
            maximum -= self._num_of_tokens_in_string(text)
            if maximum < 0:
                context = context[:index + 1]
                print(f"Exceeded maximum length, cut the first {index + 1} fragments")
                break
        return context

    def get_keywords(self, query: str) -> str:
        result = self._chat_stream([
            {'role': 'user',
             'content': f'You need to extract keywords from the statement or question and '
                        f'return a series of keywords separated by commas.\ncontent: {query}\nkeywords: '},
        ], use_stream=False)
        return result

    @staticmethod
    def create_embedding(text: str) -> (str, list[float]):
        embedding = openai.Embedding.create(model="text-embedding-ada-002", input=text)
        return text, embedding.data[0].embedding

    def create_embeddings(self, texts: list[str]) -> (list[tuple[str, list[float]]], int):
        result = []
        query_len = 0
        start_index = 0
        tokens = 0

        def get_embedding(input_slice: list[str]):
            embedding = openai.Embedding.create(model="text-embedding-ada-002", input=input_slice)
            return [(txt, data.embedding) for txt, data in
                    zip(input_slice, embedding.data)], embedding.usage.total_tokens

        for index, text in enumerate(texts):
            query_len += self._num_of_tokens_in_string(text)
            if query_len > 8192 - 1024:
                ebd, tk = get_embedding(texts[start_index:index + 1])
                print(f"Query fragments used tokens: {tk}, cost: ${tk / 1000 * 0.0004}")
                query_len = 0
                start_index = index + 1
                tokens += tk
                result.extend(ebd)

        if query_len > 0:
            ebd, tk = get_embedding(texts[start_index:])
            print(f"Query fragments used tokens: {tk}, cost: ${tk / 1000 * 0.0004}")
            tokens += tk
            result.extend(ebd)
        return result, tokens

    def generate_summary(self, embeddings, num_candidates=3, use_sif=False):
        avg_func = self._calc_paragraph_avg_embedding_with_sif if use_sif else self._calc_avg_embedding
        avg_embedding = np.array(avg_func(embeddings))

        paragraphs = [e[0] for e in embeddings]
        embeddings = np.array([e[1] for e in embeddings])
        similarity_scores = cosine_similarity(embeddings, avg_embedding.reshape(1, -1)).flatten()

        candidate_indices = np.argsort(similarity_scores)[::-1][:num_candidates]
        candidate_paragraphs = [f"paragraph {i}: {paragraphs[i]}" for i in candidate_indices]

        print("Calculation completed, start generating summary")

        candidate_paragraphs = self.trim_texts(candidate_paragraphs)

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(candidate_paragraphs))
        result = self._chat_stream([
            {'role': 'system',
             'content': f'As a helpful AI article assistant, '
                        f'I have retrieved the following relevant text fragments from the article, '
                        f'sorted by relevance from high to low. '
                        f'You need to summarize the entire article from these fragments, '
                        f'and present the final result in {self._language}:\n\n{text}\n\n{self._language} summary:'},
        ])
        return result

    @staticmethod
    def _calc_avg_embedding(embeddings) -> list[float]:
        avg_embedding = np.zeros(len(embeddings[0][1]))
        for emb in embeddings:
            avg_embedding += np.array(emb[1])
        avg_embedding /= len(embeddings)
        return avg_embedding.tolist()

    @staticmethod
    def _calc_paragraph_avg_embedding_with_sif(paragraph_list) -> list[float]:
        alpha = 0.001
        n_sentences = len(paragraph_list)
        n_dims = len(paragraph_list[0][1])

        vectorizer = TfidfVectorizer(use_idf=True)
        vectorizer.fit_transform([paragraph for paragraph, _ in paragraph_list])
        idf = vectorizer.idf_

        weights = np.zeros((n_sentences, n_dims))
        for i, (sentence, embedding) in enumerate(paragraph_list):
            sentence_words = sentence.split()
            for word in sentence_words:
                try:
                    word_index = vectorizer.vocabulary_[word]
                    word_idf = idf[word_index]
                    word_weight = alpha / (alpha + word_idf)
                    weights[i] += word_weight * (np.array(embedding) / np.max(embedding))
                except KeyError:
                    pass

        weights_sum = np.sum(weights, axis=0)
        weights_sum /= n_sentences
        avg_embedding = np.zeros(n_dims)
        for i, (sentence, embedding) in enumerate(paragraph_list):
            avg_embedding += (np.array(embedding) / np.max(embedding)) - weights[i]
        avg_embedding /= n_sentences

        return avg_embedding.tolist()
