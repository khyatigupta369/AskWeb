from config import Config
import tiktoken #tokensizer
import openai #Python SDK

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AI:
    '''
    tokenizing, creating embeddings, generating the summary
    '''

    def __init__(self, config):
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
        # calling openai 
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
                    print(data, end = '')
            print() 
            return data.strip()
        else:
            chunk = response.choice[0].message.content.strip()
            print(chunk)
            print(f"Total tokens used: {response.usage.total_tokens}, "
f"cost: ${response.usage.total_tokens / 1000 * 0.002}")
            
            return chunk

        
    def _num_of_tokens_in_string(self, string) -> int:
        '''
        Returns the number of tokens
        '''
        return len(self._encoding.encode(string))

    def completion(self, query: str, context: list[str]):
        context = self.trim_texts(context)
        print(f"Number of query fragments:{len(context)}")

        text = "\n".join(f"{index}. {text}" for index, text in enumerate(context))

    # def trim_texts(self, context):
        
    # def get_keywords(self, query: str) -> str:

    # @staticmethod
    # def create_embedding(text: str) -> (str, list[float]):

    # def create_embedding(self) -> (list[tuple[str, list[float]]], int):

    #     def get_embedding():

    # def generate_summary(self):
    #     '''
    #     Generates a summary
    #     '''

    # # Benchmarking and contexting

    # @staticmethod
    # def _calc_Avg_embedding() -> list[float]:

    # @staticmethod
    # def _calc_paragraph_avg_embedding() -> list[float]:
