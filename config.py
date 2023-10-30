import json
import os

class Config:
    def __init__(self):
        '''
        Read each parameter
            open_ai_key
            temperature
            language
            open_ai_chat_model
            use_stream
            use_postgres
            index_path
            postgres_url
            mode
            api_port
            api_host
            webui_port
        '''

        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        # handle file not found
        if not os.path.exists(config_path):
            raise Exception('config file not found')
        
        # read the file and capture all parameter 
        with open(config_path, 'r') as file:
            self.config = json.load(file)

            self.open_ai_key = self.config.get('open_ai_key')
            if not self.open_ai_key:
                raise Exception('open_ai_key is not set')
            
            self.temperature = self.config.get('temperature', 0.2)
            if self.temperature < 0 or self.temperature > 1:
                raise Exception('temperature must be between 0 and 1')
            
            self.use_postgres = self.config.get('use_postgres', False)
            if not self.use_postgres:
                self.index_path = self.config.get('index_path', './temp')
                os.makedirs(self.index_path, exist_ok=True)

            self.postgres_url = self.config.get('postgres_url')
            if self.use_postgres and self.postgres_url is None:
                raise Exception('postgres_url is not set')

            self.mode = self.config.get('mode', 'webui')
            if self.mode not in ['console', 'api', 'webui']:
                raise Exception('mode must be console or api or webui')
            
            self.language = self.config.get('language', 'English')
            self.open_ai_proxy = self.config.get('open_ai_proxy')
            self.open_ai_chat_model = self.config.get('open_ai_chat_model', 'gpt-3.5-turbo')
            self.use_stream = self.config.get('use_stream', False)
            self.api_port = self.config.get('api_port', 9090)
            self.api_host = self.config.get('api_host', 'localhost')
            self.webui_port = self.config.get('webui_port', 8080)
            self.webui_host = self.config.get('webui_host', '0.0.0.0')