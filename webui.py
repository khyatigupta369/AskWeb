import gradio as gr
import xxhash
from gradio.components import _Keywords

from ai import AI
from config import Config
from contents import *
from storage import Storage

def webui(cfg: Config):
    """Run the web UI."""
    Webui(cfg).run()

class Webui:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ai = AI(cfg)

    def _save_to_storage(self, contents, hash_id):
        print(f"Saving to storage {hash_id}")
        print(f"Contents: \n{contents}")
        self.storage = Storage.create_storage(self.cfg)
        if self.storage.been_indexed(hash_id):
            return 0
        else:
            embeddings, tokens = self.ai.create_embeddings(contents)
            self.storage.add_all(embeddings, hash_id)
            return tokens

    def _get_hash_id(self, contents):
        return xxhash.xxh3_128_hexdigest('\n'.join(contents))

    def run(self):
        iface = gr.Interface(fn=self.respond, inputs=["text", "text"], outputs="text")

        # Customize the UI appearance
        iface.launch(share=True, live=True)

    def respond(self, query, context):
        hash_id = self._get_hash_id(context.split('\n'))
        tokens = self._save_to_storage(context.split('\n'), hash_id)
        response = self.ai.completion(query, context)
        return response

if __name__ == "__main__":
    config = Config()  # You may need to initialize your Config object here
    webui(config)
