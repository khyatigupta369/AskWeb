import xxhash
import os
from ai import AI
from config import Config
from storage import Storage
from contents import *


def console(config: Config):
    try:
        while True:
            if not run_console(config):
                return
    except KeyboardInterrupt:
        print("exit")


def run_console(config: Config) -> bool:
    contents, lang, identify = get_contents()

    print("The article has been retrieved, and the number of text fragments is:", len(contents))
    for content in contents:
        print('\t', content)

    ai = AI(config)
    storage = Storage.create_storage(config)

    print("=====================================")
    if storage.been_indexed(identify):
        print("The article has already been indexed, so there is no need to index it again.")
        print("=====================================")
    else:
        embeddings, tokens = ai.create_embeddings(contents)
        print(f"Embeddings have been created with {len(embeddings)} embeddings, using {tokens} tokens, "
              f"costing ${tokens / 1000 * 0.0004}")

        storage.add_all(embeddings, identify)
        print("The embeddings have been saved.")
        print("=====================================")

    while True:
        query = input("Please enter your query (/help to view commands):").strip()
        if query.startswith("/"):
            if query == "/quit":
                return False
            elif query == "/reset":
                print("=====================================")
                return True
            elif query == "/summary":
                ai.generate_summary(storage.get_all_embeddings(identify), num_candidates=100,
                                     use_sif=lang not in ['zh', 'ja', 'ko', 'hi', 'ar', 'fa'])
            elif query == "/reindex":
                storage.clear(identify)
                embeddings, tokens = ai.create_embeddings(contents)
                print(f"Embeddings have been created with {len(embeddings)} embeddings, using {tokens} tokens, "
                      f"costing ${tokens / 1000 * 0.0004}")

                storage.add_all(embeddings, identify)
                print("The embeddings have been saved.")
            elif query == "/help":
                print("Enter /summary to generate an embedding-based summary.")
                print("Enter /reindex to re-index the article.")
                print("Enter /reset to start over.")
                print("Enter /quit to exit.")
                print("Enter any other content for a query.")
            else:
                print("Invalid command.")
                print("Enter /summary to generate an embedding-based summary.")
                print("Enter /reindex to re-index the article.")
                print("Enter /reset to start over.")
                print("Enter /quit to exit.")
                print("Enter any other content for a query.")
            print("=====================================")
            continue
        else:
            print("Generate keywords.")
            keywords = ai.get_keywords(query)
            _, embedding = ai.create_embedding(keywords)
            texts = storage.get_texts(embedding, identify)
            print("Related fragments found (first 5):")
            for text in texts[:5]:
                print('\t', text)
            ai.completion(query, texts)
            print("=====================================")


def get_contents() -> tuple[list[str], str, str]:
    while True:
        try:
            url = input("Please enter the link to the article or the file path of the PDF/TXT/DOCX document: ").strip()
            if os.path.exists(url):
                if url.endswith('.pdf'):
                    contents, data = extract_text_from_pdf(url)
                elif url.endswith('.txt'):
                    contents, data = extract_text_from_txt(url)
                elif url.endswith('.docx'):
                    contents, data = extract_text_from_docx(url)
                else:
                    print("Unsupported file format.")
                    continue
            else:
                contents, data = web_crawler_newspaper(url)
            if not contents:
                print("Unable to retrieve the content of the article. Please enter the link to the article or "
                      "the file path of the PDF/TXT/DOCX document again.")
                continue
            return contents, data, xxhash.xxh3_128_hexdigest('\n'.join(contents))
        except Exception as e:
            print("Error:", e)

