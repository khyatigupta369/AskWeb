import os
import shutil

import uvicorn
import xxhash
from fastapi import FastAPI, UploadFile, File
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import JSONResponse

from ai import AI
from config import Config
from contents import web_crawler_newspaper, extract_text_from_txt, extract_text_from_docx, extract_text_from_pdf
from storage import Storage


def run_api(config: Config):
    """Run the API."""
    config.use_stream = False
    ai_instance = AI(config)

    app = FastAPI()

    class CrawlerUrlRequest(BaseModel):
        url: str

    @app.post("/crawler_url")
    async def crawl_url(request: CrawlerUrlRequest):
        """Crawl the URL."""
        contents, language = web_crawler_newspaper(request.url)
        hash_id = xxhash.xxh3_128_hexdigest('\n'.join(contents))
        tokens = save_to_storage(contents, hash_id)
        return {"code": 0, "msg": "ok", "data": {"uri": f"{hash_id}/{language}", "tokens": tokens}}

    def save_to_storage(contents, hash_id):
        storage_instance = Storage.create_storage(config)
        if storage_instance.been_indexed(hash_id):
            return 0
        else:
            embeddings, tokens = ai_instance.create_embeddings(contents)
            storage_instance.add_all(embeddings, hash_id)
            return tokens

    @app.post("/upload_file")
    async def upload_file(file: UploadFile = File(...)):
        """Upload file."""
        # Save file to disk
        file_name = file.filename
        os.makedirs('./upload', exist_ok=True)
        upload_path = os.path.join('./upload', file_name)
        with open(upload_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if file_name.endswith('.pdf'):
            contents, language = extract_text_from_pdf(upload_path)
        elif file_name.endswith('.txt'):
            contents, language = extract_text_from_txt(upload_path)
        elif file_name.endswith('.docx'):
            contents, language = extract_text_from_docx(upload_path)
        else:
            return {"code": 1, "msg": "not support", "data": {}}
        hash_id = xxhash.xxh3_128_hexdigest('\n'.join(contents))
        tokens = save_to_storage(contents, hash_id)
        os.remove(upload_path)
        return {"code": 0, "msg": "ok", "data": {"uri": f"{hash_id}/{language}", "tokens": tokens}}

    @app.get("/summary")
    async def generate_summary(uri: str):
        """Generate summary."""
        hash_id, language = uri.split('/')
        storage_instance = Storage.create_storage(config)
        if not storage_instance or not language:
            return {"code": 1, "msg": "not found", "data": {}}
        summary = ai_instance.generate_summary(storage_instance.get_all_embeddings(hash_id), num_candidates=100,
                                               use_sif=language not in ['zh', 'ja', 'ko', 'hi', 'ar', 'fa'])
        return {"code": 0, "msg": "ok", "data": {"summary": summary}}

    class AnswerRequest(BaseModel):
        uri: str
        query: str

    @app.get("/answer")
    async def query_answer(request: AnswerRequest):
        """Query for an answer."""
        hash_id, language = request.uri.split('/')
        storage_instance = Storage.create_storage(config)
        if not storage_instance or not language:
            return {"code": 1, "msg": "not found", "data": {}}
        keywords = ai_instance.get_keywords(request.query)
        _, embedding = ai_instance.create_embedding(keywords)
        texts = storage_instance.get_texts(embedding, hash_id)
        answer = ai_instance.completion(request.query, texts)
        return {"code": 0, "msg": "ok", "data": {"answer": answer}}

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        """Error handler for validation errors."""
        print("Validation Error Handler: ", request.url, exc)
        return JSONResponse(
            status_code=400,
            content={"code": 1, "msg": str(exc.errors()), "data": {}},
        )

    @app.exception_handler(HTTPException)
    async def handle_http_error(request: Request, exc):
        """Error handler for HTTP errors."""
        print("HTTP Error Handler: ", request.url, exc)
        return JSONResponse(
            status_code=400,
            content={"code": 1, "msg": exc.detail, "data": {}},
        )

    # Run the API
    uvicorn.run(app, host=config.api_host, port=config.api_port)
