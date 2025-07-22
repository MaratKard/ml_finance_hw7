from contextlib import asynccontextmanager
import logging
import os

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import prometheus_client
from prometheus_client import CONTENT_TYPE_LATEST
from transformers import AutoTokenizer, AutoModel
import torch
import typer
import uvicorn

from middlewares import GetMetrics

logger = logging.getLogger('uvicorn')

app = FastAPI()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def generate_embeddings(
    model: AutoModel, 
    tokenizer: AutoTokenizer, 
    input: str | list[str]
    ) -> list[list[float]]:
    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')

    with torch.inference_mode():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(
        model_output, 
        encoded_input['attention_mask']
        )
    return sentence_embeddings.cpu().numpy().tolist()

def main() -> None:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info('Loading models...')
        app.tokenizer = AutoTokenizer.from_pretrained(
            os.environ['MODEL_PATH'],
            local_files_only=True
            )
        app.model = AutoModel.from_pretrained(
            os.environ['MODEL_PATH'],
             local_files_only=True
             )
        logger.info('Models loaded')
        yield

    app = FastAPI(
            title='Text embedder',
            version='1.0.0',
            lifespan=lifespan
            )
    app.add_middleware(GetMetrics)

    @app.post("/embeds")
    async def get_embeddings(request: Request) -> dict[str, list[list[float]]]:
        data = await request.json()

        logger.info(f'Get request with data: {data["text"]}')

        sentence_embeddings = generate_embeddings(
            model=request.app.model, 
            tokenizer=request.app.tokenizer,
             input=data['text']
             )

        return {"embedding": sentence_embeddings}

    @app.get('/metrics')
    async def get_metrics(request: Request) -> Response:
        resp = Response(
            content=prometheus_client.generate_latest(), 
            media_type=CONTENT_TYPE_LATEST
            )
        return resp

    logger.info('Starting server...')
    
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ['PORT']))
    logger.info('Server stopped')


if __name__ == "__main__":
    os.environ["PORT"] = "7777"
    os.environ["MODEL_PATH"] = "./models"
    typer.run(main)