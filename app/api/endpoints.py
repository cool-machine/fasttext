from fastapi import APIRouter
from app.schemas.input import TextInput
from app.models.sentiment_model import sentiment_pipeline
from app.preprocessing.clean_text import clean_text

import fasttext

router = APIRouter()

# # Load the trained sentiment analysis model and tokenizer
# model_path = '/finetuned/predict_lstm/lstm_fasttext.pth'
# tokenizer_path = '/finetuned/embedding_fasttext/fasttext_model.bin'

@router.post("/predict/")
async def analyze_sentiment(input: TextInput):
    preprocessed_text = clean_text(input.text)
    sentiment = sentiment_pipeline(preprocessed_text)
    return sentiment