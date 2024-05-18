from fastapi import APIRouter
from app.schemas.input import TextInput
from app.models.sentiment_model import load_classifier, predict_sentiment
from app.preprocessing.clean_text import clean_text
from app.preprocessing.clean_text import clean_text
import fasttext

router = APIRouter()

# Load the trained sentiment analysis model and tokenizer
model = load_classifier('/finetuned/predict_lstm/bert_classifier_best.pth')
tokenizer = fasttext.load_model('/finetuned/embedding_fasttext/fasttext_model.bin')

@router.post("/predict/")
async def analyze_sentiment(input: TextInput):
    preprocessed_text = clean_text(input.text)
    sentiment = predict_sentiment(model, tokenizer, preprocessed_text)
    return sentiment
