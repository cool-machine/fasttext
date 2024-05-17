from fastapi import APIRouter
from app.schemas.input import TextInput
from app.models.sentiment_model import load_classifier, predict_sentiment
from app.preprocessing.clean_text import clean_text
from transformers import BertTokenizer
from app.preprocessing.clean_text import clean_text

router = APIRouter()

# Load the trained sentiment analysis model and tokenizer
model = load_classifier('models/predict_bert/bert_classifier_best.pth')
tokenizer = BertTokenizer.from_pretrained('models/embedding_bert/tokenizer')

@router.post("/predict/")
async def analyze_sentiment(input: TextInput):
    preprocessed_text = clean_text(input.text)
    sentiment = predict_sentiment(model, tokenizer, preprocessed_text)
    return sentiment
