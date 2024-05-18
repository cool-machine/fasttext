# from gensim.models import FastText
import fasttext
import os
import torch


print(os.getcwd())
fasttext_model_path = "../../finetuned/embedding_fasttext/fasttext_model.bin"  # Replace with the correct path

try:
    model = fasttext.load_model(fasttext_model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
