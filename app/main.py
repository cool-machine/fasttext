import warnings


from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


import logging
from typing import List
import pandas as pd
import torch

# from app.models.sentiment_model import SentimentModel  # Ensure this is correctly imported
# from app.preprocessing.clean_text import clean_text  # Ensure this is correctly imported
from app.api.endpoints import analyze_sentiment


# global variables
input_size = 128  # Assuming FastText embeddings are of size 128
hidden_size = 64
num_layers = 2
num_classes = 2  # Number of output classes
dropout = 0.5


# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# Mounting static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Setting up templates
templates = Jinja2Templates(directory="app/templates")

# Defining the request model
class TextRequest(BaseModel):
    text: str

# # Load models and initialize SentimentModel
# fasttext_model_path = "finetuned/embedding_fasttext/fasttext_model.bin"  # Update the path accordingly
# lstm_model_path = "finetuned/predict_lstm/lstm_fasttext.pth"  # Update the path accordingly

# try:
#     sentiment_model = SentimentModel(fasttext_model_path, lstm_model_path, input_size, hidden_size, num_layers, num_classes, dropout)
# except Exception as e:
#     print(f"Error loading models: {e}")
#     raise

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # Read the uploaded file into a DataFrame with header=None
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file.file, header=None)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file.file, header=None)
        else:
            logger.error(f"Unsupported file type: {file.filename}")
            return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

        # Log the DataFrame to ensure it is read correctly
        logger.info(f"DataFrame head: {df.head()}")

        # Extract the documents (assuming each row is a document)
        documents = df.iloc[:, 0].dropna().tolist()  # Assuming the first column contains the documents
        logger.info(f"Extracted documents: {documents}")

        # Check if documents list is empty
        if not documents:
            logger.error("No documents found in the uploaded file")
            return JSONResponse(content={"error": "No documents found in the uploaded file"}, status_code=400)

        # Limit to first 10 documents
        if len(documents) > 10:
            documents = documents[:10]
            warning = "Only the first ten documents will be taken into account."
        else:
            warning = ""

        return JSONResponse(content={"documents": documents, "warning": warning})

    except pd.errors.EmptyDataError:
        logger.error("Uploaded file is empty")
        return JSONResponse(content={"error": "Uploaded file is empty"}, status_code=400)
    except pd.errors.ParserError:
        logger.error("Error parsing the uploaded file")
        return JSONResponse(content={"error": "Error parsing the uploaded file"}, status_code=400)
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


# Defining the /analyze endpoint
@app.post("/analyze")
async def analyze_text(text_request: TextRequest):
    try:
        text = text_request
        # Log the received text
        logger.info(f"Received text: {text}")

        # Analyze the text (await the async function)
        sentiment = await analyze_sentiment(text)

        # Log the analysis result
        logger.info(f"Sentiment analysis result: {sentiment}")

        return JSONResponse(content={"sentiment": sentiment})
    
    except Exception as e:
        # Log the error details
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# @app.post("/analyze")
# async def analyze_text(request: TextRequest):
#     try:
#         text = request

#         # Log the received text
#         logger.info(f"Received text: {text}")

#         # Analyze the text (await the async function)
#         sentiment = await analyze_sentiment(text)

#         # Log the analysis result
#         logger.info(f"Sentiment analysis result: {sentiment}")

#         return JSONResponse(content={"sentiment": sentiment})

#     cleaned_text = clean_text(text)
#     words = cleaned_text.split()  # Split the cleaned text into words
#     sentiment = sentiment_model.predict(words)
#     return {"sentiment": sentiment}




# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     content = await file.read()
#     df = pd.read_csv(file.file)
#     documents = df.iloc[:, 0].tolist()  # Assuming the first column contains the documents
#     documents = documents[:10]  # Limit to first 10 documents
#     return {"documents": documents}

# @app.post("/analyze_document")
# async def analyze_document(text: str = Form(...)):
#     cleaned_text = clean_text(text)
#     words = cleaned_text.split()
#     sentiment = sentiment_model.predict(words)
#     return {"sentiment": sentiment}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



# import warnings

# # Suppress the specific warning from huggingface_hub
# warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


# import warnings

# # Suppress the specific warning from huggingface_hub
# warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

# from fastapi import FastAPI, UploadFile, File, Form
# from pydantic import BaseModel
# from typing import List
# import pandas as pd
# import torch
# from app.models.sentiment_model import SentimentModel
# from app.preprocessing.clean_text import clean_text

# app = FastAPI()

# # Load models and initialize SentimentModel
# fasttext_model_path = "finetuned/embedding_fasttext/fasttext_model.bin"  # Adjust the path as needed
# lstm_model_path = "finetuned/predict_lstm/lstm_fasttext.pth"
# input_size = 128  # Assuming FastText embeddings are of size 128
# hidden_size = 64
# num_layers = 2
# num_classes = 2  # Number of output classes
# dropout = 0.5

# sentiment_model = SentimentModel(fasttext_model_path, lstm_model_path, input_size, hidden_size, num_layers, num_classes, dropout)

# class TextRequest(BaseModel):
#     text: str

# @app.post("/analyze")
# async def analyze_text(request: TextRequest):
#     text = request.text
#     cleaned_text = clean_text(text)
#     words = cleaned_text.split()  # Split the cleaned text into words
#     sentiment = sentiment_model.predict(words)
#     return {"sentiment": sentiment}

# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     content = await file.read()
#     df = pd.read_csv(file.file)
#     documents = df.iloc[:, 0].tolist()  # Assuming the first column contains the documents
#     documents = documents[:10]  # Limit to first 10 documents
#     return {"documents": documents}

# @app.post("/analyze_document")
# async def analyze_document(text: str = Form(...)):
#     cleaned_text = clean_text(text)
#     words = cleaned_text.split()
#     sentiment = sentiment_model.predict(words)
#     return {"sentiment": sentiment}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# print(f"main.py file's end reached")
# # The rest of your imports
# from fastapi import FastAPI, Request, File, UploadFile
# from fastapi.responses import JSONResponse, HTMLResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# # import pandas as pd
# import logging
# from typing import List
# from app.api.endpoints import analyze_sentiment  # Importing the function

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# app = FastAPI()

# # Mounting static files
# app.mount("/static", StaticFiles(directory="app/static"), name="static")

# # Setting up templates
# templates = Jinja2Templates(directory="app/templates")

# # Defining the request model
# class TextRequest(BaseModel):
#     text: str

# # Root endpoint serving the index.html
# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# # File upload endpoint
# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         # Read the uploaded file into a DataFrame with header=None
#         if file.filename.endswith('.csv'):
#             df = pd.read_csv(file.file, header=None)
#         elif file.filename.endswith('.xlsx'):
#             df = pd.read_excel(file.file, header=None)
#         else:
#             logger.error(f"Unsupported file type: {file.filename}")
#             return JSONResponse(content={"error": "Unsupported file type"}, status_code=400)

#         # Log the DataFrame to ensure it is read correctly
#         logger.info(f"DataFrame head: {df.head()}")

#         # Extract the documents (assuming each row is a document)
#         documents = df.iloc[:, 0].dropna().tolist()  # Assuming the first column contains the documents
#         logger.info(f"Extracted documents: {documents}")

#         # Check if documents list is empty
#         if not documents:
#             logger.error("No documents found in the uploaded file")
#             return JSONResponse(content={"error": "No documents found in the uploaded file"}, status_code=400)

#         # Limit to first 10 documents
#         if len(documents) > 10:
#             documents = documents[:10]
#             warning = "Only the first ten documents will be taken into account."
#         else:
#             warning = ""

#         return JSONResponse(content={"documents": documents, "warning": warning})
#     except pd.errors.EmptyDataError:
#         logger.error("Uploaded file is empty")
#         return JSONResponse(content={"error": "Uploaded file is empty"}, status_code=400)
#     except pd.errors.ParserError:
#         logger.error("Error parsing the uploaded file")
#         return JSONResponse(content={"error": "Error parsing the uploaded file"}, status_code=400)
#     except Exception as e:
#         logger.error(f"Error uploading file: {str(e)}", exc_info=True)
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# # Defining the /analyze endpoint
# @app.post("/analyze")
# async def analyze_text(text_request: TextRequest):
#     try:
#         text = text_request
#         # Log the received text
#         logger.info(f"Received text: {text}")

#         # Analyze the text (await the async function)
#         sentiment = await analyze_sentiment(text)

#         # Log the analysis result
#         logger.info(f"Sentiment analysis result: {sentiment}")

#         return JSONResponse(content={"sentiment": sentiment})
#     except Exception as e:
#         # Log the error details
#         logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
