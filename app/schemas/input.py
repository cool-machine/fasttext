from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str = Field(..., example="This is a sample text.")