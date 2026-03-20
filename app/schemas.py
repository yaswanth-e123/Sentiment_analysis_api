from pydantic import BaseModel


class SentimentRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    input_text: str
    cleaned_text: str
    prediction: str
    