import joblib
from fastapi import FastAPI, HTTPException
from app.schemas import SentimentRequest, SentimentResponse
from app.utils import clean_text
try:
    model = joblib.load("model/sentiment_pipeline.pkl")
    label_encoder = joblib.load("model/label_encoder.pkl")
except Exception as e:
    model = None
    label_encoder = None
    load_error = str(e)

app = FastAPI(
    title="Sentiment Analysis API",
    description="API for classifying text as Positive or Negative",
    version="1.0.0"
)


@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is running"}


@app.get("/health")
def health():
    if model is None or label_encoder is None:
        return {"status": "error", "message": load_error}
    return {"status": "ok"}


@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly.")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    cleaned = clean_text(request.text)
    pred_encoded = model.predict([cleaned])[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    return SentimentResponse(
        input_text=request.text,
        cleaned_text=cleaned,
        prediction=pred_label
    )