from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Fake News Detection API (XGBoost + 20k TF-IDF)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("models/final_xgb_model.pkl")
tfidf = joblib.load("models/final_tfidf.pkl")

class ArticleRequest(BaseModel):
    title: str = ""
    text: str = ""
    strip_reuters: bool = False

@app.post("/predict")
def predict_news(req: ArticleRequest):
    content = f"{req.title} {req.text}".strip()
    
    if req.strip_reuters:
        content = re.sub(r"reuters", "", content, flags=re.IGNORECASE)
        
    X_vec = tfidf.transform([content])
    
    pred_num = int(model.predict(X_vec)[0])
    probabilities = model.predict_proba(X_vec)[0]
    
    label = "REAL" if pred_num == 1 else "FAKE"
    confidence = float(max(probabilities) * 100)
    
    return {
        "prediction": label,
        "class_id": pred_num,
        "confidence": round(confidence, 2),
        "prob_fake": round(float(probabilities[0]) * 100, 2),
        "prob_real": round(float(probabilities[1]) * 100, 2)
    }