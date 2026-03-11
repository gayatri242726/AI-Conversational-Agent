from fastapi import FastAPI
from pydantic import BaseModel
from services.intent_service import predict_intent
from services.entity_service import extract_entities
from services.sentiment_service import analyze_sentiment
from services.llm_service import generate_response

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/chat")
def chat(msg: Message):
    intent = predict_intent(msg.text)
    entities = extract_entities(msg.text)
    sentiment = analyze_sentiment(msg.text)
    response = generate_response(msg.text)

    return {
        "intent": intent,
        "entities": entities,
        "sentiment": sentiment,
        "response": response
    }
