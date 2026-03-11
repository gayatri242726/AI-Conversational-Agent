import joblib

model = joblib.load("models/intent_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_intent(text):
    X = vectorizer.transform([text])
    return model.predict(X)[0]
