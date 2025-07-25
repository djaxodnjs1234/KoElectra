from fastapi import FastAPI
from pydantic import BaseModel
from model import predict

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.get("/hello")
def say_hello():
    return {"message": "Hello, GET!"}

@app.get("/good")
def goo_haha():
    return {"good": "kaka"}

@app.post("/predict")
def predict_text(input_data: TextInput):
    prediction = predict(input_data.text)
    return {"text": input_data.text, "emotion": prediction}