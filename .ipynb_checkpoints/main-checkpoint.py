from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import logging
from sklearn.preprocessing import StandardScaler


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
binary_classification = tf.keras.models.load_model("models/logistic_regression_model.keras")
linear_model = tf.keras.models.load_model("models/linear_regression_model.keras")

class BinaryInput(BaseModel):
    features: list[list[int]]

class LinearInput(BaseModel):
    age: int
    income: float
    years_of_experience: int

@app.get("/")
def read_root():
    return {"message": "ML Model API is running"}

@app.post("/binary/predict")
def predict(data: BinaryInput):
    input_array = np.array(data.features).reshape(1, -1) / 255.0
    prediction = binary_classification.predict(input_array)
    return {"prediction": float(prediction[0][0])}

@app.post("/linear/predict")
def predict_linear(data: LinearInput):
    
    try:
        logger.info(f"Received input: {data}")
        input = np.array([
            data.age,
            data.income,
            data.years_of_experience
        ]).reshape(1, -1)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(input)
        prediction = linear_model.predict(scaled)

        logger.info(f"Model prediction: {prediction.tolist()}")
        return {"prediction": float(prediction[0])}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))