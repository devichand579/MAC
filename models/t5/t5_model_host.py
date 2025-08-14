import sys
sys.path.append(".")
from models.t5.t5_model import T5Model
from fastapi import FastAPI, HTTPException
import torch
from pydantic import BaseModel



# --- Pydantic model for request body ---
class PredictionRequest(BaseModel):
    """
    Defines the expected structure of the request body for the /predict endpoint.
    """
    model: str = "t5_coasst"
    text: str

# --- Pydantic model for response body ---
class PredictionResponse(BaseModel):
    """
    Defines the structure of the response from the /predict endpoint.
    """
    input_text: str
    predicted_text: str

# --- FastAPI app instance ---
app = FastAPI(
    title="T5 Model Prediction API",
    description="An API to serve a fine-tuned T5 model for text generation.",
    version="1.0.0",
)

try:
    model_t5_coasst = T5Model(
        model_name="t5-base",
        context=True,
        ckpt="ckpt/t5/t5-base-boozy-cerulean-shark-epoch_39.pth",
        device= "cuda" if torch.cuda.is_available() else "cpu"
    )
except Exception as e:
    print(f"Error loading the T5Model: {e}")
    model = None


# --- API endpoint definition ---
@app.post("/predict", response_model=PredictionResponse)
async def predict_text(request: PredictionRequest):
    """
    Predicts the next part of a sentence using the T5 model.

    - **request**: The request body containing the input text.
    - **Returns**: A JSON object with the input text and the model's prediction.
    """
    if request.model != "t5_coasst":
        raise HTTPException(
            status_code=400,
            detail="Invalid model specified. Only 't5_coasst' is supported."
        )
    if not model_t5_coasst:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check server logs for details."
        )

    # Use a try-except block to gracefully handle potential errors during prediction
    try:
        predicted_text = model_t5_coasst.predict(request.text)
        return PredictionResponse(
            input_text=request.text,
            predicted_text=predicted_text
        )
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred during prediction: {str(e)}"
        )

# --- Instructions for running the server ---
# To run this server, you need to have `fastapi` and `uvicorn` installed.
# You can install them using pip:
# pip install fastapi "uvicorn[standard]"
#
# Then, run the server from your terminal:
# uvicorn t5_model_host:app --reload
#
# The server will be available at http://127.0.0.1:8000
# The interactive API documentation will be at http://127.0.0.1:8000/docs
