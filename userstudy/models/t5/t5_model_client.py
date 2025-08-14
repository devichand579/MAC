import sys
from pydantic import BaseModel


sys.path.append(".")

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





class T5ModelClient:
    """
    A client class to interact with the T5 model prediction API.
    """
    def __init__(self, api_url: str):
        """
        Initializes the T5ModelClient with the API URL.

        Args:
            api_url (str): The base URL of the T5 model prediction API.
        """
        self.api_url = api_url

    def predict(self, text: str) -> PredictionResponse:
        """
        Sends a prediction request to the T5 model API.

        Args:
            text (str): The input text for which to generate a prediction.

        Returns:
            PredictionResponse: The response from the API containing the predicted text.
        """
        import requests

        response = requests.post(
            f"{self.api_url}/predict",
            json={"model": "t5_coasst", "text": text}
        )

        if response.status_code != 200:
            raise Exception(f"Error in prediction: {response.text}")

        return PredictionResponse(**response.json())