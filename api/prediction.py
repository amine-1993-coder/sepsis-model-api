import sys
import json
import logging
import os
from api.sepsis_predict_rf import rf_predict

# Set logging level
logging.getLogger().setLevel(logging.INFO)

class Prediction:
    def __init__(self):
        # You can preload models here if needed
        pass

    def post(self, body):
        """
        POST /prediction/
        Expects JSON body with 'sepsis_fv' as a list of feature dicts.
        Returns: JSON with 'sepsis_risk': list of 0/1 predictions
        """
        logging.info(f"Received request body: {body}")
        try:
            sepsis_input = body.get("sepsis_fv")
            if not isinstance(sepsis_input, list):
                raise ValueError("Expected 'sepsis_fv' to be a list of feature dicts")

            prediction = rf_predict(sepsis_input)
            logging.info(f"Predictions: {prediction}")

            return {"sepsis_risk": prediction.tolist()}, 200

        except Exception as e:
            logging.exception("Prediction failed")
            return {"error": str(e)}, 500

    def get(self):
        """
        GET /prediction/
        Basic health check endpoint.
        """
        return {"status": "Sepsis prediction API is up"}, 200

# Required by Connexion to map to class_instance.post
class_instance = Prediction()

