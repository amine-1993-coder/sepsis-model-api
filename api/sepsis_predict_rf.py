import numpy as np
import joblib
import logging

# Global mapping of features to indices (must match training)
sepsis_fv_index = {
    'Age': 0, 'Gender': 1, 'HeartRate': 2, 'Temp': 3, 'SystolicBP': 4,
    'MeanBP': 5, 'DiastolicBP': 6, 'RespRate': 7, 'OximetrySat': 8,
    'Potassium': 9, 'Chloride': 10, 'Calcium': 11, 'Hemoglobin': 12,
    'pH': 13, 'BaseExcess': 14, 'Bicarbonate': 15, 'FiO2': 16,
    'Glucose': 17, 'BUN': 18, 'Creatinine': 19, 'Magnesium': 20,
    'SGOT': 21, 'SGPT': 22, 'TotalBili': 23, 'WBC': 24,
    'Platelets': 25, 'PaCO2': 26, 'Lactate': 27
}

# Prediction handler for Connexion
def predict(body):
    logging.getLogger().setLevel(logging.INFO)

    # Adjust path if needed (for Docker context)
    model_path = "api/models/model_rf.pkl"
    imputer_path = "api/models/imp_pl_rf.pkl"

    try:
        rf_model = joblib.load(model_path)
        rf_imp_pl = joblib.load(imputer_path)
        logging.info("Loaded model and imputer")

        jarr = body if isinstance(body, list) else [body]  # Accepts single or list of dicts
        num_samples = len(jarr)
        num_features = len(sepsis_fv_index)

        sepsis_fv = np.empty((num_samples, num_features))
        sepsis_fv[:] = np.nan  # fill with NaN to allow imputation

        for i in range(num_samples):
            for key, val in jarr[i].items():
                if key in sepsis_fv_index:
                    sepsis_fv[i][sepsis_fv_index[key]] = val

        sepsis_fv_imputed = rf_imp_pl.transform(sepsis_fv)
        preds = rf_model.predict(sepsis_fv_imputed)

        # Return list of prediction objects
        return [{"prediction": str(p)} for p in preds]

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}
