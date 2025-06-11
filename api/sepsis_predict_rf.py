import numpy as np
import joblib
import logging

# Constants
SEPSIS_FV_LEN = 28
MODEL_PATH = "api/models/model_rf.pkl"
IMPUTER_PATH = "api/models/imp_pl_rf.pkl"

# Feature name to index mapping
sepsis_fv_index = {
    'Age': 0, 'Gender': 1, 'HeartRate': 2, 'Temp': 3, 'SystolicBP': 4,
    'MeanBP': 5, 'DiastolicBP': 6, 'RespRate': 7, 'OximetrySat': 8,
    'Potassium': 9, 'Chloride': 10, 'Calcium': 11, 'Hemoglobin': 12,
    'pH': 13, 'BaseExcess': 14, 'Bicarbonate': 15, 'FiO2': 16,
    'Glucose': 17, 'BUN': 18, 'Creatinine': 19, 'Magnesium': 20,
    'SGOT': 21, 'SGPT': 22, 'TotalBili': 23, 'WBC': 24,
    'Platelets': 25, 'PaCO2': 26, 'Lactate': 27
}

# Main prediction function
def rf_predict(jarr):
    """
    Predict sepsis given a list of feature dicts (jarr).
    Each dict represents one patient and must include some or all of the 28 known features.
    """
    logging.getLogger().setLevel(logging.INFO)
    prediction = np.full((len(jarr),), -1)  # default prediction

    try:
        # Load model and imputer pipeline
        rf_model = joblib.load(MODEL_PATH)
        rf_imp_pl = joblib.load(IMPUTER_PATH)
        logging.info("Loaded model and imputer successfully")

        # Prepare input feature matrix
        num_samples = len(jarr)
        sepsis_fv = np.empty((num_samples, SEPSIS_FV_LEN))
        sepsis_fv[:] = np.nan  # prefill with NaN for imputation

        for i, entry in enumerate(jarr):
            for key, value in entry.items():
                if key in sepsis_fv_index:
                    idx = sepsis_fv_index[key]
                    sepsis_fv[i][idx] = value

        # Impute and predict
        sepsis_fv_imputed = rf_imp_pl.transform(sepsis_fv)
        prediction = rf_model.predict(sepsis_fv_imputed)

    except Exception as e:
        logging.error(f"rf_predict(): Exception occurred: {e}")
    finally:
        return prediction
