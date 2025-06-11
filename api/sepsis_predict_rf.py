
# A simple model to predict onset of Sepsis

import os
import sys
import json
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

SEPSIS_FV_LEN = 28

sepsis_fv_index= {'Age': 0,
  'Gender': 1,
  'HeartRate': 2,
  'Temp': 3,
  'SystolicBP': 4,
  'MeanBP': 5,
  'DiastolicBP': 6,
  'RespRate': 7,
  'OximetrySat': 8,
  'Potassium': 9,
  'Chloride': 10,
  'Calcium': 11,
  'Hemoglobin': 12,
  'pH': 13,
  'BaseExcess': 14,
  'Bicarbonate': 15,
  'FiO2': 16,
  'Glucose': 17,
  'BUN': 18,
  'Creatinine': 19,
  'Magnesium': 20,
  'SGOT': 21,
  'SGPT': 22,
  'TotalBili': 23,
  'WBC': 24,
  'Platelets': 25,
  'PaCO2': 26,
  'Lactate': 27
  }

def get_test_json(fname):
    jtest = dict()
    try:
        df = pd.read_csv(fname)
        df = df.drop('id',axis=1)
        label = df['SepsisLabel'].copy().to_numpy()
        df = df.drop('SepsisLabel',axis=1)
        jtest = df.to_dict(orient='records')
    except Exception as e:
        logging.error(f"get_test_json():Exception:{e}")
    finally:
        return jtest,label

def rf_predict(jarr):
    model_fname = "/code/api/models/model_rf.pkl"
    impute_pl_fname = "/code/api/models/imp_pl_rf.pkl"
    prediction = np.full((len(jarr),), -1)
    try:
        #num_pipeline = Pipeline([ ('imputer', SimpleImputer(strategy="median")),
        #                          ('std_scaler',StandardScaler())
        #                        ])
        rf_model = joblib.load(model_fname)
        rf_imp_pl = joblib.load(impute_pl_fname)
        num_samples = len(jarr)
        num_features = len(sepsis_fv_index)
        sepsis_fv = np.empty((num_samples,num_features))
        for i in range(num_samples):
            for key in jarr[i]:
                if key in sepsis_fv_index:
                    fv_idx = sepsis_fv_index[key]
                    sepsis_fv[i][fv_idx]=jarr[i][key]
        sepsis_fv_imputed = rf_imp_pl.transform(sepsis_fv)
        prediction = rf_model.predict(sepsis_fv_imputed)

    except Exception as e:
        logging.error(f"sepsis_predict(): Exception :{e}")
    finally:
        return prediction

if __name__ == '__main__':
    try:
        logging.getLogger().setLevel(logging.INFO)
        #home_dir = os.getenv("HOME")
        #load the model to from file
        model_fname = "models/dl_rf_2024_09_23_11_52_52.pkl"
        model_fname = "models/model_rf.pkl"
        impute_pl_fname = "models/dl_rf_imp_pl2024_09_23_11_52_52.pkl"
        impute_pl_fname = "models/imp_pl_rf.pkl"
        test_fname = "data/test.csv"
        #rf_model = pickle.load(open(model_fname,'rb'))
        rf_model = joblib.load(model_fname)
        #rf_imp_pl = pickle.load(open(impute_pl_fname,'rb'))
        rf_imp_pl = joblib.load(impute_pl_fname)
        logging.info(f"loaded the model from {model_fname}")
        jarr,label = get_test_json(test_fname)
        logging.info(f"created the test json data from {test_fname}")
        prediction = rf_predict(jarr)
        print(prediction)
        print(label)
    except Exception as e:
        logging.error(f"main(): Exception :{e}")
