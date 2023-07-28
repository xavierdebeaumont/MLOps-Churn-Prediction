from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import joblib
import yaml

params_path = "../../params.yaml"

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def predict(data):
    config = read_params(params_path)
    model_dir_path = config["model_webapp_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data).tolist()[0]
    return prediction 

def load_and_processed_data(item):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    df['total_net_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
    df['total_net_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']
    df['total_net_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']


    df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0}) 
    df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0}) 

    df.drop(columns=['total_day_charge', 'total_eve_charge','total_night_charge',
                    'total_day_calls','total_eve_calls', 'total_night_calls', 'total_day_minutes', 
                    'total_eve_minutes', 'total_night_minutes'], inplace=True)
    return df

app = FastAPI()

class ScoringItem(BaseModel):
    international_plan: str
    voice_mail_plan: str
    number_vmail_messages: int
    total_day_minutes: float
    total_day_calls: int
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: int
    total_eve_charge: float
    total_night_minutes: float
    total_night_calls: int
    total_night_charge: float
    total_intl_minutes: float
    total_intl_calls: int
    total_intl_charge: float
    number_customer_service_calls: int

@app.post('/predict')
async def scoring_endpoint(item:ScoringItem):
    df = load_and_processed_data(item)
    yhat = predict(df)
    return {"prediction": int(yhat)}

################TEST###################
# { 
#  "international_plan": "no", 
#  "voice_mail_plan": "yes", 
#  "number_vmail_messages": "26", 
#  "total_day_minutes": "161.6", 
#  "total_day_calls": "123", 
#  "total_day_charge": "27.47", 
#  "total_eve_minutes": "195.5", 
#  "total_eve_calls": "103", 
#  "total_eve_charge": "16.62", 
#  "total_night_minutes": "254.4", 
#  "total_night_calls": "103", 
#  "total_night_charge": "11.45", 
#  "total_intl_minutes": "13.7", 
#  "total_intl_calls": "3", 
#  "total_intl_charge": "3.7", 
#  "number_customer_service_calls": "1"
#  }