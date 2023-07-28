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

app = FastAPI()

class ScoringItem(BaseModel):
    international_plan: int
    voice_mail_plan: int 
    number_vmail_messages: int
    total_intl_minutes: float 
    total_intl_calls: int 
    total_intl_charge: float 
    number_customer_service_calls: int 
    total_net_minutes: float 
    total_net_calls: int 
    total_net_charge: float

@app.post('/predict')
async def scoring_endpoint(item:ScoringItem):
    df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
    yhat = predict(df)
    return {"prediction": int(yhat)}


