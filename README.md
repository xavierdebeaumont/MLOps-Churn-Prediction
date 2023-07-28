churn_mlops
==============================

End to End ML pipeline with MLOps practice implementation to predict churn. DVC is used to create the ml pipeline workflow and mlflow is used for its model registry.
An API serves the data and an interface with streamlit allow user to input his data via a form.

## Built with

- [![DVC](https://img.shields.io/badge/DVC-Data%20Version%20Control-945DD6?logo=dvc)](https://odvc.rg/)
- [![MLFlow](https://img.shields.io/badge/MLFlow-Model%20Registry-0193E1?style=blue&logo=MLFlow)](https://mlflow.org/)
- [![FastAPI](https://img.shields.io/badge/FastAPI-Backend-019486?style=green&logo=fastapi&logoColor=Green)](https://fastapi.tiangolo.com/)
- [![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=blue&logo=streamlit)](https://streamlit.io/)

## Data source
- **state**, string. 2-letter code of the US state of customer residence
- **account_length**, numerical. Number of months the customer has been with the current telco provider
- **area_code**, string="area_code_AAA" where AAA = 3 digit area code.
- **international_plan**, (yes/no). The customer has international plan.
- **voice_mail_plan**, (yes/no). The customer has voice mail plan.
- **number_vmail_messages**, numerical. Number of voice-mail messages.
- **total_day_minutes**, numerical. Total minutes of day calls.
- **total_day_calls**, numerical. Total number of day calls.
- **total_day_charge**, numerical. Total charge of day calls.
- **total_eve_minutes**, numerical. Total minutes of evening calls.
- **total_eve_calls**, numerical. Total number of evening calls.
- **total_eve_charge**, numerical. Total charge of evening calls.
- **total_night_minutes**, numerical. Total minutes of night calls.
- **total_night_calls**, numerical. Total number of night calls.
- **total_night_charge**, numerical. Total charge of night calls.
- **total_intl_minutes**, numerical. Total minutes of international calls.
- **total_intl_calls**, numerical. Total number of international calls.
- **total_intl_charge**, numerical. Total charge of international calls
- **number_customer_service_calls**, numerical. Number of calls to customer service
- **churn**, (yes/no). Customer churn - target variable.

You can find more info about this dataset on https://www.kaggle.com/competitions/customer-churn-prediction-2020/data.

## Setup

Run the mlflow server to capture experiments:
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts --host 0.0.0.0 -p 1234
```

Run the dvc workflows to preprocess the data, train and select the best model:
```bash
dvc repro
```

Run the fastapi backend:
```bash
uvicorn app/fastapi/app:app
```

Run the streamlit frontend:
```bash
streamlit run app/streamlit/app.py
```

Request example (churn : no):
```
{ 
"international_plan": "no", 
voice_mail_plan": "yes", 
"number_vmail_messages": "26", 
"total_day_minutes": "161.6", 
"total_day_calls": "123", 
"total_day_charge": "27.47", 
"total_eve_minutes": "195.5", 
"total_eve_calls": "103", 
"total_eve_charge": "16.62", 
"total_night_minutes": "254.4", 
"total_night_calls": "103", 
"total_night_charge": "11.45", 
"total_intl_minutes": "13.7", 
"total_intl_calls": "3", 
"total_intl_charge": "3.7", 
"number_customer_service_calls": "1"
}
```

State, account_length and area_code were dropped.