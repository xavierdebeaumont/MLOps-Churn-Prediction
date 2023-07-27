import yaml
import argparse
import numpy as np 
import pandas as pd 
from make_raw_data import read_params

def total_net_minutes(df, interim_data_path):
    df['total_net_minutes'] = df['total_day_minutes'] + df['total_eve_minutes'] + df['total_night_minutes']
    df['total_net_calls'] = df['total_day_calls'] + df['total_eve_calls'] + df['total_night_calls']
    df['total_net_charge'] = df['total_day_charge'] + df['total_eve_charge'] + df['total_night_charge']


    df['voice_mail_plan'] = df['voice_mail_plan'].map({'yes': 1, 'no': 0}) 
    df['international_plan'] = df['international_plan'].map({'yes': 1, 'no': 0}) 
    df['churn'] = df['churn'].map({'yes': 1, 'no': 0}) 

    df.drop(columns=['total_day_charge', 'total_eve_charge','total_night_charge',
                    'total_day_calls','total_eve_calls', 'total_night_calls', 'total_day_minutes', 
                    'total_eve_minutes', 'total_night_minutes'], inplace=True)
    df.to_csv(interim_data_path, sep=",", index=False, encoding="utf-8")  
    
def total_net_minutes_saved(config_path):
    config = read_params(config_path)
    interim_data_path = config["interim_data_config"]["interim_data_csv"]
    raw_data_path = config["raw_data_config"]["raw_data_csv"]
    raw_df=pd.read_csv(raw_data_path)
    total_net_minutes(raw_df, interim_data_path)

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    total_net_minutes_saved(config_path=parsed_args.config)