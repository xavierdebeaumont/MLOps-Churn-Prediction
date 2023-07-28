import json
import yaml
import joblib
import mlflow
import argparse
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix, classification_report

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_feat_and_target(df, target):
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x, y

def accuracymeasures(y_test, predictions, avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def train_model(model, train_x, train_y):
    model.fit(train_x, train_y)
    return model

def evaluate_model(model, test_x, test_y, avg_method):
    y_pred = model.predict(test_x)
    metrics = accuracymeasures(test_y, y_pred, avg_method)

    # Create a dictionary to store the metrics
    metrics_dict = {
        "accuracy": metrics[0],
        "precision": metrics[1],
        "recall": metrics[2],
        "f1score": metrics[3]
    }

    return metrics_dict

def log_mlflow_metrics(metrics_dict):
    for metric_name, metric_value in metrics_dict.items():
        mlflow.log_metric(f"{metric_name}", metric_value)

def log_mlflow_params(params_dict):
    for param_name, param_value in params_dict.items():
        mlflow.log_param(param_name, param_value)

def log_mlflow_model(model_name, model):
    tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

    if tracking_url_type_store != "file":
        mlflow.sklearn.log_model(
            model, 
            artifact_path="models", 
            registered_model_name=model_name
        )
    else:
        mlflow.sklearn.load_model(model, "model")

def train_and_evaluate_model(model_name, model, train_x, train_y, test_x, test_y, params, avg_method):
    model = train_model(model, train_x, train_y)
    metrics = evaluate_model(model, test_x, test_y, avg_method)
    log_mlflow_params(params)
    log_mlflow_metrics(metrics)
    log_mlflow_model(model_name, model)

def main(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["interim_data_config"]["target"]
    avg_method = config["metrics"]["average_method"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x, train_y = get_feat_and_target(train, target)
    test_x, test_y = get_feat_and_target(test, target)

    # MLFlow
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    # Random Forest
    rf_params = {
        "max_depth": config["random_forest"]["max_depth"],
        "n_estimators": config["random_forest"]["n_estimators"]
    }
    with mlflow.start_run(run_name=mlflow_config["run_name"] + "_random_forest") as mlops_run:
        train_and_evaluate_model("random_forest", RandomForestClassifier(), train_x, train_y, test_x, test_y, rf_params, avg_method)

    # Logistic Regression
    logreg_params = {
        "max_depth": config["random_forest"]["max_depth"],
        "n_estimators": config["random_forest"]["n_estimators"]
    }
    with mlflow.start_run(run_name=mlflow_config["run_name"] + "_logistic_regression") as mlops_run:
        train_and_evaluate_model("logistic_regression", LogisticRegression(), train_x, train_y, test_x, test_y, logreg_params, avg_method)

    # K-Nearest Neighbors (KNN)
    knn_params = {}
    with mlflow.start_run(run_name=mlflow_config["run_name"] + "_knn") as mlops_run:
        train_and_evaluate_model("knn", KNeighborsClassifier(), train_x, train_y, test_x, test_y, knn_params, avg_method)

    # Decision Tree
    dt_params = {}
    with mlflow.start_run(run_name=mlflow_config["run_name"] + "_decision_tree") as mlops_run:
        train_and_evaluate_model("decision_tree", DecisionTreeClassifier(), train_x, train_y, test_x, test_y, dt_params, avg_method)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    main(config_path=parsed_args.config)
