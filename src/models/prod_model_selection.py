import joblib
import mlflow
import argparse
from pprint import pprint
from train_models import read_params
from mlflow.tracking import MlflowClient

def log_production_model(config_path):
    config = read_params(config_path)
    mlflow_config = config["mlflow_config"] 
    model_names = mlflow_config["model_names"]  # Assuming model_names is a list of model names
    model_dir = config["model_dir"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    client = MlflowClient()
    experiment = mlflow.get_experiment_by_name(mlflow_config["experiment_name"])
    if experiment is None:
        raise ValueError(f"Experiment '{mlflow_config['experiment_name']}' not found.")

    experiment_id = experiment.experiment_id

    for model_name in model_names:
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        if runs.empty:
            print(f"No runs found for model name: {model_name}")
            continue
        
        max_accuracy = max(runs["metrics.accuracy"])
        max_accuracy_run_id = list(runs[runs["metrics.accuracy"] == max_accuracy]["run_id"])[0]

        for mv in client.search_model_versions(f"name='{model_name}'"):
            mv = dict(mv)

            if mv["run_id"] == max_accuracy_run_id:
                current_version = mv["version"]
                logged_model = mv["source"]
                pprint(mv, indent=4)
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_version,
                    stage="Production"
                )
            else:
                current_version = mv["version"]
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_version,
                    stage="Staging"
                )

    loaded_model = mlflow.pyfunc.load_model(logged_model)
    joblib.dump(loaded_model, model_dir)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = log_production_model(config_path=parsed_args.config)
