from src.get_data import read_params
import argparse
import mlflow
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import os

def log_production_model(config_path):
    config = read_params(config_path)

    mlflow_config = config["mlflow_config"]
    model_name = mlflow_config["registered_model_name"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    
    mlflow.set_tracking_uri(remote_server_uri)
    client = MlflowClient()

    # Get the correct experiment ID dynamically
    experiment = client.get_experiment_by_name(mlflow_config["experiment_name"])
    if not experiment:
        print(f"Experiment '{mlflow_config['experiment_name']}' not found.")
        return
    
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])

    if runs.empty:
        print("No runs found for this experiment.")
        return

    # Find the best model with lowest MAE
    best_run = runs.loc[runs["metrics.mae"].idxmin()]
    best_run_id = best_run["run_id"]
    print(f"Best run ID: {best_run_id} with MAE: {best_run['metrics.mae']}")

    logged_model = None

    for mv in client.search_model_versions(f"name='{model_name}'"):
        mv = dict(mv)
        current_version = mv["version"]

        if mv["run_id"] == best_run_id:
            logged_model = mv["source"]  # Store the best model path
            pprint(mv, indent=4)

            # Move all old Production models to Staging
            if mv["current_stage"] == "Production":
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_version,
                    stage="Staging"
                )
                print(f"Old Production model {current_version} moved to Staging.")

            # Move the best model to Production
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Production"
            )
            print(f"Model {current_version} moved to Production.")
        else:
            if mv["current_stage"] != "Staging":
                client.transition_model_version_stage(
                    name=model_name,
                    version=current_version,
                    stage="Staging"
                )
                print(f"Model {current_version} moved to Staging.")

    if not logged_model:
        print("Error: No valid model found for loading.")
        return

    # Load and save best model
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    model_path = config["webapp_model_dir"]
    joblib.dump(loaded_model, model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_production_model(config_path=parsed_args.config)
