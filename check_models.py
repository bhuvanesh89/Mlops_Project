import mlflow
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI to the running server
mlflow.set_tracking_uri("http://127.0.0.1:1234")

# Check registered models
client = MlflowClient()
models = client.search_registered_models()

if not models:
    print("No registered models found in MLflow.")
else:
    for model in models:
        print("Registered Model:", model.name)
