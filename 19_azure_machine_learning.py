"""
============================================================
AI-102 | Program 19 — Azure Machine Learning
Service : Azure Machine Learning
Skill   : Implement and manage AI models
============================================================
Features:
  • Connect to AML Workspace
  • Register and version datasets
  • Submit training jobs
  • Register trained models
  • Deploy to managed online endpoint
  • Monitor endpoint performance
============================================================
"""

import os
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    Model,
    Environment,
    BuildContext,
    CodeConfiguration,
)
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential, ClientSecretCredential

# AML Workspace config
SUBSCRIPTION_ID = os.getenv("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
RESOURCE_GROUP  = os.getenv("AZURE_RESOURCE_GROUP", "<your-rg>")
WORKSPACE_NAME  = os.getenv("AZURE_ML_WORKSPACE", "<your-aml-workspace>")

def get_ml_client():
    """
    Authenticate to Azure ML Workspace.
    Uses DefaultAzureCredential (env vars, managed identity, CLI login).
    """
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=SUBSCRIPTION_ID,
        resource_group_name=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )

# ── 1. Workspace Info ─────────────────────────────────────
def get_workspace_info() -> None:
    """Display workspace details and available compute."""
    ml_client = get_ml_client()
    ws = ml_client.workspaces.get(WORKSPACE_NAME)

    print("\n" + "="*65)
    print("  AZURE ML WORKSPACE")
    print("="*65)
    print(f"  Name     : {ws.name}")
    print(f"  Location : {ws.location}")
    print(f"  SKU      : {ws.sku}")

    print("\n  Compute targets:")
    for compute in ml_client.compute.list():
        print(f"    • {compute.name} [{compute.type}] — {compute.provisioning_state}")

    print("\n  Registered models:")
    for model in ml_client.models.list():
        print(f"    • {model.name} v{model.version}")

# ── 2. Register Dataset ───────────────────────────────────
def register_dataset(data_path: str, dataset_name: str) -> None:
    """
    Register a dataset as a versioned Data Asset in AML.
    Supports: URI_FILE, URI_FOLDER, MLTABLE types.
    """
    from azure.ai.ml.entities import Data

    ml_client = get_ml_client()

    data_asset = Data(
        path=data_path,
        type=AssetTypes.URI_FILE,   # or URI_FOLDER, MLTABLE
        description=f"Training data for {dataset_name}",
        name=dataset_name,
        version="1"
    )

    registered = ml_client.data.create_or_update(data_asset)

    print("\n" + "="*65)
    print("  DATASET REGISTRATION")
    print("="*65)
    print(f"  Name   : {registered.name}")
    print(f"  Version: {registered.version}")
    print(f"  Path   : {registered.path}")

# ── 3. Submit Training Job ────────────────────────────────
def submit_training_job(
    script_path: str,
    compute_name: str,
    environment_name: str = "AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest"
) -> str:
    """
    Submit a training script as a Command Job.
    Returns the job name for tracking.
    """
    ml_client = get_ml_client()

    job = command(
        code=os.path.dirname(script_path),         # Source directory
        command=f"python {os.path.basename(script_path)} "
                f"--learning_rate ${{inputs.learning_rate}} "
                f"--epochs ${{inputs.epochs}}",
        inputs={
            "learning_rate": 0.01,
            "epochs": 10,
        },
        environment=environment_name,
        compute=compute_name,
        display_name="ai102-training-job",
        description="Model training job for AI-102 demo",
        experiment_name="ai102-experiment",
        tags={"framework": "sklearn", "task": "classification"}
    )

    returned_job = ml_client.jobs.create_or_update(job)

    print("\n" + "="*65)
    print("  TRAINING JOB SUBMITTED")
    print("="*65)
    print(f"  Job Name    : {returned_job.name}")
    print(f"  Status      : {returned_job.status}")
    print(f"  Studio URL  : {returned_job.studio_url}")
    print(f"  Compute     : {compute_name}")

    return returned_job.name

# ── 4. Register Model ─────────────────────────────────────
def register_model(model_path: str, model_name: str, job_name: str = None) -> None:
    """
    Register a trained model as a versioned Model Asset.
    Can reference output from a training job.
    """
    ml_client = get_ml_client()

    if job_name:
        # Register from job output
        model_path = f"azureml://jobs/{job_name}/outputs/artifacts/paths/model/"

    model = Model(
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=model_name,
        description="Trained classification model",
        version="1",
        tags={"framework": "sklearn", "task": "text-classification"}
    )

    registered = ml_client.models.create_or_update(model)

    print("\n" + "="*65)
    print("  MODEL REGISTERED")
    print("="*65)
    print(f"  Name      : {registered.name}")
    print(f"  Version   : {registered.version}")
    print(f"  Asset ID  : {registered.id}")

# ── 5. Deploy to Online Endpoint ──────────────────────────
def deploy_online_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str = "1"
) -> str:
    """
    Deploy a registered model to a managed online endpoint.
    Creates endpoint → creates deployment → sets traffic.
    """
    ml_client = get_ml_client()

    print("\n" + "="*65)
    print("  DEPLOYING TO MANAGED ONLINE ENDPOINT")
    print("="*65)

    # Step 1: Create endpoint
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="AI-102 demo endpoint",
        auth_mode="key",
        tags={"model": model_name}
    )
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"  ✅ Endpoint created: {endpoint_name}")

    # Step 2: Create deployment
    deployment = ManagedOnlineDeployment(
        name="blue",                               # Deployment slot name
        endpoint_name=endpoint_name,
        model=f"azureml:{model_name}:{model_version}",
        instance_type="Standard_DS3_v2",
        instance_count=1,
        code_configuration=CodeConfiguration(
            code="./scoring",                      # Scoring script directory
            scoring_script="score.py"              # Entry script
        ),
        environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu@latest"
    )
    ml_client.online_deployments.begin_create_or_update(deployment).result()
    print(f"  ✅ Deployment 'blue' created")

    # Step 3: Set 100% traffic to 'blue'
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()
    print(f"  ✅ Traffic: 100% → blue")

    # Get endpoint URI
    ep = ml_client.online_endpoints.get(endpoint_name)
    print(f"  Endpoint URI: {ep.scoring_uri}")
    return ep.scoring_uri

# ── 6. Invoke Endpoint ────────────────────────────────────
def invoke_endpoint(endpoint_name: str, request_data: dict) -> None:
    """
    Call a deployed endpoint with test data.
    """
    import json
    ml_client = get_ml_client()

    result = ml_client.online_endpoints.invoke(
        endpoint_name=endpoint_name,
        request_file=None,
        deployment_name="blue",
        input_data=json.dumps(request_data)
    )

    print("\n" + "="*65)
    print("  ENDPOINT INVOCATION")
    print("="*65)
    print(f"  Input : {request_data}")
    print(f"  Output: {result}")

# ── 7. Get Endpoint Details ───────────────────────────────
def get_endpoint_details(endpoint_name: str) -> None:
    """List endpoint status, URI, and deployment traffic."""
    ml_client = get_ml_client()
    ep = ml_client.online_endpoints.get(endpoint_name)

    print("\n" + "="*65)
    print("  ENDPOINT DETAILS")
    print("="*65)
    print(f"  Name         : {ep.name}")
    print(f"  Status       : {ep.provisioning_state}")
    print(f"  Scoring URI  : {ep.scoring_uri}")
    print(f"  Swagger URI  : {ep.openapi_uri}")
    print(f"  Auth Mode    : {ep.auth_mode}")
    print(f"  Traffic      : {ep.traffic}")

    # Get keys
    keys = ml_client.online_endpoints.get_keys(endpoint_name)
    print(f"  Primary Key  : {keys.primary_key[:10]}...")

# ── Sample Training Script Content ────────────────────────
SAMPLE_TRAIN_SCRIPT = '''
import argparse
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=10)
args = parser.parse_args()

# Enable MLflow autologging
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train model
    model = LogisticRegression(C=args.learning_rate, max_iter=args.epochs * 100)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("learning_rate", args.learning_rate)
    mlflow.log_param("epochs", args.epochs)

    print(f"Accuracy: {accuracy:.4f}")

    # Save model
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(model, "outputs/model.pkl")
    print("Model saved to outputs/model.pkl")
'''

# ── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  AZURE MACHINE LEARNING — AI-102 DEMO")

    # Show what a training script looks like
    print("\n" + "="*65)
    print("  SAMPLE TRAINING SCRIPT (train.py)")
    print("="*65)
    print(SAMPLE_TRAIN_SCRIPT)

    # Workspace info (requires live connection)
    # get_workspace_info()
    # submit_training_job("./train.py", "cpu-cluster")
    # register_model("./outputs", "text-classifier")
    # deploy_online_endpoint("ai102-endpoint", "text-classifier")

    print("\n  KEY POINTS FOR AI-102:")
    print("  • MLClient is the main AML SDK entry point")
    print("  • DefaultAzureCredential handles auth automatically")
    print("  • command() creates a Command Job from a Python script")
    print("  • Data assets: URI_FILE, URI_FOLDER, MLTABLE")
    print("  • Models: CUSTOM_MODEL, MLFLOW_MODEL types")
    print("  • ManagedOnlineEndpoint → real-time inference")
    print("  • BatchEndpoint → batch/async inference")
    print("  • Traffic splits support blue/green deployments")
    print("  • MLflow autolog() captures metrics automatically")
    print("  • Endpoint auth: 'key' or 'aml_token'")
    print("="*65 + "\n")
