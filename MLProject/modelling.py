import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

def train(data_path):
    mlflow.set_experiment("WorkflowCI")
    
    # df = pd.read_csv(f"{data_path}/dataset.csv")
    df = pd.read_csv(f"{data_path}/TelcoCustomerChurn_preprocessing.csv")
    # df = pd.read_csv(f"{data_path}/TelcoCustomerChurn_preprocessing.csv")


    
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]
    
    with mlflow.start_run():
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_metric("accuracy", model.score(X, y))
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    train(args.data_path)
