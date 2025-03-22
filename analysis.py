import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import openpyxl
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from mlflow.models.signature import infer_signature

# Configura o URI de rastreamento do MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("credit_card_fraud_detection")

def load_data(dataset_name):
    df = pd.read_csv(dataset_name)
    return df

def preprocess_data(df):
    df = df.drop_duplicates()

    # Normalização
    scaler = StandardScaler()
    df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])

    # Balanceamento
    X = df.drop(columns=['Class'])
    y = df['Class']

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    df_resampled = pd.concat([X_res, y_res], axis=1)
    return X_res, y_res, df_resampled

def train_logistic_regression(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regressão Logística
    lr = LogisticRegression(solver='liblinear', random_state=42)

    lr.fit(X_train, y_train)
    y_test_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)

    with mlflow.start_run(run_name="Logistic_Regression"):
        mlflow.log_params(lr.get_params())
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.set_tag("dataset_used", dataset_name)

        signature = infer_signature(X_train, y_train)
        model_info = mlflow.sklearn.log_model(lr, "logistic_regression_model", 
                                             signature=signature, 
                                             input_example=X_train, 
                                             registered_model_name="LogisticRegression")

        loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
        predictions = loaded_model.predict(X_test)

        result = pd.DataFrame(X_test, columns=X.columns.values)
        result["label"] = y_test.values
        result["predictions"] = predictions

        # Usando o MLflow para avaliação (opcional)
        mlflow.evaluate(
            data=result,
            targets="label",
            predictions="predictions",
            model_type="classifier",
        )

        print(result.head())
    return lr

def train_random_forest(X, y, dataset_name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10],
    }

    rf = RandomForestClassifier(random_state=42)

    for params in (dict(zip(param_grid.keys(), values)) for values in 
                   [(n, d, s) for n in param_grid["n_estimators"] 
                               for d in param_grid["max_depth"] 
                               for s in param_grid["min_samples_split"]]):
        
        rf.set_params(**params)
        rf.fit(X_train, y_train)
        y_test_pred = rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred)
        recall = recall_score(y_test, y_test_pred)

        with mlflow.start_run(run_name=f"RF_{params['n_estimators']}_{params['max_depth']}_{params['min_samples_split']}"):
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.set_tag("dataset_used", dataset_name)

            signature = infer_signature(X_train, y_train)
            model_info = mlflow.sklearn.log_model(rf, "random_forest_model", 
                                                 signature=signature, 
                                                 input_example=X_train, 
                                                 registered_model_name="RandomForestGridSearch")

            loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
            predictions = loaded_model.predict(X_test)

            result = pd.DataFrame(X_test, columns=X.columns.values)
            result["label"] = y_test.values
            result["predictions"] = predictions

            # Usando o MLflow para avaliação (opcional)
            mlflow.evaluate(
                data=result,
                targets="label",
                predictions="predictions",
                model_type="classifier",
            )

            print(result.head())
    return rf

def main():
    dataset_name = "creditcard.csv"
    df = load_data(dataset_name)
    X, y, mlflow_dataset = preprocess_data(df)

    # Treinar a Regressão Logística
    lr_model = train_logistic_regression(X, y, dataset_name)

    # Treinar o Random Forest
    rf_model = train_random_forest(X, y, dataset_name)

if __name__ == "__main__":
    main()
