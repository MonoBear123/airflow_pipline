import numpy as np
import polars as pl
import joblib
import mlflow
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from mlflow.models import infer_signature
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_BASE = Path("/home/gintoki/airflow/data")
MLRUNS = Path("/home/gintoki/airflow/mlruns")


def scale_frame(df):

    df = df.clone()
    X_categorical = df.drop(["budget", "revenue", "runtime"]).to_numpy()
    X_numeric = df.select(["budget", "runtime"]).to_numpy()
    y = df.select("revenue").to_numpy().ravel()

    (X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val) = train_test_split(
        X_numeric, X_categorical, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_num_train_scaled = scaler.fit_transform(X_num_train)
    X_num_val_scaled = scaler.transform(X_num_val)
    X_train = np.hstack([X_num_train_scaled, X_cat_train])
    X_val = np.hstack([X_num_val_scaled, X_cat_val])

    power = PowerTransformer()
    y_train_scaled = power.fit_transform(y_train.reshape(-1, 1)).ravel()
    y_val_scaled = power.transform(y_val.reshape(-1, 1)).ravel()

    return X_train, X_val, y_train_scaled, y_val_scaled, scaler, power


def eval_metrics(actual, pred):

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train():

    df = pl.read_csv(DATA_BASE / "processed/movie/movies.csv")
    X_train, X_val, y_train, y_val, scaler, power_trans = scale_frame(df)

    params = {
        "alpha": [0.0001, 0.001, 0.01, 0.05, 0.1],
        "l1_ratio": [0.001, 0.01, 0.05, 0.2],
        "penalty": ["l1", "l2", "elasticnet"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "fit_intercept": [False, True],
    }

    mlflow.set_tracking_uri(f"file://{MLRUNS}")
    mlflow.set_experiment("movies_sgd")
    with mlflow.start_run(run_name="SGDRegressor"):
        lr = SGDRegressor(random_state=42, max_iter=5000)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring="r2")
        clf.fit(X_train, y_train)

        best = clf.best_estimator_

        y_pred_scaled = best.predict(X_val)
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1))

        metrics = eval_metrics(y_val_orig, y_pred)
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, name="sgd_model", signature=signature)

        joblib.dump(best, DATA_BASE / "output/movie/sgd_movies.skops")
        joblib.dump(scaler, DATA_BASE / "output/movie/scaler_movies.skops")
        joblib.dump(power_trans, DATA_BASE / "output/movie/power_trans_movies.skops")


train()
