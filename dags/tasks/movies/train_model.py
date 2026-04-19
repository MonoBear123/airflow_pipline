import numpy as np
import polars as pl
import mlflow
from pathlib import Path
from sklearn.model_selection import GridSearchCV
from mlflow.models import infer_signature
from sklearn.preprocessing import (
    StandardScaler,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_BASE = Path("data")
MLRUNS = Path("mlruns")


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
        "depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "iterations": [300, 500, 800],
        "l2_leaf_reg": [1, 3, 5],
    }

    mlflow.set_tracking_uri(f"./{MLRUNS}")
    mlflow.set_experiment("movies_sgd")
    with mlflow.start_run(run_name="CatRegressor"):
        lr = CatBoostRegressor(
            loss_function="RMSE", eval_metric="RMSE", random_state=42, verbose=100
        )
        clf = GridSearchCV(
            lr, params, cv=3, n_jobs=1, scoring="neg_root_mean_squared_error"
        )
        clf.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=100,
        )

        best = clf.best_estimator_

        y_pred_scaled = best.predict(X_val)
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1))

        metrics = eval_metrics(y_val_orig, y_pred)

        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_train, best.predict(X_train))
        mlflow.sklearn.log_model(best, name="sgd_model", signature=signature)

        output = DATA_BASE / "output/movie"
        output.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, DATA_BASE / "sgd_movies.skops")
        joblib.dump(scaler, DATA_BASE / "scaler_movies.skops")
        joblib.dump(power_trans, DATA_BASE / "power_trans_movies.skops")

        output = DATA_BASE / "output/movie"
        output.mkdir(exist_ok=True)
        joblib.dump(best, DATA_BASE / "sgd_movies.skops")
        joblib.dump(scaler, DATA_BASE / "scaler_movies.skops")
        joblib.dump(power_trans, DATA_BASE / "power_trans_movies.skops")


train()
