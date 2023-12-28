import logging

import optuna
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.metrics import roc_auc_score


class Tuning:
    def __init__(self, n_trials, X_train, y_train, X_val, y_val):
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 30, 200),
            "early_stopping_rounds": trial.suggest_int("early_stopping_rounds", 5, 20),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "max_depth": trial.suggest_int("max_depth", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.05, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        }
        model = xgb.XGBClassifier(random_state=0, **params)
        model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=False,
        )
        y_score = model.predict_proba(self.X_val)[:, 1]
        roc_auc = float(roc_auc_score(self.y_val, y_score))
        return roc_auc

    def compute(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.n_trials)
        return study.best_params


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Start")
    X_train = pd.read_parquet("data/dataset/X_train.parquet")
    y_train = pd.read_parquet("data/dataset/y_train.parquet")
    X_val = pd.read_parquet("data/dataset/X_val.parquet")
    y_val = pd.read_parquet("data/dataset/y_val.parquet")
    logging.info("Tuning")
    tuning = Tuning(50, X_train, y_train, X_val, y_val)
    best_params = tuning.compute()
    logging.info(f"{best_params=}")
    with open("params.yaml", "w") as f:
        yaml.dump(best_params, f)

    logging.info("End")


if __name__ == "__main__":
    main()
