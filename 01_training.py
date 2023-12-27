import logging
import pathlib

import pandas as pd
import xgboost as xgb


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

    logging.info("Training")
    model = xgb.XGBClassifier(
        random_state=0,
        n_estimators=200,
        early_stopping_rounds=20,
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pathlib.Path("models").mkdir(parents=True, exist_ok=True)
    model.save_model("models/model.json")
    logging.info("End")


if __name__ == "__main__":
    main()
