import logging
import pathlib

import dvc.api
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

    params = dvc.api.params_show()

    logging.info("Training")
    model = xgb.XGBClassifier(random_state=0, **params)

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    model.save_model("model.json")
    logging.info("End")


if __name__ == "__main__":
    main()
