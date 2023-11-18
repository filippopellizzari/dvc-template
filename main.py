import logging

import pandas as pd
import xgboost as xgb
from dvclive.xgb import DVCLiveCallback
from sklearn.model_selection import train_test_split

from dvclive import Live


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Start")
    df = pd.read_csv("data/Train.csv", engine="pyarrow")
    logging.info("Train val test split")
    train, test = train_test_split(df, random_state=0, stratify=df["labels"])
    train, val = train_test_split(train, random_state=0, stratify=train["labels"])
    X_train = train.drop(columns=["labels"])
    y_train = train["labels"]
    X_val = val.drop(columns=["labels"])
    y_val = val["labels"]
    X_test = test.drop(columns=["labels"])
    y_test = test["labels"]
    logging.info("Model training")
    with Live("custom_dir") as live:
        model = xgb.XGBClassifier(
            n_estimators=50,
            callbacks=[DVCLiveCallback()],
        )

        model.fit(X_train, y_train)
        accuracy = float(model.score(X_test, y_test))
        live.log_metric("accuracy", accuracy, plot=True)
    logging.info("End")


if __name__ == "__main__":
    main()
