import logging

import pandas as pd
import xgboost as xgb
from dvclive import Live
from sklearn.model_selection import train_test_split


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
            random_state=0,
            n_estimators=50,
        )

        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        accuracy = float(model.score(X_test, y_test))
        live.log_metric("accuracy", accuracy, plot=True)
        live.log_sklearn_plot("calibration", y_test.values, y_score)
        # live.log_sklearn_plot("roc", y_test.values, y_score)
        live.log_sklearn_plot("confusion_matrix", y_test.values, y_pred, name="cm.json")
    logging.info("End")


if __name__ == "__main__":
    main()
