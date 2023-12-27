import logging
import pathlib

import pandas as pd
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
    y_train = pd.DataFrame(train["labels"])
    X_val = val.drop(columns=["labels"])
    y_val = pd.DataFrame(val["labels"])
    X_test = test.drop(columns=["labels"])
    y_test = pd.DataFrame(test["labels"])
    logging.info("Output")
    pathlib.Path("data/dataset").mkdir(parents=True, exist_ok=True)
    X_train.to_parquet("data/dataset/X_train.parquet")
    y_train.to_parquet("data/dataset/y_train.parquet")
    X_val.to_parquet("data/dataset/X_val.parquet")
    y_val.to_parquet("data/dataset/y_val.parquet")
    X_test.to_parquet("data/dataset/X_test.parquet")
    y_test.to_parquet("data/dataset/y_test.parquet")
    logging.info("End")


if __name__ == "__main__":
    main()
