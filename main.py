import logging

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score
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
    y_train = train["labels"].values
    X_test = test.drop(columns=["labels"])
    y_test = test["labels"].values
    logging.info("Model training")
    with Live(report="md") as live:
        model = xgb.XGBClassifier(
            random_state=0,
            n_estimators=200,
        )

        model.fit(X_train, y_train)
        logging.info("Evaluation")
        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        accuracy_test = float(model.score(X_test, y_test))
        roc_auc_test = float(roc_auc_score(y_test, y_score))
        live.log_metric("accuracy:test", accuracy_test, plot=False)
        live.log_metric("roc_auc:test", roc_auc_test, plot=False)
        live.log_sklearn_plot("calibration", y_test, y_score)

        prec, recall, _ = precision_recall_curve(
            y_test, y_score, pos_label=model.classes_[1]
        )
        prec_recall_dict = pd.DataFrame({"precision": prec, "recall": recall}).to_dict(
            "records"
        )
        live.log_plot(
            "precision_recall",
            prec_recall_dict,
            x="recall",
            y="precision",
            template="linear",
            title="Precision Recall plot",
        )

        feat_importance_dict = (
            pd.DataFrame(
                {
                    "feature": model.feature_names_in_,
                    "importance": model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .to_dict("records")
        )
        live.log_plot(
            "feature_importance",
            feat_importance_dict,
            x="importance",
            y="feature",
            template="bar_horizontal",
            title="Feature importance",
        )
        fig, _ = plt.subplots()
        xgb.plot_importance(model)
        plt.tight_layout()
        live.log_image("feature_importance.png", fig)

        live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="cm.json")
        live.make_report()
    logging.info("End")


if __name__ == "__main__":
    main()
