import logging

import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score

from dvclive import Live


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Start")
    X_test = pd.read_parquet("data/dataset/X_test.parquet")
    y_test = pd.read_parquet("data/dataset/y_test.parquet")
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    logging.info("Evaluation")
    with Live() as live:
        y_score = model.predict_proba(X_test)[:, 1].tolist()
        y_pred = model.predict(X_test)
        accuracy_test = float(model.score(X_test, y_test))
        roc_auc_test = float(roc_auc_score(y_test, y_score))
        live.log_metric("accuracy:test", accuracy_test, plot=False)
        live.log_metric("roc_auc:test", roc_auc_test, plot=False)
        live.log_sklearn_plot("calibration", y_test, y_score)

        """
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
        """
        live.log_sklearn_plot(
            "precision_recall",
            y_test.values.tolist(),
            y_score,
            # drop_intermediate=True,
        )

        # feature importance
        fig, _ = plt.subplots()
        xgb.plot_importance(model)
        plt.tight_layout()
        live.log_image("feature_importance.png", fig)
        # confusion matrix
        live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="cm.json")
    logging.info("End")


if __name__ == "__main__":
    main()
