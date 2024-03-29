import logging

import mlflow
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from mlflow.data.pandas_dataset import PandasDataset
from sklearn.metrics import precision_recall_curve, roc_auc_score

from dvclive import Live


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Start")
    X_test = pd.read_parquet("data/dataset/X_test.parquet")
    y_test = pd.read_parquet("data/dataset/y_test.parquet").values.tolist()
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    logging.info("Evaluation")
    y_score = model.predict_proba(X_test)[:, 1].tolist()
    y_pred = model.predict(X_test)
    accuracy_test = float(model.score(X_test, y_test))
    roc_auc_test = float(roc_auc_score(y_test, y_score))
    logging.info("DVC live")
    with Live() as live:
        live.log_metric("accuracy:test", accuracy_test, plot=False)
        live.log_metric("roc_auc:test", roc_auc_test, plot=False)
        # calibration plot
        live.log_sklearn_plot("calibration", y_test, y_score)
        # precision recall plot
        live.log_sklearn_plot(
            "precision_recall",
            y_test,
            y_score,
        )
        # roc auc plot
        live.log_sklearn_plot(
            "roc",
            y_test,
            y_score,
        )
        # feature importance
        fig, _ = plt.subplots()
        xgb.plot_importance(model)
        plt.tight_layout()
        live.log_image("feature_importance.png", fig)
        # confusion matrix
        live.log_sklearn_plot("confusion_matrix", y_test, y_pred, name="cm.json")

    """
    logging.info("MLFlow")
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_experiment("MLflow Quickstart")
    test_set: PandasDataset = mlflow.data.from_pandas(X_test)
    with mlflow.start_run():
        mlflow.log_metric("accuracy-test", accuracy_test)
        mlflow.log_metric("roc_auc-test", roc_auc_test)

        signature = mlflow.models.infer_signature(X_test, model.predict(X_test))

        model_info = mlflow.xgboost.log_model(model, "classifier", signature=signature)
        mlflow.log_input(test_set, context="testing")
        mlflow.log_figure(fig, "feature_importance.png")
    """

    logging.info("End")


if __name__ == "__main__":
    main()
