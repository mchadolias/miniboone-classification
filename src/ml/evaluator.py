from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test):
    preds = model.predict_proba(X_test)[:, 1]
    labels = (preds > 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_test, preds),
        "f1_score": f1_score(y_test, labels),
        "confusion_matrix": confusion_matrix(y_test, labels).tolist(),
        "classification_report": classification_report(y_test, labels, output_dict=True),
    }
