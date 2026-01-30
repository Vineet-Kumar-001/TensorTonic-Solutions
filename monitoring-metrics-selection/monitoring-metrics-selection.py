import math

def compute_monitoring_metrics(system_type, y_true, y_pred):
    """
    Compute the appropriate monitoring metrics for the given system type.
    """

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    n = len(y_true)
    metrics = []

    if system_type == "classification":
        TP = FP = FN = TN = 0

        for yt, yp in zip(y_true, y_pred):
            if yt == 1 and yp == 1:
                TP += 1
            elif yt == 0 and yp == 1:
                FP += 1
            elif yt == 1 and yp == 0:
                FN += 1
            else:
                TN += 1

        accuracy = safe_div(TP + TN, n)
        precision = safe_div(TP, TP + FP)
        recall = safe_div(TP, TP + FN)
        f1 = safe_div(2 * precision * recall, precision + recall)

        metrics = [
            ("accuracy", accuracy),
            ("f1", f1),
            ("precision", precision),
            ("recall", recall),
        ]

    elif system_type == "regression":
        abs_err = 0
        sq_err = 0

        for yt, yp in zip(y_true, y_pred):
            diff = yt - yp
            abs_err += abs(diff)
            sq_err += diff ** 2

        mae = abs_err / n
        rmse = math.sqrt(sq_err / n)

        metrics = [
            ("mae", mae),
            ("rmse", rmse),
        ]

    elif system_type == "ranking":
        pairs = list(zip(y_true, y_pred))
        pairs.sort(key=lambda x: x[1], reverse=True)

        top3 = pairs[:3]
        relevant_top3 = sum(rel for rel, _ in top3)
        total_relevant = sum(y_true)

        precision_at_3 = safe_div(relevant_top3, 3)
        recall_at_3 = safe_div(relevant_top3, total_relevant)

        metrics = [
            ("precision_at_3", precision_at_3),
            ("recall_at_3", recall_at_3),
        ]

    return sorted(metrics, key=lambda x: x[0])
