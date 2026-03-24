from src.data_loader import load_data
from src.train import train_models
from src.evaluate import evaluate_model, format_results


def main() -> None:
    df = load_data()
    best_model, results, X_train, y_train, X_test, y_test = train_models(df)

    summaries = [
        {"name": r.name, "auc": r.auc, "best_params": r.best_params}
        for r in results
    ]
    print("Model selection summary:\n" + format_results(summaries))

    metrics = evaluate_model(best_model, X_test, y_test)
    print("\nBest model metrics:")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{key}: {metrics[key]:.3f}")


if __name__ == "__main__":
    main()
