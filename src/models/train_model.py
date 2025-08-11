from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.models.classifier import SklearnClassifier
from src.utils.config import load_config
from src.utils.guardrails import validate_evaluation_metrics
from src.utils.store import AssignmentStore
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV


@validate_evaluation_metrics
def main():
    store = AssignmentStore()
    config = load_config()

    df = store.get_processed("transformed_dataset.csv")
    df_train, df_test = train_test_split(df, test_size=config["test_size"])

    X_train = df_train[config["features"]]
    y_train = df_train[config["target"]]
    X_test = df_test[config["features"]]
    y_test = df_test[config["target"]]

    # Define base estimator without params to tune
    # rf_estimator = RandomForestClassifier(random_state=33, class_weight='balanced')
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    # Define parameter distributions for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(5, 30),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None]
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=30,
        scoring='roc_auc',
        cv=3,
        verbose=2,
        random_state=42,
        n_jobs=-1
    )

    # Fit RandomizedSearchCV to find best hyperparameters
    random_search.fit(X_train, y_train)

    print("Best params:", random_search.best_params_)
    print("Best ROC AUC:", random_search.best_score_)

    # Use the best estimator to wrap in your SklearnClassifier (assuming it takes an estimator and feature/target lists)
    best_rf = random_search.best_estimator_
    model = SklearnClassifier(best_rf, config["features"], config["target"])

    # Train model on full training data (optional, since RandomizedSearchCV already fits it)
    model.train(df_train)

    # Evaluate on test data
    metrics = model.evaluate(df_test)

    # Save model and metrics
    store.put_model("saved_model.pkl", model)
    store.put_metrics("metrics.json", metrics)


if __name__ == "__main__":
    main()
