import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def main():
    data = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)

        acc = model.score(X_test, y_test)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print("Accuracy:", acc)

if __name__ == "__main__":
    main()
