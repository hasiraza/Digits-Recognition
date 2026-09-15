from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score
import mlflow
import mlflow.sklearn
mlflow.set_experiment("Digits Recognition")

# loading dataset
X,y=load_digits(return_X_y=True)

# spilting the dataset into two parts training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

# MLflow experiment
with mlflow.start_run():
    model=SVC(kernel="linear",C=100,gamma=0.5,max_iter=500)
    mlflow.log_param("C",100)
    mlflow.log_param("gamma",0.5)
    mlflow.log_param("max_iter",500)
    mlflow.log_param("kernel",'linear')
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    print(f"Accuracy: {accuracy}")
    classification_report = classification_report(y_test,y_pred)
    print(f"Classification Report: {classification_report}")
    mlflow.log_metric("accuracy",accuracy)
    with open("classification_report","w") as f:
        f.write(classification_report)
    mlflow.log_artifact("classification_report.txt")

    mlflow.sklearn.log_model(
        sk_model=model,
        name="model"
    )


