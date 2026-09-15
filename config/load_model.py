import mlflow

def load_models():

    model_uri = "models:/Digits-SVC/1"

    model = mlflow.sklearn.load_model(model_uri)

    return model