import pickle

def load_models():
    model=None
    try:
        with open('Model/model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")

    except FileNotFoundError:
        print("Error: The file 'Model.pickle' was not found. Please check the path.")

    except pickle.UnpicklingError:
        print("Error: The file is corrupted or could not be unpickled.")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return model