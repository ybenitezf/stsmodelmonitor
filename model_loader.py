"""This will be used as an entry point when serving the model"""
import joblib
import os

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf
