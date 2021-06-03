"""This will be used as an entry point when serving the model"""
from sagemaker_containers.beta.framework import content_types, encoders
import numpy as np
import joblib
import os

def model_fn(model_dir):
    """Deserialized and return fitted model
    Note that this should have the same name as the serialized model in the main method
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf


def input_fn(input_data, content_type):
    """Takes request data and de-serializes the data into an object for prediction.
        When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
        the model server receives two pieces of information:
            - The request Content-Type, for example "application/json"
            - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
        The input_fn is responsible to take the request data and pre-process it before prediction.
    Args:
        input_data (obj): the request data.
        content_type (str): the request Content-Type.
    Returns:
        (obj): data ready for prediction.
    """
    np_array = encoders.decode(input_data, content_type)
    ret = np_array.astype(np.float32) if content_type in content_types.UTF8_TYPES else np_array
    # reshaping if contains a single sample, necesary if when using CSV as
    # content_type
    if len(ret.shape) == 1:
        # the model expect a 2D array
        ret = ret.reshape(1,-1)

    return ret
