"Baseline script for model quality monitoring"""
import logging
import pathlib
import pickle
import tarfile

import numpy as np
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

if __name__ == "__main__":
    logger.info("Setup model quality baline dataset")

    # Load the trained model
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    # In this case the we are using directly the model
    # it may be more efficient to deploy the model and use the endpoint
    # to do this.
    logger.debug("Loading sklearn model.")
    model = pickle.load(open("model.pkl", "rb"))

    # set the output dir
    output_dir = "/opt/ml/processing/validate"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    data_path = "/opt/ml/processing/validation/validation.csv"
    validate_set_df = pd.read_csv(data_path, header=None)
    logger.info(validate_set_df.describe())

    # labels this is of type pandas.core.series.Series
    y_test = validate_set_df.iloc[:, 0]
    validate_set_df.drop(validate_set_df.columns[0], axis=1, inplace=True)
    topredict = validate_set_df.values

    # predictions is numpy.ndarray
    logger.info("Performing predictions against test data.")
    predictions = model.predict(topredict)
    df_predictions = pd.DataFrame(predictions)

    # create te df to output as validate data for Model Quality baseline
    out_df = pd.concat([df_predictions, y_test], axis=1)
    logger.info(out_df.describe())

    # write model quality baseline dataset
    out_df.to_csv(
        f"{output_dir}/baseline.csv",
        index=False,
        header=['prediction', 'label'])

    logger.info(
        f"Model quality baseline dataset in {output_dir}/baseline.csv")
