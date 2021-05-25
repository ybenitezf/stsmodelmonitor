"""Load and prepare sts dataset."""

import os
import pickle
import pathlib
import boto3
import logging
import argparse
import warnings
import numpy as np
import sklearn
import pandas as pd
from subprocess import run

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# main routine
if __name__ == "__main__":
    logger.debug("Starting modeling.")
    base_dir = "/opt/ml/processing"
    bucket = "sts-demo-datasets"

    logger.debug("Reading train data.")
    train_path = os.environ.get('SM_CHANNEL_TRAIN')
    logger.info(run("ls "+train_path, shell=True))
    df = pd.read_csv(train_path + "/train.csv", header=None)


    # Extracting data from pandas DataFrame
    Y_train = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_train = df.values

    logger.info("Starting model creation.")
    '''
    Classification
    multi_class = 'ovr' is set for this problem where there are only two binary classes.
    '''
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter=200, n_jobs=4, multi_class='ovr')

    # fit the model with data
    logreg.fit(X_train, Y_train)

    logger.info("Saving trained model.")

    filename = f"model"
    model_path = os.environ.get('SM_MODEL_DIR')+"/"
    pickle.dump(logreg, open(model_path + filename + '.pkl', 'wb'))

    logger.info("End modeling.")
