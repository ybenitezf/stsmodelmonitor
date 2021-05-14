"""Load and prepare sts dataset."""

import os
import pickle
import string
import pathlib
import boto3
import logging
import argparse
import warnings
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# main routine
if __name__ == "__main__":
    logger.debug("Starting modeling.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    input_data = args.input_data

    '''
    Load dataset
    '''

    base_dir = "/opt/ml/modeling"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    # bucket = "sts-demo-datasets"
    # key = "distances_matrix.pkl"

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)

    filename = f"{base_dir}/data/{key}"
    s3_client = boto3.resource("s3")
    s3_client.Bucket(bucket).download_file(key, filename)

    logger.info("Reading downloaded data.")

    train_data = open(filename, 'rb')
    x, y = pickle.load(train_data)

    logger.info("Reading data finished.")
    os.unlink(filename)

    '''
    Spliting Data
    '''

    def get_train_test(x,y):
        """
        Perpare a stratified train and test split
        """
        train_size = 0.7
        test_size = 1-train_size
        stratified_split = StratifiedShuffleSplit(n_splits=5,test_size=test_size)
        
        for train_index,test_index in stratified_split.split(x,y):
            train_x, test_x = x[train_index], x[test_index]
            train_y, test_y = y[train_index], y[test_index]
        return train_x,train_y,test_x,test_y

    X_train, Y_train, X_test, Y_test = get_train_test(x, y)

    '''
    Dump Splitted Data
    '''

    logger.info("Saving splitted data.")
    
    filepath = f"{base_dir}/data/"
    s3_client = boto3.client('s3')

    data = [X_train, X_test, Y_train, Y_test]

    filename = f"strat-split-data"
    pickle.dump(data, open(filepath + filename + '.pkl', 'wb'))

    logger.info("Uploading data to bucket: %s, key: %s", bucket, filename + '.pkl')
    s3_client.upload_file(filepath + filename + '.pkl', bucket, filename + '.pkl')
    os.unlink(filepath + filename + '.pkl')

    '''
    Classification
    multi_class = 'ovr' is set for this problem where there are only two binary classes.
    '''

    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(max_iter=200, n_jobs=4, multi_class='ovr')

    # fit the model with data
    logreg.fit(X_train, Y_train)

    '''
    Save Model
    '''

    logger.info("Saving trained model.")

    filename = f"logreg_model"
    pickle.dump(data, open(filepath + filename + '.pkl', 'wb'))

    logger.info("Uploading data to bucket: %s, key: %s", bucket, filename + '.pkl')
    s3_client.upload_file(filepath + filename + '.pkl', bucket, filename + '.pkl')
    os.unlink(filepath + filename + '.pkl')

    logger.info("Model saved.")

    logger.info("End modeling.")
