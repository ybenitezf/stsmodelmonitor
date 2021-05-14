"""Load and prepare sts dataset."""

import os
import csv
import pickle
import string
import pathlib
import boto3
import logging
import argparse
import warnings
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import * #support sparse matrix inputs
from scipy.spatial.distance import * #do not support sparse matrix inputs

warnings.filterwarnings(action='ignore')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# helper functions
def count_words(list_of_words):
    """"""
    corpus_dict = {}
    for w in list_of_words:
        corpus_dict[w] = corpus_dict.get(w, 0.0) + 1.0
        
    return corpus_dict

def CountVectorizer(string_dict, set_both_strings):
    """Whit padding included""" 
    vector = []
    for key in set_both_strings: 
        if key in string_dict: 
            vector.append(int(string_dict[key])) 
        else:
            vector.append(0)
    return vector

def min_max_range(x, range_values):
    return [round(((xx-min(x))/(1.0*(max(x)-min(x))))*(range_values[1]-range_values[0])+range_values[0],5) for xx in x]

# main routine
if __name__ == "__main__":
    logger.debug("Starting preprocessing.")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str, required=True)
    args = parser.parse_args()
    input_data = args.input_data

    '''
    Load dataset

    Loading txt (csv) file, expected format is:
    Quality ID#1 ID#2 String#1 String#2
    Separator is tab (\t) character
    '''

    base_dir = "/opt/ml/processing"
    pathlib.Path(f"{base_dir}/data").mkdir(parents=True, exist_ok=True)
    bucket = input_data.split("/")[2]
    key = "/".join(input_data.split("/")[3:])
    # bucket = "sts-demo-datasets"
    # key = "stsmsrpc.txt"

    logger.info("Downloading data from bucket: %s, key: %s", bucket, key)

    filename = f"{base_dir}/data/{key}"
    s3_client = boto3.resource("s3")
    s3_client.Bucket(bucket).download_file(key, filename)

    logger.info("Reading downloaded data.")

    sentences = []
    y = [] # y-data
    with open(filename, errors = 'ignore') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        # skip first (header) row
        next(csv_reader)
        for row in csv_reader:
            try:
                sentences.append((row[3], row[4]))
                y.append(float(row[0]))
            except:
                pass

    logger.info("Reading data finished.")
    os.unlink(filename)

    '''
    Feature Engineering
    '''

    _VALID_METRICS_ = ['euclidean', 'l2', 'l1', 'manhattan', 'cityblock',
        'braycurtis', 'canberra', 'chebyshev', 'correlation',
        'cosine', 'dice', 'hamming', 'jaccard', 'kulsinski',
        'matching', 'minkowski', 'rogerstanimoto',
        'russellrao', 'seuclidean', 'sokalmichener',
        'sokalsneath', 'sqeuclidean', 'yule',]

    distances_matrix = []
    for s1, s2 in sentences:
        # clean each pair of sentences
        s1 = s1.translate(str.maketrans("", "", string.punctuation))
        s2 = s2.translate(str.maketrans("", "", string.punctuation))

        s1 = count_words(s1.split())
        s2 = count_words(s2.split())

        c = set(s1).union(set(s2))

        s1 = CountVectorizer(s1, c)
        s2 = CountVectorizer(s2, c)

        s1 = np.array(s1,dtype=np.int32)
        s2 = np.array(s2,dtype=np.int32)

        s1 = s1.reshape(1,-1)
        s2 = s2.reshape(1,-1)

        # get all distances
        vector = []
        for distance in _VALID_METRICS_:
            _, dist =  pairwise_distances_argmin_min(s1, s2, axis=1, metric=distance)
            vector.append(dist[0])

        distances_matrix.append(vector)

    '''
    Scaling
    '''
    _DISTANCE_MATRIX_NORM_ = []
    for vector in distances_matrix:
        _DISTANCE_MATRIX_NORM_.append(min_max_range(vector, (0.0,1.0)))
    distances_matrix = np.array(_DISTANCE_MATRIX_NORM_)

    '''
    Clean null values if any
    '''

    distances_matrix[np.isnan(distances_matrix)] = np.nanmean(distances_matrix)
    
    '''
    Split data
    '''
    
    y = np.array(y).reshape(len(y), 1)
    X = np.concatenate((y, distances_matrix), axis=1)
    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])
    
    '''
    Saving data
    '''

    logger.info("Saving transformed data.")
    
    s3_client = boto3.client('s3')

    # train
    filepath = f"{base_dir}/train/"
    filename = f"train"
    np.savetxt(filepath + filename + '.csv', train, delimiter=",")

    logger.info("Uploading data to bucket: %s, key: %s", bucket, filename + '.csv')
    s3_client.upload_file(filepath + filename + '.csv', bucket, filename + '.csv')
    
    # validation
    filepath = f"{base_dir}/validation/"
    filename = f"validation"
    np.savetxt(filepath + filename + '.csv', validation, delimiter=",")

    logger.info("Uploading data to bucket: %s, key: %s", bucket, filename + '.csv')
    s3_client.upload_file(filepath + filename + '.csv', bucket, filename + '.csv')
    
    # test
    filepath = f"{base_dir}/test/"
    filename = f"test"
    np.savetxt(filepath + filename + '.csv', test, delimiter=",")

    logger.info("Uploading data to bucket: %s, key: %s", bucket, filename + '.csv')
    s3_client.upload_file(filepath + filename + '.csv', bucket, filename + '.csv')

    logger.info("Data saved.")

    logger.info("End preprocessing.")
