from sagemaker.s3 import S3Downloader
import sagemaker
import boto3
import sagemaker.session
import tempfile
import pandas as pd
import os


def load_dataset(
    s3_uri: str, filename: str, sagemaker_session=None
) -> pd.DataFrame:
    """Load a data set from a S3 uri"""
    S3Downloader.download(
        s3_uri, tempfile.gettempdir(),
        sagemaker_session=sagemaker_session)
    dataset_filename = os.path.join(
        tempfile.gettempdir(), filename)
    return pd.read_csv(dataset_filename, header=None)


def get_sm_session(
        region=None, profile_name=None,
        aws_access_key_id=None,
        aws_secret_access_key=None):
    """Returns a tuple of boto3 session, sm client, sm runtime client, sm session"""
    b3_session = boto3.Session(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region,
        profile_name=profile_name)
    sm_client = b3_session.client('sagemaker')
    sm_runtime = b3_session.client('sagemaker-runtime')
    sm_session = sagemaker.session.Session(
        boto_session=b3_session,
        sagemaker_client=sm_client,
        sagemaker_runtime_client=sm_runtime
    )

    # boto3 session
    # sagemaker client
    # sagemaker runtime client
    # sagemaker session
    return b3_session, sm_client, sm_runtime, sm_session
