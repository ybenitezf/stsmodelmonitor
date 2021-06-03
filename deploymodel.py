"""Deploy register and deploy model


config by environment variable:

- MODEL_PACKAGE_GROUP_NAME
- ROLE_ARN
- AWS_DEFAULT_REGION

Will search for the lastest approved package on MODEL_PACKAGE_GROUP_NAME
and deploy an endpoint.

If --capture is passed the endpoint will have datacapture enabled

https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/model-registry.html
"""
from botocore.exceptions import ClientError
from sagemaker.deserializers import CSVDeserializer
from sagemaker.model_monitor import DataCaptureConfig
from sagemaker.serializers import CSVSerializer
from sagemaker.sklearn.model import SKLearnModel
from dotenv import load_dotenv
from sts.utils import get_sm_session
import sagemaker
import os
import logging
import datetime
import json
import argparse


_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def json_default(o):
    """Dump  datatime to isoformat in json"""
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


def get_approved_package(model_package_group_name, sm_client):
    """Gets the latest approved model package for a model package group.

    Args:
        model_package_group_name: The model package group name.

    Returns:
        The SageMaker Model Package ARN.
    """
    try:
        # Get the latest approved model package
        response = sm_client.list_model_packages(
            ModelPackageGroupName=model_package_group_name,
            ModelApprovalStatus="Approved",
            SortBy="CreationTime",
            MaxResults=100,
        )
        approved_packages = response["ModelPackageSummaryList"]

        # Fetch more packages if none returned with continuation token
        while len(approved_packages) == 0 and "NextToken" in response:
            _l.debug("Getting more packages for token: {}".format(
                response["NextToken"]))
            response = sm_client.list_model_packages(
                ModelPackageGroupName=model_package_group_name,
                ModelApprovalStatus="Approved",
                SortBy="CreationTime",
                MaxResults=100,
                NextToken=response["NextToken"],
            )
            approved_packages.extend(response["ModelPackageSummaryList"])

        # Return error if no packages found
        if len(approved_packages) == 0:
            error_message = (
                f"No approved ModelPackage found for ModelPackageGroup: {model_package_group_name}"
            )
            _l.error(error_message)
            raise Exception(error_message)

        # Return the pmodel package arn
        model_package_arn = approved_packages[0]["ModelPackageArn"]
        _l.info(
            f"Identified the latest approved model package: {model_package_arn}")
        _l.debug(f"Model package info {approved_packages[0]}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        _l.error(error_message)
        raise Exception(error_message)


def main(datacapture=False):
    # Load config from environment and set required defaults
    # AWS especific
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', 'default')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    b3_session, sm_client, sm_runtime, sm_session = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())

    MODEL_PACKAGE_GROUP_NAME = os.getenv(
        'MODEL_PACKAGE_GROUP_NAME', 'sts-sklearn-grp')
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')

    # define useful const's
    bucket = sm_session.default_bucket()
    endpoint_name = "{}-sklearn-{}".format(
        BASE_JOB_PREFIX,
        datetime.datetime.now().strftime("%Y%m%d%H%M")
    )
    prefix = "{}/{}".format(BASE_JOB_PREFIX, endpoint_name)
    data_capture_prefix = "{}/datacapture".format(prefix)
    s3_capture_upload_path = "s3://{}/{}".format(bucket, data_capture_prefix)
    # outputs is a dict to save to json
    outputs = dict()

    if datacapture is True:
        # if data capture was enabled output the S3 Uri for data capture
        outputs['monitor'] = {
            's3_capture_upload_path': s3_capture_upload_path
        }

    # get the last version aproved in the model package group
    model_package_arn = get_approved_package(
        MODEL_PACKAGE_GROUP_NAME, sm_client)
    _l.info(f"Latest approved model package: {model_package_arn}")
    model_info = sm_client.describe_model_package(
        ModelPackageName=model_package_arn)
    outputs['model_info'] = model_info
    model_uri = model_info.get(
        'InferenceSpecification')['Containers'][0]['ModelDataUrl']
    _l.info(f"Model data uri: {model_uri}")

    sk_model = SKLearnModel(
        model_uri,  # s3 uri for the model.tar.gz
        ROLE_ARN,   # sagemaker role to be used
        'model_loader.py',  # script to load the model
        framework_version='0.23-1'
    )

    data_capture_config=None
    if datacapture is True:
        # if data capture was enabled generated the required config
        _l.info("Enabling data capture as requested")
        _l.info(f"s3_capture_upload_path: {s3_capture_upload_path}")
        data_capture_config = DataCaptureConfig(
            enable_capture=True, sampling_percentage=100, 
            destination_s3_uri=s3_capture_upload_path,
            capture_options=["REQUEST", "RESPONSE"],
            sagemaker_session=sm_session
        )

    # Deploy the endpoint
    predictor = sk_model.deploy(
        instance_type="ml.m5.xlarge", 
        initial_instance_count=1,
        serializer=CSVSerializer(),
        deserializer=CSVDeserializer(),
        data_capture_config=data_capture_config,
        endpoint_name=endpoint_name
    )

    _l.info(f"Endpoint name: {predictor.endpoint_name}")
    outputs['endpoint'] = {
        'name': endpoint_name,
        'config_name': predictor.endpoint_name # is the same as the endpoint ?
    }
    outputs['model_info'].update({"name": sk_model.name})
    # ENDPOINT deploy done

    # save useful outputs to a file
    with open('deploymodel_out.json', 'w') as f:
        json.dump(outputs, f, default=json_default)
    # --


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--capture',
        action='store_true',
        help="Enable data capture in the endpoint")
    args, _ = parser.parse_known_args()

    main(datacapture=args.capture)
