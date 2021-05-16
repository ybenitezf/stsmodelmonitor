"""Deploy register and deploy model


needs:
    MODEL_PACKAGE_GROUP_NAME
    ROLE_ARN
    AWS_DEFAULT_REGION


https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/model-registry.html
"""
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import sagemaker
import boto3
import os
import logging
import pprint
import datetime


_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()

sm_client = boto3.client("sagemaker")


def get_approved_package(model_package_group_name):
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
        _l.info(f"Model package info {approved_packages[0]}")
        return model_package_arn
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        _l.error(error_message)
        raise Exception(error_message)


def main():
    # ####
    # AWS especific
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())

    MODEL_PACKAGE_GROUP_NAME = os.getenv(
        'MODEL_PACKAGE_GROUP_NAME', 'stsPackageGroup')
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')

    # get the last version aproved in the model package group
    model_package_arn = get_approved_package(MODEL_PACKAGE_GROUP_NAME)
    _l.info(f"Latest approved model package: {model_package_arn}")
    description = sm_client.describe_model_package(
        ModelPackageName=model_package_arn)
    _l.info(f"Model package info: {pprint.pformat(description)}")

    # register the model in sagemaker model registry
    ahora = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_name = f'sts-model-{ahora}'
    _l.info(f"Model name : {model_name}")
    primary_container = {
        'ModelPackageName': model_package_arn,
    }
    response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer=primary_container
    )
    _l.info(f"Model arn : {response}")
    # -- END register model

    ######  with the model registered we can deploy the endpoint #####

    # First we create a DataCaptureConfig for the endpoint
    bucket =  sagemaker.session.Session(boto3.Session()).default_bucket()
    data_capture_prefix = f'{BASE_JOB_PREFIX}/datacapture'
    s3_capture_upload_path = 's3://{}/{}'.format(bucket, data_capture_prefix)
    _l.info(f"Capture path: {s3_capture_upload_path}")
    data_capture_config = sagemaker.model_monitor.DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,  # TODO: reduce in production
        destination_s3_uri=s3_capture_upload_path
    )._to_request_dict()
    _l.info(f"DataCaptureConfig: {pprint.pformat(data_capture_config)}")


    # Create an endpoint configuration by calling create_endpoint_config.
    # The endpoint configuration specifies the number and type of Amazon EC2
    # instances to use for the endpoint. It can also contain the
    # DataCaptureConfig
    endpoint_config_name = f'sts-model-EndpointConfig-{ahora}'
    _l.info(f"Creating EndpointConfig: {endpoint_config_name}")
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'InstanceType': 'ml.m5.xlarge',
                'InitialVariantWeight': 1,
                'InitialInstanceCount': 1,
                'ModelName': model_name,
                'VariantName': 'AllTraffic'
            }
        ],
        DataCaptureConfig=data_capture_config
    )
    # the aws docs sugest to call describe_endpoint_config before asuming the
    # config comes truth, something to do with DynamoDB write cache.
    _l.info(f"""EndpointConfig: {pprint.pformat(
        sm_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
    )}""")



if __name__ == '__main__':
    main()
