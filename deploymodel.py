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
import time


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
        _l.debug(f"Model package info {approved_packages[0]}")
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
    _l.debug(f"Model package info: {pprint.pformat(description)}")

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
    _l.debug(f"Create model response: {response}")
    _l.info(f"Model ARN: {response.get('ModelArn')}")
    # -- END register model

    ######  with the model registered we can deploy the endpoint #####

    # Set some paths and vars needed
    sagemaker_session = sagemaker.session.Session(boto3.Session())
    bucket =  sagemaker_session.default_bucket()
    data_capture_prefix = f'{BASE_JOB_PREFIX}/datacapture'
    s3_capture_upload_path = 's3://{}/{}'.format(bucket, data_capture_prefix)
    _l.info(f"Capture path: {s3_capture_upload_path}")

    # Create an endpoint configuration by calling create_endpoint_config.
    # The endpoint configuration specifies the number and type of Amazon EC2
    # instances to use for the endpoint. It can also contain the
    # DataCaptureConfig
    endpoint_config_name = f'sts-model-EndpointConfig-{ahora}'
    sagemaker_session.create_endpoint_config(
        name=endpoint_config_name,
        model_name=model_name,
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        data_capture_config_dict=sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,  # TODO: reduce in production
            destination_s3_uri=s3_capture_upload_path
        )._to_request_dict()
    )

    # Create the endpoint using the EndPointConfig
    endpoint_name = f'sts-model-Endpoint-{ahora}'
    _l.info(f"Endpoint name: {endpoint_name}")

    sagemaker_session.create_endpoint(
        endpoint_name,
        endpoint_config_name,
        wait=True
    )

    _l.info(f"Endpoint {endpoint_name} ready")

    # create_endpoint_response = sm_client.create_endpoint(
    #     EndpointName=endpoint_name,
    #     EndpointConfigName=endpoint_config_name)
    # _l.debug(f"Create endpoint response: {create_endpoint_response}")
    # _l.info(f"EndPoint ARN: {create_endpoint_response['EndpointArn']}")

    # # wait until the endpoint comes into service
    # _l.info("Waiting for the endpoint...")
    # sagemaker_session.wait_for_endpoint(endpoint_name)
    # describe_endpoint_response = sm_client.describe_endpoint(
    #     EndpointName=endpoint_name)
    # response_status = describe_endpoint_response["EndpointStatus"]
    # _l.info(f"Endpoint status: {response_status}")
    # _l.info(
    #     f"Endpoint description: {pprint.pformat(describe_endpoint_response)}")

    # try:
    #     while response_status not in ['InService', 'Failed']:
    #         time.sleep(5) # wait for 5 sec
    #         describe_endpoint_response = sm_client.describe_endpoint(
    #             EndpointName=endpoint_name)
    #         response_status = describe_endpoint_response["EndpointStatus"]
    #         _l.info(f"Waiting for the endpoint status: {response_status}")


    #     if response_status == 'Failed':
    #         _l.error("Endpoint could not be created, updated, or re-scaled")
    #         _l.error(describe_endpoint_response['FailureReason'])
    #         raise ValueError

    #     _l.info(f"Endpoint description: {describe_endpoint_response}")
    # except ClientError as e:
    #     error_message = e.response["Error"]["Message"]
    #     _l.error(error_message)
    #     raise Exception(error_message)
    # except ValueError as ve:
    #     # Can't create the endpoint, clean up the created resources:
    #     # remove the endpoint config
    #     # remove the model
    #     raise ve
    # # --


    ### ENDPOINT deploy done



if __name__ == '__main__':
    main()
