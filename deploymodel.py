"""Deploy register and deploy model


needs:
    MODEL_PACKAGE_GROUP_NAME
    ROLE_ARN
    AWS_DEFAULT_REGION


https://docs.amazonaws.cn/en_us/sagemaker/latest/dg/model-registry.html
"""
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.model_monitor import ModelQualityMonitor, EndpointInput
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor.data_capture_config import _MODEL_MONITOR_S3_PATH
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sts.utils import load_dataset, get_sm_session
import random
import sagemaker
import boto3
import os
import logging
import pprint
import datetime
import argparse
import tempfile
import json
import time


_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


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


def main(baseline_dataset_uri, test_set_uri):
    # ####
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
        'MODEL_PACKAGE_GROUP_NAME', 'stsPackageGroup')
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')

    # get the last version aproved in the model package group
    model_package_arn = get_approved_package(
        MODEL_PACKAGE_GROUP_NAME, sm_client)
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
    _l.debug(f"Create model response: {response}")
    _l.info(f"Model ARN: {response.get('ModelArn')}")
    # -- END register model

    ######  with the model registered we can deploy the endpoint #####

    # Set some paths and vars needed
    bucket = sm_session.default_bucket()
    model_monitor_s3_path = 's3://{}/{}/model-monitor'.format(
        bucket, BASE_JOB_PREFIX)
    data_capture_output_s3_path = '{}/data-capture'.format(
        model_monitor_s3_path)
    _l.info(f"Capture path: {data_capture_output_s3_path}")

    # Create an endpoint configuration by calling create_endpoint_config.
    # The endpoint configuration specifies the number and type of Amazon EC2
    # instances to use for the endpoint. It can also contain the
    # DataCaptureConfig
    endpoint_config_name = f'sts-model-EndpointConfig-{ahora}'
    sm_session.create_endpoint_config(
        name=endpoint_config_name,
        model_name=model_name,
        initial_instance_count=1,
        instance_type='ml.m5.xlarge',
        data_capture_config_dict=sagemaker.model_monitor.DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100,  # TODO: reduce in production
            capture_options=['REQUEST', 'RESPONSE'],
            csv_content_types=['text/csv'],
            json_content_types=None,
            destination_s3_uri=data_capture_output_s3_path,
            sagemaker_session=sm_session
        )._to_request_dict()
    )

    # Create the endpoint using the EndPointConfig
    endpoint_name = f'sts-model-Endpoint-{ahora}'
    _l.info(f"Endpoint name: {endpoint_name}")

    sm_session.create_endpoint(
        endpoint_name,
        endpoint_config_name,
        wait=True
    )

    _l.info(f"Endpoint {endpoint_name} ready")
    # ENDPOINT deploy done

    ################ Model Quality Monitor #####################
    mq_instance_count = 1
    mq_instance_type = 'ml.m5.xlarge'
    mq_instance_volume_size_in_gb = 5
    mq_max_run_time_in_seconds = 1800
    monitor_schedule_name = f"mq-mon-sch-{BASE_JOB_PREFIX}"

    # Create the Model Quality Monitor
    mq_monitor = ModelQualityMonitor(
        role=sagemaker.get_execution_role(sm_session),
        instance_count=mq_instance_count,
        instance_type=mq_instance_type,
        volume_size_in_gb=mq_instance_volume_size_in_gb,
        max_runtime_in_seconds=mq_max_run_time_in_seconds,
        base_job_name=f"mq-{BASE_JOB_PREFIX}",
        sagemaker_session=sm_session
    )

    # baseline job for the model quality monitor
    mq_baseline_job_name_prefix = 'mq-bsl-job-{}'.format(BASE_JOB_PREFIX)
    mq_baseline_job_name = '{}-{:%Y%m%d%H%M}'.format(
        mq_baseline_job_name_prefix, datetime.datetime.now())
    # REVIEW: Mirar si realmente este es el formato en que esta el el dataset
    mq_baseline_dataset_format = DatasetFormat.csv(
        header=True, output_columns_position='START')
    mq_problem_type = 'MulticlassClassification'
    mq_inference_attribute = 'prediction'
    mq_ground_truth_attribute = 'label'
    mq_baseline_job_output_s3_path = "s3://{}/{}/model-quality/baseline/".format(
        bucket, BASE_JOB_PREFIX)

    # Create the baseline job and generate the constraints
    _l.info("Generate constraints for ModelQualityMonitor")
    mq_monitor.suggest_baseline(
        job_name=mq_baseline_job_name,
        baseline_dataset=baseline_dataset_uri,
        dataset_format=mq_baseline_dataset_format,
        output_s3_uri=mq_baseline_job_output_s3_path,
        problem_type=mq_problem_type,
        inference_attribute=mq_inference_attribute,
        ground_truth_attribute=mq_ground_truth_attribute,
        wait=True,
        logs=False
    )

    mq_baseline_job = mq_monitor.latest_baselining_job

    # Print the statistics
    _l.debug('Model Quality statistics:')
    _l.debug(
        mq_baseline_job.baseline_statistics(
        ).body_dict['multiclass_classification_metrics'])

    # Print the constraints
    _l.debug('Model Quality constraints:')
    _l.debug(
        mq_baseline_job.suggested_constraints(
        ).body_dict['multiclass_classification_constraints'])

    # Ingest ground truth labels and merge them with predictions

    inference_id_prefix = 'sts_'
    # REVIEW:
    # Randomly set y column's value as 1,000,000 for 20% of the time.
    # This will result in violations on Model Quality that you will
    # observe when monitoring completes.

    def generate_synthetic_ground_truth(inference_id_suffix, y_test_value):
        random.seed(inference_id_suffix)
        rand = random.random()
        return {
            'groundTruthData': {
                'data': '1000000' if rand < 0.2 else y_test_value,
                'encoding': 'CSV'
            },
            'eventMetadata': {
                'eventId': '{}{}'.format(inference_id_prefix, inference_id_suffix)
            },
            'eventVersion': '0'
        }

    # Iterate over the y_test dataset
    synthetic_ground_truth_list = []
    # REVIEW: Supongo esta es la forma de seguir aqui
    test_df = load_dataset(test_set_uri, 'test.csv')
    y_test = test_df[[0]]
    y_test_rows = y_test.values.tolist()
    for index, y_test_row in enumerate(y_test_rows, start=1):
        synthetic_ground_truth_list.append(
            json.dumps(
                generate_synthetic_ground_truth(index, str(y_test_row[0]))))

    ground_truth_dir_s3_prefix = '{}/data/ground-truth'.format(
        BASE_JOB_PREFIX)
    synthetic_ground_truth_s3_path_suffix = datetime.datetime.now().strftime(
        '/%Y/%m/%d/%H')
    with tempfile.TemporaryDirectory() as ground_truth_dir:
        # Write the synthetic ground truth file to the local directory
        synthetic_ground_truth_file_name = 'synthetic_ground_truth.jsonl'
        synthetic_ground_truth_file_path = os.path.join(
            ground_truth_dir, synthetic_ground_truth_file_name)
        with open(synthetic_ground_truth_file_path, 'wt') as synthetic_ground_truth_file:
            synthetic_ground_truth_file.write(
                '\n'.join(synthetic_ground_truth_list))

        # Upload the synthetic ground truth file to S3
        sm_session.upload_data(
            path=synthetic_ground_truth_file_path,
            key_prefix="{}{}".format(
                ground_truth_dir_s3_prefix,
                synthetic_ground_truth_s3_path_suffix
            ),
            bucket=bucket)

    synthetic_ground_truth_s3_path_prefix = 's3://{}/{}'.format(
        bucket, ground_truth_dir_s3_prefix)

    # Model Quality schedule
    mq_monitor_schedule_endpoint_input = EndpointInput(
        endpoint_name=endpoint_name,
        destination='/opt/ml/processing/mq_monitor/input_data',
        inference_attribute='0',
        start_time_offset='-PT1H',
        end_time_offset='-PT0H')

    mq_mon_schedule_output_s3_path = '{}/model-quality/monitoring'.format(
        model_monitor_s3_path
    )
    mq_monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_schedule_name,
        endpoint_input=mq_monitor_schedule_endpoint_input,
        ground_truth_input=synthetic_ground_truth_s3_path_prefix,
        problem_type=mq_problem_type,
        output_s3_uri=mq_mon_schedule_output_s3_path,
        constraints=mq_monitor.suggested_constraints(),
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True
    )

    # Model Quality details
    # Describe and print status
    mq_monitor_schedule_details = mq_monitor.describe_schedule()
    while mq_monitor_schedule_details['MonitoringScheduleStatus'] == 'Pending':
        _l.info(f'Waiting for {monitor_schedule_name}')
        time.sleep(3)
        mq_monitor_schedule_details = mq_monitor.describe_schedule()
    _l.info(
        f"Model Quality Monitor - schedule details: {pprint.pformat(mq_monitor_schedule_details)}")
    _l.info(
        f"Model Quality Monitor - schedule status: {mq_monitor_schedule_details['MonitoringScheduleStatus']}")
    # END: Model Quality Monitor
    # --


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-set-uri", type=str, required=True)
    parser.add_argument("--test-set-uri", type=str, required=True)

    args, _ = parser.parse_known_args()
    _l.info(f"Using baseline dataset {args.baseline_set_uri}")
    main(args.baseline_set_uri, args.test_set_uri)
