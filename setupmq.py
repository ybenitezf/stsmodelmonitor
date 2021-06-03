"""Setup the schedule model quality monitor

Assumes the shelude is called "mq-mon-sch-sts"
"""
from sagemaker.model_monitor import ModelQualityMonitor
from sagemaker.model_monitor import EndpointInput
from sagemaker.model_monitor import CronExpressionGenerator
from sagemaker.model_monitor.dataset_format import DatasetFormat
import sagemaker

from dotenv import load_dotenv
from sts.utils import get_sm_session
import os
import pprint
import json
import argparse
import botocore
import logging
import datetime
import time

load_dotenv()


_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def show_schedule(name, client):
    """Wait until the schedule is deleted"""
    try:
        d = client.describe_monitoring_schedule(
            MonitoringScheduleName=name
        )
        print(pprint.pformat(d))
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFound':
            # ok, eso no existe
            print("No hay información")
            return
        else:
            raise e


def json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()


def main(resources, train_data):

    # configurarion
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
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')
    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())
    outputs = resources

    bucket = sm_session.default_bucket()
    prefix = "{}/{}".format(
        BASE_JOB_PREFIX,
        resources['endpoint']['name']
    )
    if 'monitor' not in resources:
        raise ValueError("Monitoring not enabled")

    if 's3_capture_upload_path' not in resources['monitor']:
        raise ValueError("Monitoring not enabled")
    
    baseline_prefix = prefix + "/baselining"
    baseline_data_prefix = baseline_prefix + "/data"
    baseline_results_prefix = baseline_prefix + "/results"
    baseline_data_uri = "s3://{}/{}".format(bucket, baseline_data_prefix)
    baseline_results_uri = "s3://{}/{}".format(bucket, baseline_results_prefix)
    outputs['monitor'].update({
        'baseline': {
            'data_uri': baseline_data_uri,
            'results_uri': baseline_results_uri
        }
    })
    _l.info("Baseline data uri: {}".format(baseline_data_uri))
    _l.info("Baseline results uri: {}".format(baseline_results_uri))

    ground_truth_upload_path = f"s3://{bucket}/{prefix}/ground_truth_data"
    _l.info(f"Ground truth uri: {ground_truth_upload_path}")
    outputs['monitor'].update({'ground truth uri': ground_truth_upload_path})

    # Create a baselining job with training dataset
    _l.info("Executing a baselining job with training dataset")
    _l.info(f"baseline_data_uri: {train_data['baseline']['validate']}")
    my_monitor = ModelQualityMonitor(
        role=ROLE_ARN, 
        sagemaker_session=sm_session,
        max_runtime_in_seconds=1800  # 30 minutes
    )
    my_monitor.suggest_baseline(
        baseline_dataset=train_data['baseline']['validate'] + "/baseline.csv",
        dataset_format=DatasetFormat.csv(header=True),
        problem_type="Regression",
        inference_attribute="prediction", ground_truth_attribute="label",
        output_s3_uri=baseline_results_uri,
        wait=True
    )
    baseline_job = my_monitor.latest_baselining_job
    _l.info("suggested baseline contrains")
    _l.info(
        pprint.pformat(
            baseline_job.suggested_constraints().body_dict[
                "regression_constraints"]
            )
    )
    _l.info("suggested baseline statistics")
    _l.info(
        pprint.pformat(
            baseline_job.baseline_statistics().body_dict[
                "regression_metrics"]
        )
    )

    monitor_schedule_name = (
        f"{BASE_JOB_PREFIX}-mq-sch-{datetime.datetime.utcnow():%Y-%m-%d-%H%M}"
    )
    _l.info(f"Monitoring schedule name: {monitor_schedule_name}")
    outputs['monitor'].update({
        'schedule_name': monitor_schedule_name})
    endpointInput = EndpointInput(
        resources['endpoint']['name'], 
        "/opt/ml/processing/input_data",
        inference_attribute='0'  # REVIEW:
    )

    my_monitor.create_monitoring_schedule(
        monitor_schedule_name=monitor_schedule_name,
        endpoint_input=endpointInput,
        output_s3_uri=baseline_results_uri,
        problem_type="Regression",
        ground_truth_input=ground_truth_upload_path,
        constraints=baseline_job.suggested_constraints(),
        # run the scheduler hourly
        schedule_cron_expression=CronExpressionGenerator.hourly(),
        enable_cloudwatch_metrics=True,
    )
    mq_schedule_details = my_monitor.describe_schedule()
    while mq_schedule_details['MonitoringScheduleStatus'] == 'Pending':
        _l.info(f'Waiting for {monitor_schedule_name}')
        time.sleep(3)
        mq_schedule_details = my_monitor.describe_schedule()
    _l.debug(
        f"Model Quality Monitor - schedule details: {pprint.pformat(mq_schedule_details)}")
    _l.info(
        f"Model Quality Monitor - schedule status: {mq_schedule_details['MonitoringScheduleStatus']}")

    # save outputs to a file
    with open('deploymodel_out.json', 'w') as f:
        json.dump(outputs, f, default=json_default)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploymodel-output", type=str, required=False, 
        default='deploymodel_out.json',
        help="JSON output from the deploy script"
    )
    parser.add_argument(
        "--trainmodel-output", type=str, required=False, 
        default='trainmodel_out.json',
        help="JSON output from the train script"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    with open(args.deploymodel_output) as f:
        data = json.load(f)
    _l.info(f"Using training info {args.trainmodel_output}")
    with open(args.trainmodel_output) as f:
        train_data = json.load(f)
    main(data, train_data)
