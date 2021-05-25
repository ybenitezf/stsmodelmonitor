"""Run to stop the schedule model quality monitor

Assumes the shelude is called "mq-mon-sch-sts"
"""
from dotenv import load_dotenv
from sts.utils import get_sm_session
import os
import pprint
import json
import argparse
import botocore
import time

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


def main(resources):

    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    AWS_PROFILE = os.getenv('AWS_PROFILE', 'default')
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    _, sm_client, _, _ = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    # remove resourses created by deploymodel.py
    print("Model Quality Schedule")
    show_schedule(
        resources['endpoint']['monitor_schedule_name'],
        sm_client)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deploymodel-output", type=str, required=False, 
        default='deploymodel_out.json',
        help="JSON output from the deploy script"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    with open(args.deploymodel_output) as f:
        data = json.load(f)
    main(data)
