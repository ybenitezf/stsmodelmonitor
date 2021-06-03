"""Execute a sagemaker pipeline

Will execute the ML pipeline defined in sts/pipeline.py

This will use the fallowing cofigs from enviroment variables:

- AWS_DEFAULT_REGION
- ROLE_ARN
- PIPELINE_NAME
- MODEL_PACKAGE_GROUP_NAME
- BASE_JOB_PREFIX
"""
from typing import List
from sts.pipeline import get_pipeline
from sts.utils import get_sm_session
from dotenv import load_dotenv
import sagemaker
import boto3
import pprint
import logging
import json
import os

_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def extract_step_from_list(steps, step_to_extract) -> dict:
    """extract the step definition from a list of pipeline steps"""
    for step in steps:
        if step.get('Name') in step_to_extract:
            return step


def get_outputs(step: dict) -> dict:
    """Returns the S3 uri outputs if present

    step: step definition in the pipeline
    """
    response = {}
    try:
        outputs = step.get(
            'Arguments')['ProcessingOutputConfig']['Outputs']
        for o in outputs:
            s3_out = o.get('S3Output').get('S3Uri')
            response[o.get('OutputName')] = s3_out
    except Exception as e:
        _l.debug(f"Error geting the outputs of {step.get('Name')}")

    return response


def main():
    # define some configurations from env

    # AWS especific
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', None)
    AWS_PROFILE = os.getenv('AWS_PROFILE', None)
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID', None)
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', None)
    b3_session, sm_client, sm_runtime, sm_session = get_sm_session(
        region=AWS_DEFAULT_REGION,
        profile_name=AWS_PROFILE,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())

    # MLOps especific
    PIPELINE_NAME = os.getenv('PIPELINE_NAME', 'stsPipeline')
    MODEL_PACKAGE_GROUP_NAME = os.getenv(
        'MODEL_PACKAGE_GROUP_NAME', 'sts-sklearn-grp')
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')

    outputs = {
        'pipeline': None,
        'baseline': None,
        'train': None
    }

    try:
        # define the ml pipeline for training
        pipe = get_pipeline(
            AWS_DEFAULT_REGION,
            sm_session,
            role=ROLE_ARN,
            pipeline_name=PIPELINE_NAME,
            model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            base_job_prefix=BASE_JOB_PREFIX)

        # output debug information
        parsed = json.loads(pipe.definition())
        outputs['pipeline'] = parsed
        _l.debug('ML Pipeline definition')
        _l.debug(json.dumps(parsed, indent=2, sort_keys=True))

        # Created/Updated SageMaker Pipeline
        upsert_response = pipe.upsert(role_arn=ROLE_ARN)
        _l.debug(
            f"C/U SageMaker Pipeline: Response received: {upsert_response}")

        _l.info("Starting the SageMaker pipeline")
        execution = pipe.start()
        _l.info("Waiting for the pipeline")
        execution.wait()

        _l.info("Pipeline finished: !!!")
        _l.debug(f"{pprint.pformat(execution.list_steps())}")

        # Take the s3 uri of the baseline datatase baseline.csv
        mse_step = extract_step_from_list(
            parsed.get('Steps'), 'CheckMSESTSEvaluation')
        mon_step = extract_step_from_list(
            mse_step.get('Arguments').get('IfSteps'),
            'SetupMonitoringData'
        )

        outputs['baseline'] = get_outputs(mon_step)
        # take de s3 uri of train, validate, and test datasets
        train_step_def = extract_step_from_list(
            parsed.get('Steps'), 'PreprocessSTSData')
        outputs['train'] = get_outputs(train_step_def)
        # --

        # whrite the pipeline def and the selected outputs to a json
        # file
        with open('trainmodel_out.json', 'w') as f:
            json.dump(outputs, f)
        # ---
    except Exception as e:
        _l.exception(f"Exception: {e}")


if __name__ == "__main__":
    main()
