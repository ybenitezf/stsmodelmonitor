"""Execute a sagemaker pipeline

This will use the fallowing cofigs from enviroment variables:

- AWS_DEFAULT_REGION
- ROLE_ARN
- PIPELINE_NAME
- MODEL_PACKAGE_GROUP_NAME
- BASE_JOB_PREFIX
"""
from sts.pipeline import get_pipeline
from dotenv import load_dotenv
import sagemaker
import pprint
import logging
import json
import os

_l = logging.getLogger()
_l.addHandler(logging.StreamHandler())
_l.setLevel(logging.INFO)
load_dotenv()


def main():
    # define some configurations from env

    # AWS especific
    AWS_DEFAULT_REGION = os.getenv('AWS_DEFAULT_REGION', 'eu-west-1')
    ROLE_ARN = os.getenv('AWS_ROLE', sagemaker.get_execution_role())

    # MLOps especific
    PIPELINE_NAME = os.getenv('PIPELINE_NAME', 'stsPipeline')
    MODEL_PACKAGE_GROUP_NAME = os.getenv(
        'MODEL_PACKAGE_GROUP_NAME', 'stsPackageGroup')
    BASE_JOB_PREFIX = os.getenv('BASE_JOB_PREFIX', 'sts')

    try:
        # define the ml pipeline for training
        pipe = get_pipeline(
            region=AWS_DEFAULT_REGION,
            role=ROLE_ARN,
            pipeline_name=PIPELINE_NAME,
            model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            base_job_prefix=BASE_JOB_PREFIX)


        # output debug information
        parsed = json.loads(pipe.definition())
        _l.info('ML Pipeline definition')
        _l.info(json.dumps(parsed, indent=2, sort_keys=True))

        # Created/Updated SageMaker Pipeline
        upsert_response = pipe.upsert(role_arn=ROLE_ARN)
        _l.info(f"C/U SageMaker Pipeline: Response received: {upsert_response}")


        _l.info("Starting the SageMaker pipeline")
        execution = pipe.start()
        _l.info("Waiting for the pipeline")
        execution.wait()


        _l.info(
            f"Pipeline finished: {pprint.pformat(execution.list_steps())}")
    except Exception as e:
        _l.exception(f"Exception: {e}")


if __name__=="__main__":
    main()
