from sts.utils import load_dataset, get_sm_session
from dotenv import load_dotenv
import os
import logging
import argparse
import boto3
import json
import time

_l = logging.getLogger()
logFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
_l.addHandler(consoleHandler)
_l.setLevel(logging.INFO)
load_dotenv()


def main(endpoint_name: str, test_dataset_uri: str):
    inference_id_prefix = 'sts_'  # Comes from deploymodel.py

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

    # read test data
    test_data = load_dataset(
        test_dataset_uri, 'test.csv', sagemaker_session=sm_session)
    _l.info(f"Loadding {test_dataset_uri}")
    _l.debug(f"Test dataset head: {test_data.head()}")

    # remove labels in the test dataset
    test_data.drop(test_data.columns[0], axis=1, inplace=True)

    # Iterate over the test data and call the endpoint for each row
    x_test_rows = test_data.values.tolist()
    for index, x_test_row in enumerate(x_test_rows, start=1):
        x_test_row_string = ','.join(map(str, x_test_row))
        # Auto-generate an inference-id to track the request/response in the captured data
        inference_id = '{}{}'.format(inference_id_prefix, index)

        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=x_test_row_string,
            InferenceId=inference_id
        )
        result = json.loads(response['Body'].read().decode())
        _l.debug(f"Result: {result} for {x_test_row_string}")
        _l.info(f"% {(100*index)//len(x_test_rows)}")
        if (index % 3) == 0:
            time.sleep(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint", type=str, required=True,
        help="Endpoint name")
    parser.add_argument(
        "--test-set-uri", type=str, required=True,
        help="S3 uri for the test dataset")

    args, _ = parser.parse_known_args()
    main(args.endpoint, args.test_set_uri)
