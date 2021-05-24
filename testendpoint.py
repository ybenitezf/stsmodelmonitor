from sts.utils import load_dataset, get_sm_session
from dotenv import load_dotenv
import os
import argparse
import json
import progressbar


load_dotenv()


def main(deploy_data, train_data):
    inference_id_prefix = 'sts_'  # Comes from deploymodel.py
    outputs = {'inferences': []}

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
        train_data['train']['test'], 'test.csv', sagemaker_session=sm_session)
    print(f"Loadding {train_data['train']['test']}")
    print("Test dataset head:")
    print(test_data.head())

    # remove labels in the test dataset
    test_data.drop(test_data.columns[0], axis=1, inplace=True)

    # Iterate over the test data and call the endpoint for each row, 
    # stop for 2 seconds for rows divisible by 3, just to make time
    x_test_rows = test_data.values.tolist()
    print(
        f"Sending trafic to the endpoint: {deploy_data['endpoint']['name']}")
    with progressbar.ProgressBar(max_value=len(x_test_rows)) as bar:
        for index, x_test_row in enumerate(x_test_rows, start=1):
            x_test_row_string = ','.join(map(str, x_test_row))
            # Auto-generate an inference-id to track the request/response 
            # in the captured data
            inference_id = '{}{}'.format(inference_id_prefix, index)

            response = sm_runtime.invoke_endpoint(
                EndpointName=deploy_data['endpoint']['name'],
                ContentType='text/csv',
                Body=x_test_row_string,
                InferenceId=inference_id
            )
            result = json.loads(response['Body'].read().decode())
            outputs['inferences'].append(
                {
                    inference_id: {
                        'input': x_test_row,
                        'result': result
                    }
                }
            )

            # show progress
            bar.update(index)
    
    with open('testendpoint_out.json', 'w') as f:
        json.dump(outputs, f)


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
    print(f"Using training info {args.trainmodel_output}")
    with open(args.deploymodel_output) as f:
        deploy_data = json.load(f)

    with open(args.trainmodel_output) as f:
        train_data = json.load(f)

    main(deploy_data, train_data)
