"""Generate fake ground truth to test MQM

requires --capture-prefix with the format YYYY/MM/DD/HH for example

python gen_fake_ground_truth.py --capture-prefix '2021/02/12/13'

with corresponds to the data capture for day 12 of month 2 of year 2021 for
the 13 hour,  assuming a hourly interval.
"""
from sts.utils import load_dataset, get_sm_session
from sagemaker_containers.beta.framework import content_types, encoders
from sagemaker.s3 import S3Downloader, S3Uploader
from dotenv import load_dotenv
import os
import argparse
import json
import uuid
import random


load_dotenv()


def ground_truth_with_id(
        inference_id, predicted, labels, inference_id_prefix):
    """Given a prediction generate ground truth label"""
    # comment the next line to use the actual label from the inference
    # i am using random here to invalidate some of the values for
    # the quality monitor
    data_label = random.choice(['1', '0'])

    # check if original label and predicted are the same and label,
    # uncomment to use test dataset
    # data_label = '1.0' if label == labels[inference_id-1] else '0.0'

    return {
        "groundTruthData": {
            "data": data_label,
            "encoding": "CSV",  # only supports CSV
        }, 
        "eventMetadata": {
            "eventId": f"{inference_id_prefix}{str(inference_id)}",
        },
        "eventVersion": "0",
    }


def main(deploy_data: dict, train_data: dict, capture_prefix: str):
    inference_id_prefix = 'sts_'  # the same used in testendpoint.py

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

    # read test data
    test_data = load_dataset(
        train_data['train']['test'], 'test.csv', sagemaker_session=sm_session)
    print(f"Loadding {train_data['train']['test']}")
    Y_val = test_data.iloc[:, 0].to_numpy()
    print(f"Test dataset shape: {Y_val.shape}")

    # list capture files, this is just as an example. Not used right
    # now but could be.
    capture_files = sorted(
        S3Downloader.list(
            "{}/{}".format(
                deploy_data['monitor']['s3_capture_upload_path'],
                deploy_data['endpoint']['name']),
            sagemaker_session=sm_session)
    )
    # just the files with the prefix
    filtered = list(filter(
        lambda file_name: capture_prefix in file_name, capture_files))
    print(f"Detected {len(filtered)} capture files")

    capture_records = []
    for c_file in filtered:
        print(f"Processing: {c_file}")
        # read the capture data directly from S3
        content = S3Downloader.read_file(c_file, sagemaker_session=sm_session)
        records = [json.loads(l) for l in content.split("\n")[:-1]]

        capture_records.extend(records)

    print(f"No. of records {len(capture_records)} captured")
    captured_predictions = {}

    for obj in capture_records:
        # Extract inference ID
        inference_id = obj["eventMetadata"]["inferenceId"]
        # current version of script start in 1 when id=0
        # remove the prefix and get the id
        req_id = int(inference_id[len(inference_id_prefix):])
        
        # Extract result given by the model
        Y_pred_value = encoders.decode(
            obj["captureData"]["endpointOutput"]["data"],
            # i have fixed this value here becouse 
            # obj["captureData"]["endpointOutput"]["observedContentType"]
            # some times include the encoding like: text/csv; utf-8
            # and encoders.decode() will give error.
            content_types.CSV)
        captured_predictions[req_id] = Y_pred_value  # np.array


    # save and upload the ground truth labels
    print("Generating labels")
    fake_records = []
    for i,label in captured_predictions.items():
        val = ground_truth_with_id(i,label, Y_val, inference_id_prefix)
        fake_records.append(json.dumps(val))

    data_to_upload = "\n".join(fake_records)
    target_s3_uri = "{}/{}/{}.jsonl".format(
        deploy_data['monitor']['ground truth uri'],
        capture_prefix,
        uuid.uuid4().hex)
    print(f"Uploading ground truth to {target_s3_uri} ...", end="")
    S3Uploader.upload_string_as_file_body(
        data_to_upload, target_s3_uri, sagemaker_session=sm_session)
    print("Done !")


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
    parser.add_argument(
        "--capture-prefix", type=str, required=True, 
        help="Capture data prefix in the format YYYY/MM/DD/HH"
    )

    args, _ = parser.parse_known_args()
    print(f"Using deploy info {args.deploymodel_output}")
    print(f"Using training info {args.trainmodel_output}")
    with open(args.deploymodel_output) as f:
        deploy_data = json.load(f)

    with open(args.trainmodel_output) as f:
        train_data = json.load(f)

    main(deploy_data, train_data, args.capture_prefix)
