# stsmodelmonitor

Example of ML pipeline with model monitor in AWS SagemMaker for STS

## Configurations

The configuration can be done by environment variables:

- `PIPELINE_NAME`: the name of the SM pipeline, defaults to `stsPipeline`
- `MODEL_PACKAGE_GROUP_NAME`: Model package group name for registering the model, defaults to `stsPackageGroup`
- `BASE_JOB_PREFIX`: used as a prefix for varius resources, like job names and S3 buckets keys, defaults to `sts`

This will use the defaul sagemaker bucket if not exits, a default bucket will be created based on the following format: `sagemaker-{region}-{aws-account-id}`.

You can create a `.env` file with the variables values (Don't include in the VCS repository):

```txt
PIPELINE_NAME=DDDDDDDDDDDDDDD
MODEL_PACKAGE_GROUP_NAME=DDDDDDDDDDDDD
BASE_JOB_PREFIX=DDDDDDDDDDDDDDDDDDDDDDDDD
```

## How to use

Example run from 0 to model quality monitor:

1. Create an virtualenv

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

2. Install requirements
  
   ```bash
   pip install -U pip
   pip install -r dev-requirements.txt
   ```

3. Train the model, this will define the ML pipeline and execute it on AWS Sagemaker, after the model is trained it will be added to the model group specified by `MODEL_PACKAGE_GROUP_NAME`.

   ```bash
   python trainmodel.py
   ```

   Opcional: inspect the `trainmodel_out.json` file.

4. After training the model tou can deploy an AWS Sagemaker Endpoint:

   ```bash
   python deploymodel.py --capture
   ```

   The `--capture` flag is opcional, if not provided the endpoint will not have data capture configured and the model quality monitor will not run.

   Opcional: inspect the `deploymodel_out.json` file.

5. Setup model quality monitor (MQM):

   ```bash
   python setupmq.py
   ```

   By default it will use the JSON files from the previus steps as input, and will store additional information to the `deploymodel_out.json` file.

6. Send traffic to the endpoint:

   ```bash
   python testendpoint.py
   ```

   Opcional: inspect the `testendpoint_out.json` file.

   Wait for 3 to 5 minutes until all the rows in the `test.csv` data set (838) appear in the data capture path in S3 (see `s3_capture_upload_path` in the `deploymodel_out.json` file for the S3 Uri) should be several files there.

7. Generate fake ground truth labels for the MQM:

   ```bash
   python gen_fake_ground_truth.py
   ```

   This script will read the data captures from the S3 Uri and generate labels for each inferenceId present in the captures.

8. Go to the `Sagemaker Studio`/`Sagemaker Components and registries` and select `Enpoints`, select the endpoint created by `deploymodel.py` (should by something like `sts-sklearn-YYYYMMDDHHMM`). Under _Monitoring job history_ wait until the schedule MQM run (should be more or less than 1 hour) and under _Model quality_ you will se the metrics and chars once the first job run.

9. When done cleanup:

   ```bash
   python cleanup.py
   ```

## Structure

- `example_data`: some examples of pipeline definitions, as a form of documentation
- `sts`: main py package
  - `baseline.py`: a processing script that generates a baseline dataset for the model quality monitor.
  - `evaluate.py`: using test.csv dataset evaluates the model metrics for Model registration on AWS
  - `pipeline.py`: defines the ML  pipeline for sagemaker
  - `preprocess.py`: a processing script for the sts dataset (`s3://sts-datwit-dataset/stsmsrpc.txt`)
  - `utils.py`: define some usefull functions
- `trainmodel.py`: sends to AWS SageMaker the ML pipeline definition and wait for the training to be done. It will output some information to the file `trainmodel_out.json`
- `deploymodel.py`: deploys the latest version of the model if any and optionally setup data capture on the endpoint. It will output some information to the file `deploymodel_out.json`.
- `setupmq.py`: example setup of model quality monitor for the endpoint deployed in `deploymodel.py`, this require the files `trainmodel_out.json` and `deploymodel_out.json`. It will add information to `deploymodel_out.json`.
- `gen_fake_ground_truth.py`: generate fake ground truth for the model quality monitor.
- `cleanup.py`: will remove the schedule model quality monitor, endpoint config, model endpoint and the model from the sagemaker registries.
- `testendpoint.py`: will call the model endpoint passing to it the `test.csv` dataset, it will ouput the inferences to the file `testendpoint_out.json`

## Security

This section is for admins or infraestructure managers, this example need's
the credentials configured for aws cli or the equivalent environment variables.

Anyway, the user should be able tu use/assume a rol with full sagemaker access

### Permissions on AWS

For local exceution you must configure the aws credentials, additionally the user must have the permissition to assume a role that should have the following permissions,

- Full access to the S3 bucket that will be used to store training and output data.
- Full access to launch training instances.
- Full access to deploy models.
- Full access to launch monitoring instances and schedules.
- Access to write to CloudWatch logs and metrics.

See [Using an IAM role in the AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-role.html) to configure this for the local enviroment and aws cli

### Permissions for assuming a role

In order to assume a role, the IAM user for the static credentials must have the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Action": [
                "sts:AssumeRole",
                "sts:TagSession"
            ],
            "Resource": "<SAGEMAKER_EXECUTION_ROLE_ARN>",
            "Effect": "Allow"
        }
    ]
}
```

The role's trust policy must allow the IAM user to assume the role:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "AllowIamUserAssumeRole",
            "Effect": "Allow",
            "Action": "sts:AssumeRole",
            "Principal": {"AWS": "<MY_USER_IAM_ARN>"},
        },
        {
            "Sid": "AllowPassSessionTags",
            "Effect": "Allow",
            "Action": "sts:TagSession",
            "Principal": {"AWS": "<MY_USER_IAM_ARN>"}
        }
    ]
}
```
