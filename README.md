# stsmodelmonitor


Example of ML pipeline with model monitor in AWS SagemMaker

## Configurations

The configuration can be done by environment variables:

- `AWS_ROLE`: the ARN of the role to be used in the sagemaker jobs
- `AWS_DEFAULT_REGION`: defaults to `eu-west-1`
- `PIPELINE_NAME`: the name of the SM pipeline, defaults to `stsPipeline`
- `MODEL_PACKAGE_GROUP_NAME`: Model package group name for registering the model, defaults to `stsPackageGroup`
- `BASE_JOB_PREFIX`: used as a prefix for varius resources, like job names and S3 buckets keys, defaults to `sts`

This will use the defaul sagemaker bucket if not exits, a default bucket will be created based on the following format: `sagemaker-{region}-{aws-account-id}`.

You can create a `.env` file with the variables values (Don't include in the VCS repository):

```txt
AWS_DEFAULT_REGION=DDDDDDDDDDDDDDD
AWS_PROFILE=DDDDDDDDDDDDD
AWS_ROLE=DDDDDDDDDDDDDDDDDDDDDDDDD
```

## Permissions on AWS

For local exceution you must configure the aws credentials, additionally the 
user must have the permissition to assume a role that should have the following permissions,

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
