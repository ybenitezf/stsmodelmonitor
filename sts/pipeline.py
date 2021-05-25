"""Example workflow pipeline script for sts pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.sklearn import SKLearn

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )


def get_pipeline(
    region,
    sagemaker_session,
    role=None,
    default_bucket=None,
    model_package_group_name="stsPackageGroup",
    pipeline_name="stsPipeline",
    base_job_prefix="sts",
) -> Pipeline:
    """Gets a SageMaker ML Pipeline instance working with on sts data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """

    """
        Instance types allowed:
        
        ml.r5.12xlarge, ml.m5.4xlarge, ml.p2.xlarge, ml.m4.16xlarge, ml.r5.24xlarge, 
        ml.t3.xlarge, ml.r5.16xlarge, ml.m5.large, ml.p3.16xlarge, ml.p2.16xlarge, 
        ml.c4.2xlarge, ml.c5.2xlarge, ml.c4.4xlarge, ml.c5.4xlarge, ml.c4.8xlarge, 
        ml.c5.9xlarge, ml.c5.xlarge, ml.c4.xlarge, ml.t3.2xlarge, ml.t3.medium, 
        ml.c5.18xlarge, ml.r5.2xlarge, ml.p3.2xlarge, ml.m5.xlarge, ml.m4.10xlarge, 
        ml.r5.4xlarge, ml.m5.12xlarge, ml.m4.xlarge, ml.t3.large, ml.m5.24xlarge, 
        ml.m4.2xlarge, ml.m5.2xlarge, ml.p2.8xlarge, ml.r5.8xlarge, ml.r5.xlarge, 
        ml.r5.large, ml.p3.8xlarge, ml.m4.4xlarge

        see
        https://aws.amazon.com/blogs/machine-learning/right-sizing-resources-and-avoiding-unnecessary-costs-in-amazon-sagemaker/
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    # parameters for pipeline execution
    processing_instance_count = ParameterInteger(
        name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(
        name="ProcessingInstanceType", default_value="ml.m5.xlarge"
    )

    # as of free tier of 50 hours of m4.xlarge or m5.xlarge instances
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.m5.xlarge"
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="Approved"
    )

    # preprocess 

    # preprocess input data
    input_data = ParameterString(
        name="InputDataUrl",
        default_value=f"s3://sts-datwit-dataset/stsmsrpc.txt",
    )

    # processing step for feature engineering
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        base_job_name=f"{base_job_prefix}/sklearn-sts-preprocess",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_preprocess = ProcessingStep(
        name="PreprocessSTSData",
        processor=sklearn_processor,
        outputs=[
            ProcessingOutput(output_name="train",
                            source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="validation",
                            source="/opt/ml/processing/validation"),
            ProcessingOutput(output_name="test",
                            source="/opt/ml/processing/test"),
        ],
        code=os.path.join(BASE_DIR, "preprocess.py"),
        job_arguments=["--input-data", input_data],
    )

    # training step for generating model artifacts
    model_path = f"s3://{sagemaker_session.default_bucket()}/{base_job_prefix}/stsTrain"
    image_uri = sagemaker.image_uris.retrieve(
        framework="sklearn",
        region=region,
        version="0.23-1",
        py_version="py3",
        instance_type=training_instance_type,
    )

    sklearn_estimator = SKLearn(
        entry_point='training.py',
        source_dir=BASE_DIR,
        instance_type=training_instance_type,
        instance_count=1,
        output_path=model_path,
        framework_version="0.23-1",
        py_version="py3",
        base_job_name=f"{base_job_prefix}/sts-train",
        sagemaker_session=sagemaker_session,
        role=role,)

    step_train = TrainingStep(
        name="TrainSTSModel",
        estimator=sklearn_estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "train"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # processing step for evaluation
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/script-sts-eval",
        sagemaker_session=sagemaker_session,
        role=role,
    )
    evaluation_report = PropertyFile(
        name="stsEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateSTSModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "test"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation",
                            source="/opt/ml/processing/evaluation"),
        ],
        code=os.path.join(BASE_DIR, "evaluate.py"),
        property_files=[evaluation_report],
    )

        # setup model quality monitoring baseline data
    script_process_baseline_data = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name=f"{base_job_prefix}/baseline",
        sagemaker_session=sagemaker_session,
        role=role,
    )

    step_proccess_baseline_data = ProcessingStep(
        name="SetupMonitoringData",
        processor=script_process_baseline_data,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_preprocess.properties.ProcessingOutputConfig.Outputs[
                    "validation"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/validation",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="validate",
                             source="/opt/ml/processing/validate"),
        ],
        code=os.path.join(BASE_DIR, "baseline.py")
    )
    # ---

    # register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )

    step_register = RegisterModel(
        name="RegisterSTSModel",
        estimator=sklearn_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )

    # condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step=step_eval,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value"
        ),
        right=6.0,
    )
    step_cond = ConditionStep(
        name="CheckMSESTSEvaluation",
        conditions=[cond_lte],
        if_steps=[step_register, step_proccess_baseline_data],
        # if_steps=[step_register],
        else_steps=[],
    )

    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            model_approval_status,
            input_data,
        ],
        steps=[step_preprocess, step_train, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
