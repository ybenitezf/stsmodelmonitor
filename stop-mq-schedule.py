"""Run to stop the schedule model quality monitor

Assumes the shelude is called "mq-mon-sch-sts"
"""
from dotenv import load_dotenv
from sts.utils import get_sm_session
import os
import pprint

load_dotenv()

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

res = sm_session.delete_monitoring_schedule("mq-mon-sch-sts")
print(pprint.pformat(res))
