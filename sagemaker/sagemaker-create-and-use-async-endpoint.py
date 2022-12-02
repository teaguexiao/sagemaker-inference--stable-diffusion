import sagemaker
import boto3
sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket="aws-sagemaker-stable-diffusion-2"

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")



from sagemaker.huggingface.model import HuggingFaceModel
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.s3 import s3_path_join
import os
from pathlib import Path

CURR_PATH = Path(os.path.dirname(os.path.realpath(__file__)))

with open(CURR_PATH / ".." / "terraform" / "sagemaker-role-arn.txt", "r") as f:
    sagemaker_role = f.read()
with open(CURR_PATH / ".." / "terraform" / "bucket-name.txt", "r") as f:
    bucket_name = f.read()

# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
   model_data="s3://aws-sagemaker-stable-diffusion-2/sdv1-4_model.tar.gz",
   name="huggingface-pytorch-inference-async-2",
   role=sagemaker_role,
   transformers_version="4.12",  # transformers version used
   pytorch_version="1.9",        # pytorch version used
   py_version='py38',            # python version used
)

# create async endpoint configuration
async_config = AsyncInferenceConfig(
    output_path=s3_path_join("s3://",bucket_name,"async_inference/output") , # Where our results will be stored
    # notification_config={
            #   "SuccessTopic": "arn:aws:sns:us-east-2:123456789012:MyTopic",
            #   "ErrorTopic": "arn:aws:sns:us-east-2:123456789012:MyTopic",
    # }, #  Notification configuration
)

# deploy the endpoint endpoint
async_predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge",
    endpoint_name="huggingface-pytorch-inference-async-2",
    async_inference_config=async_config
)

print(f"async_predictor object: {async_predictor}")

from sagemaker.async_inference.waiter_config import WaiterConfig

resp = async_predictor.predict_async(data={"prompt": "i like you. I love you"})

print(f"Response object: {resp}")
print(f"Response output path: {resp.output_path}")
print("Start Polling to get response:")

config = WaiterConfig(
  max_attempts=5, #  number of attempts
  delay=10 #  time in seconds to wait between attempts
  )

resp.get_result(config)
