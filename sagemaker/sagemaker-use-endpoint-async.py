import base64
from pathlib import Path

from sagemaker.predictor_async import AsyncPredictor
from sagemaker.async_inference.waiter_config import WaiterConfig
from sagemaker.predictor import Predictor

import conf

import sagemaker
import boto3
from time import gmtime, strftime
from datetime import datetime

boto_session = boto3.session.Session()
sm_session = sagemaker.session.Session()
sm_client = boto_session.client("sagemaker")
sm_runtime = boto_session.client("sagemaker-runtime")
region = "us-east-1"

response = sm_runtime.invoke_endpoint_async(
    EndpointName="huggingface-pytorch-inference-async-2", 
    InputLocation="s3://aws-sagemaker-stable-diffusion-2/input/input_text01.json",
    ContentType="application/json"
)
output_location = response["OutputLocation"]
print(f"OutputLocation: {output_location}")

''''
async_predictor = AsyncPredictor(Predictor(endpoint_name=conf.RESOURCE_NAME_ASYNC))
data = {"prompt": "darth vader dancing on top of the millennium falcon"}

#predictor = Predictor(endpoint_name=conf.RESOURCE_NAME)
#res = predictor.predict_async(data=data)

res = async_predictor.predict_async(data=data)
print(f'prompt used: {res["prompt"]}')
image_bytes = base64.b64decode(res["data"])


Path("output/").mkdir(exist_ok=True)
with open("output/image.jpg", "wb") as f:
    f.write(image_bytes)


print(f"Response object: {res}")
print(f"Response output path: {res.output_path}")
print("Start Polling to get response:")

config = WaiterConfig(
  max_attempts=5, #  number of attempts
  delay=10 #  time in seconds to wait between attempts
  )

res.get_result(config)
'''