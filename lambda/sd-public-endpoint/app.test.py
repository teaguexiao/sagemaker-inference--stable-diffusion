import os
from urllib.parse import unquote

import boto3
from botocore.config import Config
from chalice import Chalice, Response
from sagemaker.session import Session
from essential_generators import DocumentGenerator
from sagemaker.huggingface.model import HuggingFacePredictor
import uuid
import json
import time

ENDPOINT_NAME = "huggingface-pytorch-inference"
ENDPOINT_NAME_ASYNC = "huggingface-pytorch-inference-async"
BODY_TEMPLATE = '<html><head></head><body><img src="data:image/png;base64,IMAGE"/><br><h3>PROMPT</h3></body></html>'

config = Config(read_timeout=30, retries={"max_attempts": 0})
sagemaker_runtime_client = boto3.client("sagemaker-runtime", config=config)
sagemaker_session = Session(sagemaker_runtime_client=sagemaker_runtime_client)
predictor = HuggingFacePredictor(
    endpoint_name=ENDPOINT_NAME, sagemaker_session=sagemaker_session
)
sentence_generator = DocumentGenerator()

boto_session = boto3.session.Session()
sm_runtime = boto_session.client("sagemaker-runtime")
s3=boto3.client('s3')
#s3_resource=boto3.resource('s3')

app = Chalice(app_name="sd-public-endpoint")

def make_response(res):
    return Response(
        body=BODY_TEMPLATE.replace("IMAGE", res["data"]).replace(
            "PROMPT", res["prompt"]
        ),
        status_code=200,
        headers={"Content-Type": "text/html", "prompt": res["prompt"]},
    )

def async_response(res,prompt):
    output_location = res["OutputLocation"]
    print("output_location:", output_location)

    
    key = output_location.split("/",3)[3]
    #key1 = output_location[38:]
    #object_name = output_location.split("/",4)[4]
    
    #print(type(key))
    #key = "async-output/95c804f0-1153-4855-8601-182ab5bbaa7a.out"
    #KEY=output_location.split("/",3)[3]
    #object_name=output_location.split("/",4)[4]
    #print("objetct name:",object_name)
    #key1="async-output/b161a207-dd59-4093-8b61-c4d28da7b81a.out"
    #print(object_name)
    #time.sleep(20)
    #print(key)
    print("-------------")
    BUCKET ="aws-sagemaker-stable-diffusion-2"
    print("Bucket:", BUCKET)
    while True:
        try:
            s3_clientobj = s3.get_object(Bucket="aws-sagemaker-stable-diffusion-2", Key=key)
            break
        except:
            print("no such key yet, trying in 1s...")
            time.sleep(1)

    print("pointer1")
    #file_content = obj.get()['Body'].read().decode('utf-8')
    s3_clientdata = s3_clientobj['Body'].read().decode('utf-8')
    #async-output/a97de44c-62a1-4c28-bf6f-ed4b45ef962e.out
    
    print("key:",key)
    #print("object_name:",object_name)

    #s3_resource.meta.client.download_file('aws-sagemaker-stable-diffusion-2', "async-output/261452f4-e419-433b-ab45-fe67953cb68a.out", object_name)
    
    print("s3_clientdata:",s3_clientdata)
    print(type(s3_clientdata))
    print("change string to dictionary")
    print(json.loads(s3_clientdata))
    print("print base64 string")
    img_base64=json.loads(s3_clientdata)["data"]
    print(json.loads(s3_clientdata)["data"])
    
    print("all good")
    print("prompt:",prompt)
    
    return Response(
        body=BODY_TEMPLATE.replace("IMAGE", img_base64).replace(
            "PROMPT", prompt
        ),
        status_code=200,
        headers={"Content-Type": "text/html", "prompt": "to_be_filled"},
    )

def index():
    return "Online Stable Diffusion"

def inference(prompt=""):
    data = {"prompt": prompt} if prompt else {}

    print(f"Starting inference with {data}")
    res = predictor.predict(data=data)
    print("Inference completed")

    return res

#for async inference
def async_inference(prompt=""):
    data = {"prompt": prompt} if prompt else {}

    print(f"Starting inference with {data}")
    #res = predictor.predict(data=data)
    #
    id = uuid.uuid1()
    filename=id.hex + ".json"
    json_object = json.dumps(data)

    # Save to file.
    with open("/tmp/" + filename, 'w' ) as download:
        download.write(json_object)
    
    #upload to S3
    key = "input/"+ filename
    with open("/tmp/" + filename, "rb") as f:
        s3.upload_fileobj(f, "aws-sagemaker-stable-diffusion-2", key)
        print("File is successfully uploaded to S3 bucket.")
    
    res = sm_runtime.invoke_endpoint_async(
        EndpointName=ENDPOINT_NAME_ASYNC, 
        InputLocation="s3://aws-sagemaker-stable-diffusion-2/input/" + filename,
        ContentType="application/json"
    )

    print("Inference completed")
    return res

def inference_with_prompt(prompt):
    prompt = unquote(prompt)
    res = inference(prompt)
    return make_response(res)

def inference_without_prompt():
    res = inference(sentence_generator.sentence())
    return make_response(res)

def inference_without_prompt():
    res = inference("a photo of an astronaut riding a horse on mars")
    return make_response(res)

#Added for Async inference by Teague@20111116

def async_inference_with_prompt(prompt):
    prompt = unquote(prompt)
    res = async_inference(prompt)
    return async_response(res,prompt)

def async_inference_without_prompt():
    res = async_inference(sentence_generator.sentence())
    return async_response(res)

async_inference_with_prompt("basketball in hand for a girl")