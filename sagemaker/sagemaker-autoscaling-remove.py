import boto3

ENDPOINT_NAME = "huggingface-pytorch-inference-async"

client = boto3.client(
    "application-autoscaling"
)  # Common class representing Application Auto Scaling for SageMaker amongst other services

resource_id = (
    "endpoint/" + ENDPOINT_NAME + "/variant/" + "AllTraffic"
)  # This is the format in which application autoscaling references the endpoint


response = client.delete_scaling_policy(
    PolicyName="Invocations-ScalingPolicy",
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount"
    )


response = client.deregister_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId=resource_id,
    ScalableDimension="sagemaker:variant:DesiredInstanceCount"
    )