from sagemaker.local import LocalSession
from sagemaker.pytorch import PyTorch

sagemaker_session = LocalSession()
sagemaker_session.config = {'local': {'local_code': True}}

# Configure PyTorch (no training happens yet)
pytorch_estimator = PyTorch(
    entry_point='train.py',
    source_dir='code',
    role='SageMakerRole',
    instance_type='local',
    instance_count=1,
    framework_version='1.8.0',
    py_version='py36',
)

# Train the estimator
pytorch_estimator.fit(
    inputs={
        # 'images': 's3://rumc-picai-d-training-public/images/',
        # 'labels': 's3://rumc-picai-d-training-public/picai_labels/',
        'scripts': 'file://code/',  # this will be replaced by the participant's code in an S3 bucket
    }
)

# Deploys the model that was generated by fit() to local endpoint in a container
# pytorch_predictor = pytorch_estimator.deploy(initial_instance_count=1, instance_type='local')

# Serializes data and makes a prediction request to the local endpoint
# response = pytorch_predictor.predict(data)

# Tears down the endpoint container and deletes the corresponding endpoint configuration
# pytorch_predictor.delete_endpoint()

# Deletes the model
# pytorch_predictor.delete_model()
