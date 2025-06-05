from sagemaker.pytorch import PyTorch
import sagemaker

role = "arn:aws:iam::your-account-id:role/your-sagemaker-role-name" # Replace with your SageMaker execution role ARN
bucket_name = 'your-s3-bucket-name'  # Replace with your S3 bucket name

#Model 1- YOLOv11 for Face Detection
yolov11_estimator = PyTorch(
    entry_point="train_yolov11.py",          
    role=role,
    source_dir=".",
    dependencies=["requirements.txt"],  # Ensure you have a requirements.txt for YOLOv11 dependencies, add path to it if needed
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    framework_version="1.13",
    py_version="py39",
    hyperparameters={
        'epochs': 100,
        'imgsz': 640,
        'batch': 16,
        'data': '/opt/ml/input/data/train/face_data.yaml',
    },
    output_path=f"{bucket_name}/yolov11_model_output/",
)

inputs_yolo = {
    'train': f'{bucket_name}/dataset.zip',
}

# Model 2 - DenseNet for Face Classification
densenet_estimator = PyTorch(
    entry_point="train_densenet.py",        
    role=role,
    source_dir=".",
    dependencies=["requirements.txt"],
    instance_count=1,
    instance_type="ml.g4dn.xlarge",
    framework_version="1.13",
    py_version="py39",
    hyperparameters={
        'epochs': 20,
        'batch-size': 32,
        'lr': 0.001,
        'img-size': 112
    },
    output_path=f"{bucket_name}/densenet_model_output/",
)

inputs_densenet = {
    'train': f'{bucket_name}/dataset.zip',
}

print("Starting YOLOv11 Training (Face Detection)")
yolov11_estimator.fit(inputs_yolo)


print("Starting DenseNet Training (Face Classification)...")
densenet_estimator.fit(inputs_densenet)

print("Training completed for both YOLOv11 and DenseNet models.")