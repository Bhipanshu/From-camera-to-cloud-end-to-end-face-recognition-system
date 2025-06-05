from sagemaker.pytorch import PyTorchModel
import sagemaker

role = "arn:aws:iam::your-account-id:role/your-sagemaker-role-name" # Replace with your SageMaker execution role ARN
session = sagemaker.Session()

s3_bucket_path = "s3://your-s3-bucket-name"  # Replace with your S3 bucket name


yolo_model_s3_path = f"{s3_bucket_path}/yolov11_model_output/output/model.tar.gz" 
densenet_model_s3_path = f"{s3_bucket_path}/densenet_model_output/output/model.tar.gz"


print("Deploying YOLOv11 Face Detector")

yolo_model = PyTorchModel(
    model_data=yolo_model_s3_path,
    role=role,
    entry_point="inference_yolov11.py",
    source_dir=".",
    framework_version="1.13",
    py_version="py39",
)

yolo_predictor = yolo_model.deploy(
    instance_type="ml.g4dn.xlarge",
    initial_instance_count=1,
    endpoint_name="yolov11-face-detector"
)

print("YOLOv11 Face Detector deployed at endpoint: yolov11-face-detector")


print("\nDeploying DenseNet Face Classifier...")

densenet_model = PyTorchModel(
    model_data=densenet_model_s3_path,
    role=role,
    entry_point="inference_densenet.py",
    source_dir=".",
    framework_version="1.13",
    py_version="py39",
)

densenet_predictor = densenet_model.deploy(
    instance_type="ml.g4dn.xlarge",
    initial_instance_count=1,
    endpoint_name="densenet-face-classifier"
)

print("DenseNet Face Classifier deployed at endpoint: densenet-face-classifier")
