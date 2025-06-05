import shutil
import boto3

local_folder = r"path/to/your/local/unzipped/folder"  # Replace with your local folder path
zip_output = local_folder + ".zip"
bucket_name = 'your-s3-bucket-name' # Replace with your S3 bucket name
s3_key = 'dataset.zip'

shutil.make_archive(local_folder, 'zip', local_folder)

s3 = boto3.client('s3')
s3.upload_file(zip_output, bucket_name, s3_key)
print("Uploaded to S3 successfully")