import boto3
import pandas as pd
from io import StringIO
from src.logger import logging


class s3_operations:

    def __init__(self, bucket_name, aws_access_key, aws_secret_key):

        self.bucket_name = bucket_name

        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name="us-east-1"  
        )

    def fetch_file_from_s3(self, file_key):

        try:

            logging.info(
                f"Fetching file '{file_key}' from S3 bucket '{self.bucket_name}'..."
            )

            obj = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=file_key
            )

            data = obj["Body"].read().decode("utf-8")

            df = pd.read_csv(StringIO(data))

            logging.info(f"Successfully fetched '{file_key}'")

            return df

        except Exception as e:

            logging.error(
                f"Failed to fetch '{file_key}' from S3: {e}"
            )

            raise

# Example usage
# if __name__ == "__main__":
#     # Replace these with your actual AWS credentials and S3 details
#     BUCKET_NAME = "bucket-name"
#     AWS_ACCESS_KEY = "AWS_ACCESS_KEY"
#     AWS_SECRET_KEY = "AWS_SECRET_KEY"
#     FILE_KEY = "data.csv"  # Path inside S3 bucket

#     data_ingestion = s3_operations(BUCKET_NAME, AWS_ACCESS_KEY, AWS_SECRET_KEY)
#     df = data_ingestion.fetch_file_from_s3(FILE_KEY)

#     if df is not None:
#         print(f"Data fetched with {len(df)} records..")  # Display first few rows of the fetched DataFrame
