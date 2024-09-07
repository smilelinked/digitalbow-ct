import os
import boto3
from botocore.exceptions import ClientError

bucket = os.getenv("BUCKET", "smilelink")

s3_client = boto3.client('s3',
                         aws_access_key_id=os.getenv('OBS_AK', 'NCZPQASHJNW2URNGB9SI'),
                         aws_secret_access_key=os.getenv('OBS_SK', 'lXLZ9J1yUJYMrUBYZX2oAmzc3uvbSEIOSckpEsvN'),
                         endpoint_url=os.getenv('OBS_URI', 'https://obs.cn-east-3.myhuaweicloud.com'))


def new_client():
    return boto3.client('s3',
                        aws_access_key_id=os.getenv('OBS_AK', 'NCZPQASHJNW2URNGB9SI'),
                        aws_secret_access_key=os.getenv('OBS_SK', 'lXLZ9J1yUJYMrUBYZX2oAmzc3uvbSEIOSckpEsvN'),
                        endpoint_url=os.getenv('OBS_URI', 'https://obs.cn-east-3.myhuaweicloud.com'))


def get_obj(file):
    return s3_client.get_object(Bucket=bucket, Key=file)


def list_objects(prefix):
    return s3_client.list_objects(Bucket=bucket, Prefix=prefix, MaxKeys=1000)


def get_obj_exception(file):
    resp = s3_client.get_object(Bucket=bucket, Key=file)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300:
        raise Exception(f"read file {file} failed with resp {resp}")

    return resp.get('Body')


def has_obj(obj):
    try:
        s3_client.head_object(Bucket=bucket, Key=obj)
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
        else:
            raise Exception(f"head object failed with {e}")


def put_obj(obj, body=''):
    client = s3_client
    if body == '':
        client = new_client()
    return client.put_object(Bucket=bucket, Key=obj, Body=body)


def put_obj_exception(obj, body):
    resp = s3_client.put_object(Bucket=bucket, Key=obj, Body=body)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300:
        raise Exception(f"put obj {obj} failed with resp {resp}")


def generate_signed_url(key, expires=3600):
    return s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=expires
    )


def get_obj_metadata(key):
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
        # if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300:
        #     raise Exception(f"resp: {resp}")
    except Exception as e:
        raise Exception(f"get obj {key} metadata failed with error {e}")


def del_objects_by_prefix(prefix):
    resp = list_objects(prefix)
    if resp.get('ResponseMetadata').get('HTTPStatusCode') > 300 or resp.get('Contents') is None:
        return
    objs = [{'Key': obj.get('Key')} for obj in resp.get('Contents')]
    s3_client.delete_objects(Bucket=bucket, Delete={'Objects': objs, 'Quiet': False})


def list_all_files(prefix):
    # 存储所有符合条件的文件
    all_files = []
    continuation_token = None

    # 循环获取所有对象
    while True:
        # 列出对象并处理分页
        if continuation_token:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, ContinuationToken=continuation_token)
        else:
            response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

        # 检查请求结果
        if 'Contents' in response:
            # 过滤掉指定前缀的文件
            filtered_files = [content['Key'] for content in response['Contents']]
            all_files.extend(filtered_files)

        # 如果没有更多对象，退出循环
        if not response.get('IsTruncated'):  # 如果返回的响应没有被截断，表示已经获取完所有对象
            break

        # 如果有更多对象，将继续分页
        continuation_token = response.get('NextContinuationToken')

    return all_files
