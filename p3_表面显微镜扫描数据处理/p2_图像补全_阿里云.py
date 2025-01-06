import os
import time

import requests


def erase_image(image_url: str, mask_url: str, foreground_url: str) -> str:
    """图像擦除补全函数"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-DashScope-Async": "enable",
        "Content-Type": "application/json",
        "X-DashScope-DataInspection": "enable"
    }
    payload = {
        "model": "image-erase-completion",
        "input": {
            "image_url": image_url,
            "mask_url": mask_url,
            "foreground_url": foreground_url
        },
        "parameters": {
            "dilate_flag": False,
            'fast_mode': False,
            'add_watermark': False,
        }
    }
    response = requests.post(
        'https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis',
        headers=headers,
        json=payload
    )
    task_id = response.json()['output']['task_id']
    while True:
        status_response = requests.get(
            f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}',
            headers={"Authorization": f"Bearer {api_key}"}
        )
        status = status_response.json()['output']['task_status']
        match status:
            case "SUCCEEDED":
                return status_response.json()['output']['output_image_url']
            case "FAILED":
                message = status_response.json()['output'].get('message', '未知错误')
                raise Exception(f"任务执行失败: {message}")
            case _:
                time.sleep(5)


from pathlib import Path
import subprocess

from p3_表面显微镜扫描数据处理.config import endpoint, bucket_name


def upload_to_oss(base_dir: Path, file_path: Path) -> str:
    """上传文件到OSS并返回访问路径"""
    relative_path = file_path.relative_to(base_dir)
    oss_path = Path("temp") / relative_path
    cmd = [
        'ossutil64',
        f'--endpoint={endpoint}',
        f'--include="{file_path.name}"',
        'cp',
        str(file_path),
        f'oss://{bucket_name}/{oss_path}'
    ]
    subprocess.run(cmd, check=True)
    return f'https://{bucket_name}.{endpoint}/{oss_path}'


def erase_image_with_oss(base_dir: Path,
                         local_image_path: Path,
                         local_mask_path: Path,
                         local_foreground_path: Path) -> str:
    """上传本地文件到OSS并调用erase_image函数"""
    image_url = upload_to_oss(base_dir, local_image_path)
    mask_url = upload_to_oss(base_dir, local_mask_path)
    foreground_url = upload_to_oss(base_dir, local_foreground_path)
    return erase_image(image_url, mask_url, foreground_url)
