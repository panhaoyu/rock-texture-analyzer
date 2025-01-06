import subprocess
import time
from pathlib import Path
from pprint import pprint

import requests

from p3_表面显微镜扫描数据处理.config import endpoint, bucket_name, dashscope_api_key


def erase_image(image_url: str, mask_url: str, foreground_url: str) -> str:
    """图像擦除补全函数"""
    api_key = dashscope_api_key
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
            "dilate_flag": True,
            'fast_mode': False,
            'add_watermark': False,
        }
    }
    response = requests.post(
        'https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis',
        headers=headers,
        json=payload
    )
    data = response.json()
    try:
        task_id = data['output']['task_id']
    except Exception:
        pprint(data)
        raise
    while True:
        status_response = requests.get(
            f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}',
            headers={"Authorization": f"Bearer {api_key}"}
        )
        data = status_response.json()
        try:
            status = data['output']['task_status']
        except Exception:
            pprint(data)
            raise
        match status:
            case "SUCCEEDED":
                return status_response.json()['output']['output_image_url']
            case "FAILED":
                message = status_response.json()['output'].get('message', '未知错误')
                raise Exception(f"任务执行失败: {message}")
            case _:
                time.sleep(5)


def upload_to_oss(base_dir: Path, file_path: Path) -> str:
    """上传文件到OSS并返回访问路径"""
    relative_path = file_path.relative_to(base_dir)
    oss_path = (Path("temp") / relative_path).as_posix()
    cmd = [
        'ossutil64',
        f'--endpoint={endpoint}',
        'cp',
        '-f',
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
