"""
尝试调用阿里的接口，不过失败，无法使用。
"""

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
