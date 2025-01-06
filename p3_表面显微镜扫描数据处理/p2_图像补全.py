import os
import pprint
import time

import requests

from p3_表面显微镜扫描数据处理.config import raw_image_url, mask_image_url, keep_image_url


def erase_image(image_url: str, mask_url: str, foreground_url: str) -> str:
    """图像擦除补全函数"""
    api_key = os.getenv('DASHSCOPE_API_KEY')
    print(f'{api_key=}')
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
            "dilate_flag": True
        }
    }
    response = requests.post(
        'https://dashscope.aliyuncs.com/api/v1/services/aigc/image2image/image-synthesis',
        headers=headers,
        json=payload
    )
    pprint.pprint(response.json())
    task_id = response.json()['output']['task_id']
    while True:
        status_response = requests.get(
            f'https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}',
            headers={"Authorization": f"Bearer {api_key}"}
        )
        pprint.pprint(status_response.json())
        status = status_response.json()['output']['task_status']
        match status:
            case "SUCCEEDED":
                return status_response.json()['output']['output_image_url']
            case "FAILED":
                raise Exception("任务执行失败")
            case _:
                time.sleep(5)


output_image_url = erase_image(raw_image_url, mask_image_url, keep_image_url)
