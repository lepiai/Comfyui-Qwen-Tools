import torch 
import requests
import io
import os
import base64
import time
from PIL import Image
import numpy as np
from comfy.utils import ProgressBar

# 配置文件路径（用于存储API密钥）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "qwen_image_edit_config.txt")

class QwenImageEditNode:
    """
    调用阿里云 DashScope Qwen 图像编辑API的ComfyUI节点
    支持URL和Base64两种输入，显示实时执行状态
    """

    def __init__(self):
        self.api_key = self._load_api_key()
        self.api_endpoint = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "将图中的人物改为站立姿势，弯腰握住狗的前爪",
                    "tooltip": "图像编辑的文本指令"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 300,
                    "step": 10,
                    "tooltip": "API请求超时的超时时间（秒）"
                }),
            },
            "optional": {
                "input_image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "阿里云 DashScope API 密钥"
                }),
                "manual_image_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "手动输入图像URL（若无图像输入则必填）"
                }),
            }
        }


    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "AliCloud/Qwen"
    EXPERIMENTAL = True

    # 所有辅助方法保持不变
    def _load_api_key(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("api_key="):
                        return line.strip().split("=", 1)[1]
        return ""

    def _save_api_key(self, api_key):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            if api_key:
                f.write(f"api_key={api_key}\n")
        self.api_key = api_key

    def _image_to_input(self, image_tensor, manual_image_url):
        if manual_image_url:
            return {"type": "image", "image": manual_image_url}

        img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np).convert("RGB")
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        return {"type": "image", "image": f"data:image/png;base64,{img_base64}"}

    def _download_image_from_url(self, url, progress=None):
        if progress:
            progress.update(90)  # 移除desc参数
            
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0
        image_data = b""
        
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                image_data += chunk
                downloaded_size += len(chunk)
                if total_size > 0 and progress:
                    # 移除desc参数，只保留进度值更新
                    progress.update(90 + min(10, int(downloaded_size / total_size * 10)))

        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        
        if progress:
            progress.update(100)  # 移除desc参数
            
        return torch.from_numpy(img_np).unsqueeze(0)
    
    # 修复进度条更新方式
    def edit_image(self, prompt, input_image=None, api_key="", manual_image_url="", timeout=120):
        # 创建进度条（总进度值设为100）
        pbar = ProgressBar(100)
        pbar.update(0)  # 移除desc参数

        # API key 处理
        current_key = api_key if api_key else self.api_key
        if not current_key:
            raise ValueError("API 密钥未设置，请输入或保存")
        pbar.update(5)  # 移除desc参数

        if api_key:
            self._save_api_key(api_key)
            pbar.update(10)  # 移除desc参数

        # 检查图像输入
        if input_image is None and not manual_image_url:
            raise ValueError("必须提供 input_image 或 manual_image_url 其中之一")

        # 处理图像输入
        try:
            image_input = self._image_to_input(input_image, manual_image_url) if input_image is not None or manual_image_url else None
            pbar.update(20)  # 移除desc参数
        except Exception as e:
            raise RuntimeError(f"图像处理失败: {str(e)}")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}"
        }

        payload = {
            "model": "qwen-image-edit",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            image_input,
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
            },
            "parameters": {
                "result_format": "message"
            }
        }

        # 发送API请求（使用同步方式）
        try:
            pbar.update(30)  # 移除desc参数
            response = requests.post(
                self.api_endpoint, 
                json=payload, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            pbar.update(60)  # 移除desc参数
        except requests.exceptions.Timeout:
            raise RuntimeError(f"API请求超时（{timeout}秒）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {str(e)}")

        # 解析响应
        try:
            result = response.json()
            pbar.update(70)  # 移除desc参数
            
            message = result["output"]["choices"][0]["message"]
            image_url = None
            for c in message["content"]:
                if "image" in c:
                    image_url = c["image"]
                    break
            
            if not image_url:
                raise RuntimeError("API 响应中未找到图像")
            pbar.update(80)  # 移除desc参数
            
        except Exception as e:
            raise RuntimeError(f"解析响应失败：{e}\n完整响应: {result}")

        # 下载编辑后的图像
        edited_image = self._download_image_from_url(image_url, pbar)
        pbar.update(100)  # 移除desc参数
        
        return (edited_image,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "QwenImageEditNode": QwenImageEditNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditNode": "Qwen Image Edit 6 (DashScope)"
}