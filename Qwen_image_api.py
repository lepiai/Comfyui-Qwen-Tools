import torch 
import requests
import io
import os
import base64
import json
from PIL import Image
import numpy as np
from comfy.utils import ProgressBar

# 配置文件路径（用于存储API密钥）
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "qwen_image_edit_config.txt")

# 共享工具类
class QwenApiUtils:
    """Qwen API 共享工具方法"""
    
    @staticmethod
    def load_api_key():
        """加载保存的API密钥"""
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("api_key="):
                        return line.strip().split("=", 1)[1]
        return ""
    
    @staticmethod
    def save_api_key(api_key):
        """保存API密钥到配置文件"""
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            if api_key:
                f.write(f"api_key={api_key}\n")
    
    @staticmethod
    def tensor_to_base64(image_tensor):
        """将ComfyUI图像张量转换为Base64编码"""
        img_np = (image_tensor[0].cpu().numpy() * 255).astype(np.uint8)
        img_pil = Image.fromarray(img_np).convert("RGB")
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    @staticmethod
    def download_image_from_url(url, progress=None):
        """从URL下载图像并转换为ComfyUI张量"""
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        image_data = response.content
        img_pil = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        
        if progress:
            progress.update(100)
            
        return torch.from_numpy(img_np).unsqueeze(0)


# 图像编辑节点
class QwenImageEditNodeExp:
    """
    调用阿里云 DashScope Qwen 图像编辑API的ComfyUI节点
    支持 URL 和 Base64 输入，支持可选参数：negative_prompt / prompt_extend / watermark
    """

    def __init__(self):
        self.api_key = QwenApiUtils.load_api_key()
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
            },
            "optional": {
                "input_image": ("IMAGE",),                
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "反向提示词，用于描述不希望出现的内容"
                }),
                "manual_image_url": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "手动输入图像URL（若无图像输入则必填）"
                }),
                "prompt_extend": (["true", "false"], {
                    "default": "true",
                    "tooltip": "是否开启prompt智能改写"
                }),
                "watermark": (["false", "true"], {
                    "default": "false",
                    "tooltip": "是否添加水印标识"
                }),
                "timeout": ("INT", {
                    "default": 120,
                    "min": 30,
                    "max": 300,
                    "step": 10,
                    "tooltip": "API请求超时的超时时间（秒）"
                }),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "阿里云 DashScope API 密钥"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "AliCloud/Qwen"
    DESCRIPTION = "填写api，正向提示词必填，负向提示词可选填，支持本地上传图片或填写图片网络地址（manual_image_url），prompt_extend开启后会智能优化提示词，针对短提示词效果明显，watemark水印默认关闭"
    EXPERIMENTAL = True

    def _image_to_input(self, image_tensor, manual_image_url):
        if manual_image_url:
            return {"type": "image", "image": manual_image_url}

        img_base64 = QwenApiUtils.tensor_to_base64(image_tensor)
        return {"type": "image", "image": f"data:image/png;base64,{img_base64}"}

    def edit_image(self, prompt, timeout=120, input_image=None, manual_image_url="", api_key="", 
                   negative_prompt="", prompt_extend="true", watermark="false"):
        pbar = ProgressBar(100)
        pbar.update(0)

        # API key处理
        current_key = api_key if api_key else self.api_key
        if not current_key:
            raise ValueError("API 密钥未设置，请输入或保存")

        if api_key:
            QwenApiUtils.save_api_key(api_key)

        # 图像输入校验
        if input_image is None and not manual_image_url:
            raise ValueError("必须提供 input_image 或 manual_image_url 其中之一")

        image_input = self._image_to_input(input_image, manual_image_url)
        pbar.update(20)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {current_key}"
        }

        # 构造请求体
        parameters = {
            "result_format": "message",
            "prompt_extend": prompt_extend.lower() == "true",
            "watermark": watermark.lower() == "true"
        }
        if negative_prompt.strip():
            parameters["negative_prompt"] = negative_prompt.strip()

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
            "parameters": parameters
        }

        try:
            pbar.update(40)
            response = requests.post(
                self.api_endpoint, 
                json=payload, 
                headers=headers, 
                timeout=timeout
            )
            response.raise_for_status()
            pbar.update(70)
        except requests.exceptions.Timeout:
            raise RuntimeError(f"API请求超时（{timeout}秒）")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API请求失败: {str(e)}")

        try:
            result = response.json()
            pbar.update(80)

            message = result["output"]["choices"][0]["message"]
            image_url = None
            for c in message["content"]:
                if "image" in c:
                    image_url = c["image"]
                    break

            if not image_url:
                raise RuntimeError("API 响应中未找到图像")
            pbar.update(85)

        except Exception as e:
            raise RuntimeError(f"解析响应失败：{e}\n完整响应: {result}")

        edited_image = QwenApiUtils.download_image_from_url(image_url, pbar)
        pbar.update(100)
        
        return (edited_image,)


# 提示词反推节点
class QwenPromptInversionOpenAICompat:
    """
    使用阿里云百炼（OpenAI兼容接口，国内区）调用 Qwen-VL 模型，
    将输入图片反推为 SD/SDXL 的正向提示词。
    """

    def __init__(self):
        self.api_endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_image": ("IMAGE",),
                "api_key": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "阿里云百炼 API Key（必须填写）"
                }),
            },
            "optional": {
                "model": ("STRING", {
                    "default": "qwen-vl-plus",
                    "tooltip": "可用：qwen-vl-plus / qwen-vl-max / qwen2.5-vl-* 等"
                }),
                "instruction": ("STRING", {
                    "multiline": True,
                    "default": "用中文描述这张图片，包括画面风格、元素、文字等,使用自然语言描述，不需要格式和列表，不需要做任何说明",
                    "tooltip": "引导模型把图像描述重写为 Qwen image 的正向提示词"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 64, "max": 2048, "step": 32
                }),
                "timeout": ("INT", {
                    "default": 60, "min": 10, "max": 300, "step": 5
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "infer_prompt"
    CATEGORY = "AliCloud/Qwen"
    DESCRIPTION = "调用阿里云百炼（OpenAI兼容接口，国内区）图文理解API，将本地图片反推为 Qwen image 提示词"
    EXPERIMENTAL = True

    def _tensor_to_data_url(self, image_tensor):
        """将ComfyUI图像张量转换为data URL"""
        img_base64 = QwenApiUtils.tensor_to_base64(image_tensor)
        return f"data:image/png;base64,{img_base64}"

    def infer_prompt(self, input_image=None, api_key="",
                     model="qwen-vl-plus", instruction="", temperature=0.2,
                     max_tokens=512, timeout=60):

        pbar = ProgressBar(100); pbar.update(0)

        if not api_key.strip():
            raise ValueError("API Key 未设置，请在节点中填写。")

        if input_image is None:
            raise ValueError("必须提供 input_image。")

        # 将输入图像转为 data URL
        image_url = self._tensor_to_data_url(input_image)
        pbar.update(20)

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction or "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens)
        }

        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json"
        }

        try:
            pbar.update(40)
            resp = requests.post(self.api_endpoint, headers=headers, json=payload, timeout=timeout)
            pbar.update(70)
        except Exception as e:
            raise RuntimeError(f"网络请求失败：{e}")

        if not resp.ok:
            raise RuntimeError(f"API 请求失败（HTTP {resp.status_code}）：{resp.text}")

        try:
            data = resp.json()
            choice0 = data["choices"][0]
            prompt_text = choice0["message"]["content"].strip()
        except Exception as e:
            raise RuntimeError(f"解析响应失败：{e}\n完整响应：{json.dumps(data, ensure_ascii=False)}")

        pbar.update(100)
        return (prompt_text,)


# 节点映射表
NODE_CLASS_MAPPINGS = {
    "QwenImageEditNodeExp": QwenImageEditNodeExp,
    "QwenPromptInversionOpenAICompat": QwenPromptInversionOpenAICompat
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "QwenImageEditNodeExp": "Qwen Image Edit (Expert)",
    "QwenPromptInversionOpenAICompat": "Qwen Prompt Inversion v3"
}

