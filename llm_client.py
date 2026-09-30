import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Dict, Optional

# 加载 .env 文件中的环境变量
load_dotenv()

class AgentsLLM:
    """
    兼容 OpenAI 接口的 LLM 客户端，返回完整响应。
    """
    def __init__(self, model: str = None, apiKey: str = None, baseUrl: str = None, timeout: int = None):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("模型ID、API密钥和服务地址必须被提供或在.env文件中定义。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> Optional[str]:
        """
        等待模型完整响应后返回文本，由调用方统一展示。
        """
        print(f"🧠 正在调用 {self.model} 模型，请稍候...", flush=True)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=False,
            )

            if not response.choices:
                return None
            content = response.choices[0].message.content
            return content.strip() if content else None

        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return None

