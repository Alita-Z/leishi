import os
import json
import asyncio
from typing import AsyncGenerator, List, Dict
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
from pydantic import BaseModel
import openai

# 加载环境变量
load_dotenv()

# 初始化FastAPI应用
app = FastAPI(title="职能沟通翻译助手")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="."), name="static")

# 导入提示词
from prompt import PRODUCT_TO_DEV_PROMPT, DEV_TO_PRODUCT_PROMPT

# 获取OpenAI API密钥
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.base_url = os.getenv("OPENAI_API_BASE")
model_name = os.getenv("OPENAI_MODEL_NAME")

class TranslateRequest(BaseModel):
    text: str
    direction: str  # "product_to_dev" 或 "dev_to_product"
    history: List[Dict[str, str]] = []  # 多轮对话历史

async def simulate_streaming_response(text: str) -> AsyncGenerator[str, None]:
    """模拟流式响应"""
    sentences = text.split('。')
    for sentence in sentences:
        if sentence.strip():
            yield sentence + '。\n'
            await asyncio.sleep(0.5)

async def openai_streaming_response(messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """使用OpenAI API获取流式响应"""
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
            temperature=0.7
        )
        
        for chunk in response:
            if chunk.choices:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
    except Exception as e:
        pass

async def translate_text(text: str, direction: str, history: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """翻译文本"""
    # 构造系统提示词
    if direction == "product_to_dev":
        system_prompt = PRODUCT_TO_DEV_PROMPT
    elif direction == "dev_to_product":
        system_prompt = DEV_TO_PRODUCT_PROMPT
    else:
        raise ValueError("Invalid direction. Must be 'product_to_dev' or 'dev_to_product'")
    
    # 构造消息历史
    messages = []
    
    # 添加系统提示词
    messages.append({"role": "system", "content": system_prompt})
    
    # 添加历史对话
    messages.extend(history)
    
    # 添加当前用户输入
    messages.append({"role": "user", "content": text})
    
    # 如果有API密钥则使用OpenAI，否则使用模拟数据
    if model_name:
        async for chunk in openai_streaming_response(messages):
            yield chunk
    else:
        # 模拟数据
        mock_responses = {
            "product_to_dev": "推荐采用协同过滤算法实现推荐功能。数据源需要用户行为日志和商品信息表。要求支持实时推荐，响应时间不超过200ms。需要开发推荐引擎、数据处理模块和缓存机制。预计开发周期为10人日。",
            "dev_to_product": "数据库查询优化将显著提升用户页面加载速度，改善用户体验。此优化可支撑用户量增长30%，同时降低服务器成本约15%。无业务风险，可直接上线。"
        }
        async for chunk in simulate_streaming_response(mock_responses.get(direction, "模拟翻译结果")):
            yield chunk

@app.post("/translate")
async def translate(request: TranslateRequest):
    """
    翻译接口
    """
    return StreamingResponse(
        translate_text(request.text, request.direction, request.history),
        media_type="text/plain"
    )

@app.get("/", response_class=HTMLResponse)
async def read_index():
    """
    返回主页
    """
    with open("index.html", "r", encoding="utf-8") as file:
        content = file.read()
    return HTMLResponse(content=content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)