import os
import sys
from pathlib import Path

# 保证从任意 cwd 启动都能找到工程包
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Generator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json

from optimization.inference_context import reset_inference_options, set_inference_options
from services.customer_agent import AgentWebContext, get_customer_agent
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tools import get_abs_path

app = FastAPI(title="煤矿挖掘机智能客服", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    enable_sft: bool = True
    enable_rl: bool = True


def _serialize_messages(messages: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        name = type(m).__name__
        role = getattr(m, "type", None) or name.lower().replace("message", "")
        content = getattr(m, "content", "") or ""
        tool_calls = getattr(m, "tool_calls", None)
        
        # 过滤掉：AI/assistant 角色且内容为空但有工具调用的消息
        # 这类消息是 Agent 内部的工具调用决策，对用户无意义
        if (role == "ai" or role == "assistant") and not content and tool_calls:
            continue
        
        row: dict[str, Any] = {"role": role, "content": content}
        if tool_calls:
            row["tool_calls"] = tool_calls
        out.append(row)
    return out


_NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


@app.get("/")
def index():
    index_path = _static / "index.html"
    if not index_path.exists():
        raise HTTPException(404, "index.html 缺失")
    return FileResponse(index_path, headers=_NO_CACHE)


@app.get("/api/health")
def health():
    """启动自检：常见 500 原因（未配置 Key、向量库目录不存在）。"""
    key_set = bool(os.environ.get("DASHSCOPE_API_KEY"))
    chroma_dir = get_abs_path(chroma_conf["persist_directory"])
    chroma_exists = Path(chroma_dir).is_dir()
    return {
        "service": "excavator-cs-webapp",
        "ok": key_set and chroma_exists,
        "dashscope_api_key_set": key_set,
        "chroma_persist_directory": chroma_dir,
        "chroma_persist_exists": chroma_exists,
        "hint": (
            "请设置环境变量 DASHSCOPE_API_KEY 后重启服务。"
            if not key_set
            else (
                "未找到 Chroma 持久化目录，请先在本机跑过向量入库（如 rag/vector_store 的 load_document）。"
                if not chroma_exists
                else "基础检查通过。"
            )
        ),
    }


@app.post("/api/chat")
def chat(body: ChatRequest):
    token = set_inference_options({"enable_sft": body.enable_sft, "enable_rl": body.enable_rl})
    try:
        ctx: AgentWebContext = {
            "report": False,
            "enable_sft": body.enable_sft,
            "enable_rl": body.enable_rl,
        }
        result = get_customer_agent().invoke(
            {"messages": [("user", body.message.strip())]},
            context=ctx,
        )
        msgs = result.get("messages", [])
        return {"ok": True, "messages": _serialize_messages(msgs)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[api/chat] 请求失败")
        hint = "若提示与鉴权/配额相关，请检查 DASHSCOPE_API_KEY 与账户额度；若与向量库相关，请确认 chroma_db 已生成。"
        raise HTTPException(
            status_code=500,
            detail={"error": f"{type(e).__name__}: {e}", "hint": hint},
        ) from e
    finally:
        reset_inference_options(token)


def _get_final_answer(messages: list[Any]) -> str:
    """从消息列表中提取最终的AI回答（只返回最后一个AI消息的内容）"""
    for m in reversed(messages):
        name = type(m).__name__
        role = getattr(m, "type", None) or name.lower().replace("message", "")
        content = getattr(m, "content", "") or ""
        if (role == "ai" or role == "assistant") and content:
            return content
    return ""


def _stream_answer_chunks(answer: str) -> Generator[str, None, None]:
    """逐字流式输出回答内容"""
    buffer = ""
    for char in answer:
        buffer += char
        if len(buffer) >= 10 or char in ["。", "！", "？", "\n"]:
            yield f"data: {json.dumps({'type': 'chunk', 'content': buffer})}\n\n"
            buffer = ""
    if buffer:
        yield f"data: {json.dumps({'type': 'chunk', 'content': buffer})}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"


def _stream_rag_answer(query: str, enable_sft: bool, enable_rl: bool):
    """直接RAG管道 + 流式输出（支持SFT和RL）"""
    from agent.agent_tools import get_rag_service
    
    try:
        rag = get_rag_service()
        
        # 使用流式方法，支持SFT和RL
        buffer = ""
        for chunk in rag.stream_rag_summarize(query, enable_sft=enable_sft, enable_rl=enable_rl):
            if chunk:
                buffer += chunk
                if len(buffer) >= 10 or buffer[-1:] in "。！？\n":
                    yield f"data: {json.dumps({'type': 'chunk', 'content': buffer})}\n\n"
                    buffer = ""
        if buffer:
            yield f"data: {json.dumps({'type': 'chunk', 'content': buffer})}\n\n"
        
    except Exception as e:
        logger.error(f"流式输出失败: {e}")
        yield f"data: {json.dumps({'type': 'chunk', 'content': '服务异常'})}\n\n"
    
    yield f"data: {json.dumps({'type': 'done', 'content': ''})}\n\n"


@app.post("/api/chat/stream")
def chat_stream(body: ChatRequest):
    try:
        return StreamingResponse(
            _stream_rag_answer(body.message.strip(), body.enable_sft, body.enable_rl),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
    except Exception as e:
        logger.exception("[api/chat/stream] 请求失败")
        raise HTTPException(status_code=500, detail={"error": f"{type(e).__name__}: {e}"})


def main():
    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8765"))
    print(
        f"\n>>> 煤矿挖掘机智能客服（新版 webapp）\n"
        f">>> 首页: http://{host}:{port}/\n"
        f">>> 自检: http://{host}:{port}/api/health  （应含 \"service\":\"excavator-cs-webapp\"）\n"
        f">>> 若 8765 已被占用，请先结束旧进程 excavator-cs-webapp，或设置 PORT=8766\n"
    )
    uvicorn.run("webapp.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
