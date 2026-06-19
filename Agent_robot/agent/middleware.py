from typing import Callable, Any
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.prompt_loader import load_system_prompt, load_report_prompt


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command]
) -> ToolMessage | Command:
    try:
        result = handler(request)
        if request.tool_call['name'] == 'fill_context_for_report':
            request.runtime.context["report"] = True
        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败: {e}")
        raise


@before_model
def log_before_model(state:AgentState, runtime: Runtime) -> dict[str, Any] | None:
    return None


@dynamic_prompt
def report_prompt_switch(request: ModelRequest) -> str:
    is_report = request.runtime.context.get("report", False)
    if is_report:
        return load_report_prompt()
    return load_system_prompt()
