from typing import TypedDict


class AgentWebContext(TypedDict, total=False):
    report: bool
    enable_sft: bool
    enable_rl: bool


def get_customer_agent():
    """延迟构建Agent"""
    from langchain.agents import create_agent
    from agent.middleware import log_before_model, monitor_tool, report_prompt_switch
    from model.factory import get_chat_model
    from utils.prompt_loader import load_system_prompt
    from agent.agent_tools import (
        fetch_external_data,
        fill_context_for_report,
        get_current_month,
        get_user_id,
        get_user_location,
        get_weather,
        rag_summarize,
    )

    tools = [
        rag_summarize,
        get_user_location,
        get_weather,
        get_user_id,
        get_current_month,
        fetch_external_data,
        fill_context_for_report,
    ]

    return create_agent(
        get_chat_model(),
        tools,
        system_prompt=load_system_prompt(),
        middleware=[monitor_tool, log_before_model, report_prompt_switch],
        context_schema=AgentWebContext,
    )
