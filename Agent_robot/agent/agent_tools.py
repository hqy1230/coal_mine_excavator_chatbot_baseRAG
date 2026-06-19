from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from rag.vector_store import VectorStoreService
from rag.rag_service import RagSummarizeService
import random
import os
import requests
from urllib.parse import quote
from typing import Dict, Any, Optional
from utils.config_handler import agent_conf
from utils.path_tools import get_abs_path
from utils.logger_handler import logger

# 懒加载变量
_vector_store = None
_rag = None


def get_vector_store():
    """延迟获取向量存储"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store


def get_rag_service():
    """延迟获取RAG服务"""
    global _rag
    if _rag is None:
        _rag = RagSummarizeService(get_vector_store())
    return _rag

# 用户ID和月份列表
USER_IDS = ["1001", "1002", "1003", "1004", "1005", "1006", "1007", "1008", "1009", "1010"]
MONTH_ARR = ["2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06",
             "2025-07", "2025-08", "2025-09", "2025-10", "2025-11", "2025-12"]

# 外部数据缓存
EXTERNAL_DATA: Dict[str, Dict[str, Dict[str, str]]] = {}


@tool(description="从向量存储中检索煤矿挖掘机相关参考资料")
def rag_summarize(query: str) -> str:
    """
    从向量数据库中检索与查询相关的煤矿挖掘机参考资料

    Args:
        query: 查询内容

    Returns:
        检索到的煤矿挖掘机参考资料
    """
    try:
        rag = get_rag_service()
        result = rag.rag_summarize(query)
        return result
    except Exception as e:
        logger.error(f"[rag_summarize] 检索失败: {str(e)}", exc_info=True)
        return "检索参考资料时发生错误，请稍后再试"


@tool(description="获取用户所在矿区城市名称，以纯字符形式返回")
def get_user_location() -> str:
    """
    通过IP定位获取用户真实城市（矿区所在地）

    Returns:
        城市名称，如果获取失败则返回默认城市
    """
    try:
        # 国内免费IP定位接口
        response = requests.get("https://who.pc6.com/json", timeout=5)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()
        city = data.get("city")
        return city if city else "深圳"
    except requests.exceptions.RequestException as e:
        logger.warning(f"[get_user_location] IP定位请求失败: {str(e)}")
        return "深圳"
    except ValueError as e:  # JSON解析错误
        logger.warning(f"[get_user_location] 响应解析失败: {str(e)}")
        return "深圳"
    except Exception as e:
        logger.error(f"[get_user_location] 获取城市失败: {str(e)}", exc_info=True)
        return "深圳"


@tool(description="获取指定矿区城市的实时天气，返回真实数据")
def get_weather(city: str) -> str:
    """
    获取指定矿区城市的实时天气信息

    Args:
        city: 城市名称

    Returns:
        天气信息字符串
    """
    # 验证输入参数，防止XSS攻击
    if not city or not isinstance(city, str):
        return "城市名称不能为空"

    # 对城市名进行URL编码
    encoded_city = quote(city)

    try:
        # 免费天气接口（支持全国城市，无需KEY）
        url = f"https://api.vvhan.com/api/weather?city={encoded_city}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()

        # 检查API返回的错误状态
        if data.get("status") == "error":
            return f"获取天气失败：{data.get('message', '未知错误')}"

        # 解析天气数据
        city_name = data.get("city", "未知城市")
        info = data.get("info", {})
        date = info.get("date", "未知日期")
        weather = info.get("type", "未知天气")
        temp = info.get("temp", "未知温度")
        wind = info.get("fengxiang", "未知风向")
        air = info.get("air", "未知")

        return f"""
【{city_name} 实时天气】
日期：{date}
天气：{weather}
温度：{temp}℃
风向：{wind}
空气质量：{air}
"""

    except requests.exceptions.Timeout:
        return f"获取{city}天气超时，请稍后再试"
    except requests.exceptions.RequestException as e:
        logger.warning(f"[get_weather] 天气API请求失败: {str(e)}")
        return f"获取{city}天气失败，请稍后再试"
    except ValueError as e:  # JSON解析错误
        logger.warning(f"[get_weather] 天气响应解析失败: {str(e)}")
        return f"获取{city}天气失败，数据格式错误"
    except Exception as e:
        logger.error(f"[get_weather] 获取天气失败: {str(e)}", exc_info=True)
        return f"获取{city}天气时发生未知错误"


@tool(description="获取用户ID，以纯字符形式返回")
def get_user_id() -> str:
    """
    随机获取一个用户ID

    Returns:
        用户ID字符串
    """
    return random.choice(USER_IDS)


@tool(description="获取当前月份，以纯字符形式返回")
def get_current_month() -> str:
    """
    随机获取一个月份

    Returns:
        月份字符串
    """
    return random.choice(MONTH_ARR)


def load_external_data_if_needed():
    """
    如果外部数据尚未加载，则从CSV文件加载
    """
    global EXTERNAL_DATA

    if EXTERNAL_DATA:
        return  # 数据已加载，直接返回

    if "external_data_path" not in agent_conf:
        raise KeyError("配置中缺少 external_data_path 字段")

    external_data_path = get_abs_path(agent_conf["external_data_path"])

    if not os.path.exists(external_data_path):
        raise FileNotFoundError(f"外部数据文件不存在: {external_data_path}")

    try:
        with open(external_data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 跳过表头，从第二行开始解析
        for line in lines[1:]:
            line = line.strip()
            if not line:  # 跳过空行
                continue

            arr = line.split(",")

            # 验证列数
            if len(arr) < 6:
                logger.warning(f"[load_external_data] 行格式错误，跳过: {line}")
                continue

            # 清理数据，去除引号
            user_id = arr[0].strip().strip('"\'')
            feature = arr[1].strip().strip('"\'')
            efficiency = arr[2].strip().strip('"\'')
            consumables = arr[3].strip().strip('"\'')
            comparison = arr[4].strip().strip('"\'')
            time_val = arr[5].strip().strip('"\'')

            # 初始化用户数据结构
            if user_id not in EXTERNAL_DATA:
                EXTERNAL_DATA[user_id] = {}

            # 存储数据
            EXTERNAL_DATA[user_id][time_val] = {
                "特征": feature,
                "效率": efficiency,
                "耗材": consumables,
                "对比": comparison,
            }

        logger.info(f"[load_external_data] 成功加载 {len(EXTERNAL_DATA)} 个用户的数据")

    except UnicodeDecodeError:
        logger.error(f"[load_external_data] 文件编码错误，请确保文件为UTF-8格式: {external_data_path}")
        raise
    except Exception as e:
        logger.error(f"[load_external_data] 加载外部数据失败: {str(e)}", exc_info=True)
        raise


@tool(description="检索指定用户在指定月份的煤矿挖掘机完整使用记录，以纯字符形式返回，如未检索到返回空字符串")
def fetch_external_data(user_id: str, month: str) -> str:
    """
    检索指定用户在指定月份的煤矿挖掘机使用记录

    Args:
        user_id: 用户ID
        month: 月份，格式如 "2025-01"

    Returns:
        煤矿挖掘机使用记录，未找到则返回空字符串
    """
    try:
        # 验证输入参数
        if not user_id or not isinstance(user_id, str):
            logger.warning("[fetch_external_data] 用户ID参数无效")
            return ""

        if not month or not isinstance(month, str):
            logger.warning("[fetch_external_data] 月份参数无效")
            return ""

        # 确保数据已加载
        load_external_data_if_needed()

        # 查找用户数据
        user_data = EXTERNAL_DATA.get(user_id)
        if not user_data:
            logger.info(f"[fetch_external_data] 未找到用户 {user_id} 的数据")
            return ""

        # 查找月份数据
        month_data = user_data.get(month)
        if not month_data:
            logger.info(f"[fetch_external_data] 未找到用户 {user_id} 在 {month} 的数据")
            return ""

        # 格式化返回数据
        result = f"用户 {user_id} 在 {month} 的使用记录:\n"
        for key, value in month_data.items():
            result += f"- {key}: {value}\n"

        return result.rstrip()  # 移除末尾换行

    except Exception as e:
        logger.error(f"[fetch_external_data] 检索数据失败: {str(e)}", exc_info=True)
        return ""


@tool(description="无入参，无返回值，调用后触发中间件自动为煤矿挖掘机报告生成场景动态注入上下文信息，为后续提示词切换提供上下文支撑")
def fill_context_for_report():
    """
    触发上下文填充，为煤矿挖掘机报告生成提供支持
    """
    logger.info("[fill_context_for_report] 上下文填充已触发")
    return "fill_context_for_report已调用"


