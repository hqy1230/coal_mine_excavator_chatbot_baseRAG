# logger_utils.py
import logging
import os
import re
from datetime import datetime
from typing import Optional
from utils.path_tools import get_abs_path

# 把项目根目录 + 文件夹名 "logs" 拼在一起，变成一个完整路径，存到变量 LOG_ROOT 里。
LOG_ROOT = get_abs_path("logs")
# os.makedirs：Python 自带的创建文件夹的函数
# exist_ok=True：如果文件夹已经存在，就不报错，直接跳过
# 作用：确保 logs 文件夹一定存在，后续日志文件才能顺利写入，不会因为 “找不到目录” 而报错。
os.makedirs(LOG_ROOT, exist_ok=True)

# 日志格式配置定义一个日志格式模板，决定日志文件里的每一行长什么样。后面代码里用这个模板，就能生成统一格式的日志，方便你排查问题。
DEFAULT_LOG_FORMAT = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def mask_sensitive_data(text: str) -> str:
    """
    日志脱敏函数：隐藏API Key、手机号、邮箱等敏感信息
    :param text: 原始文本
    :return: 脱敏后的文本
    """
    if not isinstance(text, str):
        return text

    # 脱敏OpenAI/通义千问API Key（sk-开头）
    text = re.sub(r"sk-\w+", "sk-******", text)
    # 脱敏手机号
    text = re.sub(r"1[3-9]\d{9}", "1**********", text)
    # 脱敏邮箱
    text = re.sub(r"(\w+)@(\w+)\.(\w+)", r"\1****@\2.\3", text)
    # 脱敏密码/密钥（password/key=开头）
    text = re.sub(r"(password|key|secret)=[^& ]+", r"\1=******", text)
    return text


class SensitiveDataFilter(logging.Filter):
    """日志过滤器：自动脱敏日志中的敏感信息"""

    def filter(self, record: logging.LogRecord) -> bool:
        # 对日志消息脱敏
        if record.msg:
            record.msg = mask_sensitive_data(record.msg)
        # 对日志参数脱敏（如果有）
        if record.args:
            record.args = tuple(mask_sensitive_data(arg) for arg in record.args)
        return True


def get_logger(
        name: str = "agent",
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
        log_file: Optional[str] = None
) -> logging.Logger:
    """
    获取配置好的日志器（开箱即用）
    :param name: 日志器名称（建议按模块命名，如agent.tools/agent.rag/agent.llm）
    :param console_level: 控制台日志级别（默认INFO，开发时可设为DEBUG）
    :param file_level: 文件日志级别（默认DEBUG，记录详细信息）
    :param log_file: 自定义日志文件名（默认按日期生成：agent_20240121.log）
    :return: 配置完成的Logger对象
    """
    # 1. 创建/获取日志器
    logger = logging.getLogger(name)#获取或创建一个叫 name 的日志器。
    logger.setLevel(logging.DEBUG)  # 全局最低级别这样文件日志才能记录所有信息。
    logger.addFilter(SensitiveDataFilter())  # 添加脱敏过滤器比如防止 API 密钥、密码这类敏感信息被打印出来。

    # 避免重复添加Handler（多次导入时只配置一次）如果这个日志器已经配置过（比如多次导入 utils.py），就直接返回已有的对象，避免重复添加控制台 / 文件处理器，导致日志被打印 / 写入多次。
    if logger.handlers:
        return logger

    # 2. 配置控制台Handler（开发调试用）
    console_handler = logging.StreamHandler()#把日志输出到控制台（就是你运行代码时看到的黑框里）。
    console_handler.setLevel(console_level)#控制台只打印 console_level 及以上级别的日志（默认 INFO）。
    console_handler.setFormatter(DEFAULT_LOG_FORMAT)#用你之前定义好的日志格式，比如包含时间、文件名、行号。
    logger.addHandler(console_handler)

    # 3. 配置文件Handler（生产环境留存日志）
    if not log_file:
        log_file = os.path.join(LOG_ROOT, f"{name}_{datetime.now().strftime('%Y%m%d')}.log")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(file_level)
    file_handler.setFormatter(DEFAULT_LOG_FORMAT)
    logger.addHandler(file_handler)

    return logger


# 快捷获取默认Agent日志器
logger = get_logger("agent")
if __name__ == "__main__":
     logger.info("信息日志")
     logger.error("错误日志")
     logger.warning("警告日志")
     logger.debug("调试日志")