"""配置管理模块

使用 Pydantic Settings 实现类型安全的配置管理。

这个文件的作用：
1. 统一保存项目中的各种配置，例如模型、Milvus、RAG、MCP 等配置。
2. 自动从 .env 文件或系统环境变量中读取配置。
3. 对配置进行类型转换，例如端口转成 int，debug 转成 bool。
4. 给项目其他模块提供一个统一的 config 对象。
"""

# Dict：用于标注字典类型
# Any：表示任意类型
# 例如 Dict[str, Any] 表示：key 是字符串，value 可以是任意类型
from typing import Dict, Any

# BaseSettings：
# Pydantic 提供的配置基类，继承它之后可以自动从 .env / 环境变量读取配置
#
# SettingsConfigDict：
# 用来配置 BaseSettings 的读取规则，例如读取哪个 .env 文件、是否区分大小写等
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置类

    这个类用于集中管理整个项目的配置。

    例如：
    - 应用名称、端口
    - DashScope API Key 和模型名称
    - Milvus 向量数据库地址
    - RAG 检索参数
    - MCP 服务启动参数

    其他模块只需要导入全局 config 对象，就可以使用这些配置。
    """

    # 配置 Pydantic Settings 的读取规则
    model_config = SettingsConfigDict(
        # 指定从项目根目录下的 .env 文件读取配置
        # 例如 .env 中可以写：
        # DASHSCOPE_API_KEY=你的key
        # MILVUS_HOST=localhost
        env_file=".env",

        # 指定 .env 文件的编码格式，防止中文注释或中文内容乱码
        env_file_encoding="utf-8",

        # 是否区分环境变量大小写
        # False 表示不区分大小写
        # 例如 dashscope_api_key 可以匹配 DASHSCOPE_API_KEY
        case_sensitive=False,

        # 如果 .env 中出现了当前 Settings 类没有定义的字段，直接忽略，不报错
        # 例如 .env 中多写了 TEST_KEY=123，也不会影响程序启动
        extra="ignore",
    )

    # =========================
    # 应用基础配置
    # =========================

    # 应用名称
    # 可以用于日志、接口返回、系统展示等场景
    app_name: str = "SuperBizAgent"

    # 应用版本号
    # 可以用于 /health 接口或系统信息展示
    app_version: str = "1.0.0"

    # 是否开启调试模式
    # False：生产/正常模式
    # True：开发调试模式，通常会输出更多调试信息
    debug: bool = False

    # 服务监听地址
    # 0.0.0.0 表示允许外部设备访问这个服务
    # 如果写成 127.0.0.1，则通常只能本机访问
    host: str = "0.0.0.0"

    # 服务端口
    # 后端服务默认运行在 http://localhost:9900
    port: int = 9900

    # =========================
    # DashScope 配置
    # =========================

    # DashScope API Key
    # 默认是空字符串，实际项目运行时应该从 .env 中读取
    #
    # .env 示例：
    # DASHSCOPE_API_KEY=你的真实key
    #
    # 注意：
    # 不建议把真实 API Key 直接写在代码里
    dashscope_api_key: str = ""

    # DashScope 聊天模型名称
    # 用于普通对话、Agent 推理、RAG 最终回答生成等场景
    dashscope_model: str = "qwen-max"

    # DashScope 向量模型名称
    # 用于把文本转换成向量，主要服务于 RAG 检索
    #
    # text-embedding-v4 是 embedding 模型，不是聊天模型
    # 它的作用是：文本 -> 向量
    dashscope_embedding_model: str = "text-embedding-v4"

    # =========================
    # Milvus 向量数据库配置
    # =========================

    # Milvus 服务地址
    # localhost 表示 Milvus 跑在本机
    milvus_host: str = "localhost"

    # Milvus 服务端口
    # Milvus 默认常用端口是 19530
    milvus_port: int = 19530

    # 连接 Milvus 的超时时间
    # 单位：毫秒
    # 10000 毫秒 = 10 秒
    milvus_timeout: int = 10000

    # =========================
    # RAG 配置
    # =========================

    # RAG 检索时返回最相关的前 k 条文档
    #
    # 例如 rag_top_k = 3 表示：
    # 用户提问后，从向量数据库中召回最相似的 3 段文本
    rag_top_k: int = 3

    # RAG 生成回答时使用的模型
    # 检索出来的资料会和用户问题一起交给这个模型生成最终回答
    rag_model: str = "qwen-max"

    # =========================
    # 文档分块配置
    # =========================

    # 每个文本块的最大长度
    #
    # RAG 处理长文档时，不会把整篇文档直接存入向量库，
    # 而是先切成多个小块，再分别向量化和存储。
    chunk_max_size: int = 800

    # 文本块之间的重叠长度
    #
    # 作用：
    # 防止切块时把完整语义切断。
    #
    # 例如：
    # 第 1 块结尾和第 2 块开头会有 100 个字符的重叠。
    chunk_overlap: int = 100

    # =========================
    # MCP 服务配置
    # =========================
    #
    # MCP 可以理解为给 Agent / 大模型提供外部工具能力的服务。
    #
    # 当前配置了两个 MCP 服务：
    # 1. cls
    # 2. monitor
    #
    # 它们具体提供什么工具能力，需要看：
    # mcp_servers/cls_server.py
    # mcp_servers/monitor_server.py

    # CLS MCP 服务的通信方式
    #
    # stdio 表示：
    # 通过标准输入/标准输出和本地子进程通信。
    #
    # 简单理解：
    # 程序会启动一个本地 Python 脚本作为 MCP 服务。
    mcp_cls_transport: str = "stdio"

    # 启动 CLS MCP 服务时使用的 Python 解释器
    #
    # Windows 项目虚拟环境中通常是：
    # .venv/Scripts/python.exe
    #
    # Linux / macOS 通常是：
    # .venv/bin/python
    mcp_cls_command: str = ".venv/Scripts/python.exe"

    # 启动 CLS MCP 服务时传入的脚本路径
    #
    # 最终类似于执行：
    # .venv/Scripts/python.exe mcp_servers/cls_server.py
    mcp_cls_args: str = "mcp_servers/cls_server.py"

    # CLS MCP 服务的 URL
    #
    # 如果服务不是 stdio 模式，而是通过远程地址连接，
    # 可以在这里配置 URL。
    #
    # 默认空字符串表示没有使用 URL。
    mcp_cls_url: str = ""

    # Monitor MCP 服务的通信方式
    # 默认也是 stdio，本地启动脚本通信
    mcp_monitor_transport: str = "stdio"

    # 启动 Monitor MCP 服务时使用的 Python 解释器
    mcp_monitor_command: str = ".venv/Scripts/python.exe"

    # 启动 Monitor MCP 服务时传入的脚本路径
    #
    # 最终类似于执行：
    # .venv/Scripts/python.exe mcp_servers/monitor_server.py
    mcp_monitor_args: str = "mcp_servers/monitor_server.py"

    # Monitor MCP 服务的 URL
    #
    # 如果 Monitor 服务通过远程 URL 连接，可以在 .env 中配置：
    # MCP_MONITOR_URL=http://localhost:8004/mcp
    mcp_monitor_url: str = ""

    @property
    def mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """获取完整的 MCP 服务器配置

        这个方法会把上面分散的 MCP 配置项，组装成一个完整字典。

        使用方式：
            config.mcp_servers

        注意：
            因为加了 @property，所以访问时不用写括号。
            正确：config.mcp_servers
            错误：config.mcp_servers()

        返回结构大概如下：
            {
                "cls": {
                    "transport": "stdio",
                    "command": ".venv/Scripts/python.exe",
                    "args": ["mcp_servers/cls_server.py"]
                },
                "monitor": {
                    "transport": "stdio",
                    "command": ".venv/Scripts/python.exe",
                    "args": ["mcp_servers/monitor_server.py"]
                }
            }
        """

        # 创建一个空字典，用来保存所有 MCP 服务配置
        config = {}

        # =========================
        # CLS 服务器配置
        # =========================

        # 先创建 CLS 服务的基础配置
        # 默认只放入 transport
        #
        # 例如：
        # {
        #     "transport": "stdio"
        # }
        cls_config: Dict[str, Any] = {
            "transport": self.mcp_cls_transport
        }

        # 如果 CLS 服务使用 stdio 模式
        # 就需要配置 command 和 args
        if self.mcp_cls_transport == "stdio":

            # 配置启动 CLS 服务的命令
            #
            # 默认：
            # .venv/Scripts/python.exe
            cls_config["command"] = self.mcp_cls_command

            # 配置启动 CLS 服务时传入的参数
            #
            # 注意：
            # 这里要写成列表形式
            #
            # 最终结果：
            # "args": ["mcp_servers/cls_server.py"]
            cls_config["args"] = [self.mcp_cls_args]

        # 如果不是 stdio 模式，并且配置了 URL
        # 就把 URL 加入配置
        elif self.mcp_cls_url:
            cls_config["url"] = self.mcp_cls_url

        # 把 CLS 服务配置放入总配置字典
        #
        # 最终变成：
        # config["cls"] = {...}
        config["cls"] = cls_config

        # =========================
        # Monitor 服务器配置
        # =========================

        # 创建 Monitor 服务的基础配置
        monitor_config: Dict[str, Any] = {
            "transport": self.mcp_monitor_transport
        }

        # 如果 Monitor 服务使用 stdio 模式
        # 就配置本地启动命令和脚本路径
        if self.mcp_monitor_transport == "stdio":

            # 配置启动 Monitor 服务的 Python 解释器
            monitor_config["command"] = self.mcp_monitor_command

            # 配置启动 Monitor 服务时传入的脚本路径
            monitor_config["args"] = [self.mcp_monitor_args]

        # 如果不是 stdio 模式，并且配置了 URL
        # 就使用 URL 连接 Monitor 服务
        elif self.mcp_monitor_url:
            monitor_config["url"] = self.mcp_monitor_url

        # 把 Monitor 服务配置放入总配置字典
        config["monitor"] = monitor_config

        # 返回完整 MCP 服务配置
        return config


# =========================
# 全局配置实例
# =========================
#
# 这里创建一个 Settings 对象。
#
# 创建时会自动：
# 1. 读取默认值
# 2. 读取 .env 文件
# 3. 读取系统环境变量
# 4. 用外部配置覆盖默认值
#
# 其他模块可以这样使用：
#
# from app.config import config
#
# print(config.dashscope_model)
# print(config.milvus_host)
# print(config.mcp_servers)
config = Settings()