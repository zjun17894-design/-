"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等。

这个文件的作用：
1. 创建 FastAPI 后端应用对象 app。
2. 配置应用启动和关闭时要执行的逻辑。
3. 配置跨域 CORS，允许前端访问后端接口。
4. 注册不同功能模块的 API 路由。
5. 挂载 static 静态文件目录。
6. 配置根路径 / 返回首页或 API 欢迎信息。
7. 支持直接运行当前文件启动后端服务。
"""

# FastAPI 是后端 Web 框架
# 用来创建 API 服务，例如 /api/chat、/api/upload、/docs 等接口
from fastapi import FastAPI

# CORSMiddleware 是 FastAPI/Starlette 提供的跨域中间件
# 作用：
# 允许前端页面从不同地址访问后端接口
#
# 例如：
# 前端：http://localhost:5173
# 后端：http://localhost:9900
#
# 这两个端口不同，浏览器会认为它们是“跨域”
# 所以需要配置 CORS
from fastapi.middleware.cors import CORSMiddleware

# StaticFiles 用于挂载静态文件目录
# 例如 HTML、CSS、JS、图片等文件
#
# 挂载后可以通过类似下面的地址访问：
# http://localhost:9900/static/xxx.js
from fastapi.staticfiles import StaticFiles

# FileResponse 用于直接返回一个文件
# 这里主要用于返回 static/index.html 首页文件
from fastapi.responses import FileResponse

# asynccontextmanager 用来定义异步上下文管理器
# 在 FastAPI 中常用于管理应用生命周期 lifespan
#
# 简单理解：
# yield 前面的代码：应用启动时执行
# yield 后面的代码：应用关闭时执行
from contextlib import asynccontextmanager

# os 是 Python 标准库
# 这里用于拼接文件路径、判断文件是否存在
import os

# 导入项目配置对象
# config 里面包含：
# app_name、app_version、debug、host、port 等配置
from app.config import config

# 导入 Loguru 日志对象
# 用于打印启动日志、关闭日志、Milvus 连接日志等
from loguru import logger

# 导入不同 API 模块
#
# chat：对话相关接口
# health：健康检查接口
# file：文件管理相关接口
# aiops：智能运维相关接口
#
# 每个模块里通常都有一个 router 对象
# 后面通过 app.include_router(...) 注册到主应用中
from app.api import chat, health, file, aiops

# 导入 Milvus 管理器
# 用于在应用启动时连接 Milvus，在应用关闭时关闭连接
from app.core.milvus_client import milvus_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理

    lifespan 是 FastAPI 推荐的生命周期管理方式。

    它负责管理：
    1. 应用启动时要做什么
    2. 应用关闭时要做什么

    在这个项目中：
    - 启动时：打印启动信息，并连接 Milvus
    - 关闭时：关闭 Milvus 连接，并打印关闭日志

    参数：
        app: FastAPI 应用对象，由 FastAPI 自动传入
    """

    # =========================
    # 应用启动时执行
    # =========================

    # 打印一行分隔线，让启动日志更清晰
    logger.info("=" * 60)

    # 打印应用名称和版本号
    #
    # 例如：
    # 🚀 SuperBizAgent v1.0.0 启动中...
    logger.info(f"🚀 {config.app_name} v{config.app_version} 启动中...")

    # 打印当前运行环境
    #
    # 如果 config.debug=True，显示“开发”
    # 如果 config.debug=False，显示“生产”
    logger.info(f"📝 环境: {'开发' if config.debug else '生产'}")

    # 打印服务监听地址
    #
    # 例如：
    # http://0.0.0.0:9900
    #
    # 注意：
    # 0.0.0.0 表示监听所有网卡，不一定是浏览器访问地址。
    # 本机访问通常用：
    # http://localhost:9900
    logger.info(f"🌐 监听地址: http://{config.host}:{config.port}")

    # 打印 Swagger API 文档地址
    #
    # FastAPI 会自动生成接口文档
    # 一般可以访问：
    # http://localhost:9900/docs
    logger.info(f"📚 API 文档: http://{config.host}:{config.port}/docs")

    # =========================
    # 连接 Milvus
    # =========================

    # 打印正在连接 Milvus 的日志
    logger.info("🔌 正在连接 Milvus...")

    # 调用 Milvus 管理器的 connect 方法
    #
    # 这里通常会做：
    # 1. 根据配置连接 Milvus 服务
    # 2. 检查连接状态
    # 3. 准备 collection 等资源
    #
    # 具体逻辑要看 app/core/milvus_client.py
    milvus_manager.connect()

    # 如果 connect 没有抛异常，说明连接成功
    logger.info("✅ Milvus 连接成功")

    # 再打印一行分隔线，表示启动过程结束
    logger.info("=" * 60)

    # =========================
    # yield 是生命周期分界点
    # =========================
    #
    # yield 前面：
    #     应用启动时执行
    #
    # yield 这里：
    #     FastAPI 应用开始正式运行，等待请求
    #
    # yield 后面：
    #     应用关闭时执行
    yield

    # =========================
    # 应用关闭时执行
    # =========================

    # 打印正在关闭 Milvus 的日志
    logger.info("🔌 正在关闭 Milvus 连接...")

    # 关闭 Milvus 连接，释放资源
    milvus_manager.close()

    # 打印应用关闭日志
    logger.info(f"👋 {config.app_name} 关闭")


# =========================
# 创建 FastAPI 应用
# =========================

# app 是整个后端服务的核心对象
#
# 后面所有路由、中间件、静态文件、生命周期函数
# 都是绑定到这个 app 对象上的
app = FastAPI(
    # API 文档中的项目标题
    #
    # 会显示在 /docs 页面顶部
    title=config.app_name,

    # API 版本号
    #
    # 会显示在 /docs 页面中
    version=config.app_version,

    # API 描述信息
    #
    # 会显示在 /docs 页面中
    description="基于 LangChain 的智能oncall运维系统",

    # 注册生命周期管理函数
    #
    # 应用启动时会执行 lifespan 中 yield 前面的代码
    # 应用关闭时会执行 lifespan 中 yield 后面的代码
    lifespan=lifespan,
)


# =========================
# 配置 CORS 跨域
# =========================

# 给 FastAPI 应用添加 CORS 中间件
#
# 中间件可以理解为：
# 请求进入接口之前、响应返回浏览器之前，会经过的一层处理逻辑
app.add_middleware(
    # 使用跨域中间件
    CORSMiddleware,

    # 允许哪些前端来源访问后端
    #
    # ["*"] 表示允许所有来源
    #
    # 开发阶段方便，但生产环境不推荐这么写。
    #
    # 生产环境建议改成具体前端地址，例如：
    # allow_origins=[
    #     "http://localhost:5173",
    #     "https://your-frontend-domain.com",
    # ]
    allow_origins=["*"],

    # 是否允许携带 Cookie、Authorization 等认证信息
    #
    # True 表示允许
    allow_credentials=True,

    # 允许哪些 HTTP 方法
    #
    # ["*"] 表示允许所有方法：
    # GET、POST、PUT、DELETE、PATCH、OPTIONS 等
    allow_methods=["*"],

    # 允许哪些请求头
    #
    # ["*"] 表示允许所有请求头
    # 例如 Content-Type、Authorization 等
    allow_headers=["*"],
)


# =========================
# 注册 API 路由
# =========================
#
# 路由可以理解为接口地址。
#
# 例如：
# /health
# /api/chat
# /api/files
# /api/aiops/xxx
#
# 不同模块负责不同类型的接口，
# 主应用这里只负责统一注册。

# 注册健康检查路由
#
# 因为这里没有写 prefix，
# 所以 health.router 里面定义的路径是什么，最终就是什么。
#
# 例如 health.py 中如果写：
# @router.get("/health")
#
# 最终接口就是：
# GET /health
app.include_router(
    health.router,
    tags=["健康检查"],
)

# 注册对话路由
#
# prefix="/api" 表示给 chat.router 里的所有接口统一加上 /api 前缀。
#
# 例如 chat.py 中如果写：
# @router.post("/chat")
#
# 最终接口就是：
# POST /api/chat
app.include_router(
    chat.router,
    prefix="/api",
    tags=["对话"],
)

# 注册文件管理路由
#
# 例如 file.py 中如果写：
# @router.post("/files/upload")
#
# 最终接口就是：
# POST /api/files/upload
app.include_router(
    file.router,
    prefix="/api",
    tags=["文件管理"],
)

# 注册 AIOps 智能运维路由
#
# 例如 aiops.py 中如果写：
# @router.post("/aiops/analyze")
#
# 最终接口就是：
# POST /api/aiops/analyze
app.include_router(
    aiops.router,
    prefix="/api",
    tags=["AIOps智能运维"],
)


# =========================
# 挂载静态文件
# =========================

# 静态文件目录
#
# 这里表示当前项目下的 static 文件夹
# 一般里面会放：
# - index.html
# - CSS 文件
# - JS 文件
# - 图片文件
static_dir = "static"

# 把 static 文件夹挂载到 /static 路径
#
# 例如：
# static/logo.png
#
# 浏览器可以通过：
# http://localhost:9900/static/logo.png
#
# 访问到这个文件
app.mount(
    "/static",
    StaticFiles(directory=static_dir),
    name="static",
)


@app.get("/")
async def root():
    """返回首页

    当用户访问根路径 / 时，会执行这个函数。

    例如访问：
        http://localhost:9900/

    逻辑：
    1. 优先查找 static/index.html
    2. 如果存在，就返回这个 HTML 页面
    3. 如果不存在，就返回一个 JSON 欢迎信息
    """

    # 拼接首页文件路径
    #
    # static_dir = "static"
    # 所以 index_path = "static/index.html"
    index_path = os.path.join(static_dir, "index.html")

    # 判断 static/index.html 是否存在
    if os.path.exists(index_path):

        # 如果存在，直接把这个 HTML 文件返回给浏览器
        #
        # 浏览器收到后会显示前端页面
        return FileResponse(index_path)

    # 如果没有 static/index.html
    # 就返回一个 JSON 格式的 API 欢迎信息
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs",
    }


# =========================
# 本文件直接运行时启动服务
# =========================

# 只有当你直接运行这个文件时，下面代码才会执行。
#
# 例如：
# python app/main.py
#
# 如果是被 uvicorn 通过 app.main:app 导入，
# 这部分不会执行。
if __name__ == "__main__":

    # uvicorn 是运行 FastAPI 的 ASGI 服务器
    #
    # FastAPI 本身只是 Web 框架，
    # 真正负责监听端口、接收请求的是 uvicorn
    import uvicorn

    # 启动 FastAPI 服务
    uvicorn.run(
        # 指定应用路径
        #
        # "app.main:app" 的意思是：
        # app/main.py 文件中的 app 对象
        #
        # 前面的 app.main 是模块路径
        # 后面的 app 是 FastAPI 应用实例变量名
        "app.main:app",

        # 服务监听地址
        #
        # 来自配置文件 config.host
        # 默认是 0.0.0.0
        host=config.host,

        # 服务端口
        #
        # 来自配置文件 config.port
        # 默认是 9900
        port=config.port,

        # 是否开启自动重载
        #
        # reload=True：
        #     代码修改后自动重启服务，适合开发环境
        #
        # reload=False：
        #     不自动重启，适合生产环境
        reload=config.debug,

        # uvicorn 自身日志级别
        log_level="info",
    )