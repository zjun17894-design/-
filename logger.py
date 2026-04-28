"""日志配置模块

使用 Loguru 配置应用日志。

这个文件的作用：
1. 统一配置整个项目的日志输出格式。
2. 同时支持控制台输出和文件输出。
3. 控制台日志方便开发时查看。
4. 文件日志方便后期排查问题。
5. 根据 config.debug 控制日志详细程度。
"""

# sys 是 Python 标准库
# 这里主要用 sys.stdout，表示把日志输出到控制台
import sys

# 从 loguru 中导入 logger
# logger 是 Loguru 提供的核心日志对象
# 后面所有日志输出都通过它完成
from loguru import logger

# 从项目配置模块中导入 config
# config 里面保存了项目配置，例如 debug=True/False
from app.config import config


def setup_logger():
    """配置日志系统

    这个函数负责初始化全局日志系统。

    它主要做三件事：
    1. 移除 Loguru 默认的日志处理器
    2. 添加控制台日志输出
    3. 添加文件日志输出

    控制台日志：
        适合开发时实时查看程序运行情况。

    文件日志：
        适合保存历史日志，方便后续排查问题。
    """

    # =========================
    # 1. 移除默认日志处理器
    # =========================

    # Loguru 默认会自带一个日志输出处理器
    # 如果不移除，后面再 add 新的处理器时，可能会出现日志重复打印的问题
    #
    # 所以通常第一步先执行 logger.remove()
    logger.remove()

    # =========================
    # 2. 添加控制台日志输出
    # =========================

    # logger.add() 用来添加一个日志输出目标
    #
    # 这里第一个参数是 sys.stdout
    # 表示日志输出到控制台，也就是 PyCharm / 命令行窗口
    logger.add(
        # 输出目标：控制台标准输出
        sys.stdout,

        # 控制台日志格式
        #
        # <green>...</green> 表示绿色
        # <level>...</level> 表示根据日志级别自动着色
        # <cyan>...</cyan> 表示青色
        #
        # {time:YYYY-MM-DD HH:mm:ss}
        # 表示日志时间，例如：2026-04-28 15:30:12
        #
        # {level: <8}
        # 表示日志级别，例如 INFO、DEBUG、ERROR
        # <8 表示左对齐，占 8 个字符宽度，方便排版整齐
        #
        # {module}
        # 表示当前日志来自哪个模块，也就是哪个 Python 文件
        #
        # {function}
        # 表示当前日志来自哪个函数
        #
        # {line}
        # 表示当前日志出现在第几行
        #
        # {message}
        # 表示真正打印的日志内容
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),

        # 控制日志级别
        #
        # 如果 config.debug=True：
        #     日志级别为 DEBUG，会打印更详细的信息
        #
        # 如果 config.debug=False：
        #     日志级别为 INFO，只打印普通运行信息及以上级别
        #
        # 日志级别从低到高大概是：
        # DEBUG < INFO < WARNING < ERROR < CRITICAL
        level="DEBUG" if config.debug else "INFO",

        # 是否启用彩色日志
        #
        # True 表示控制台会显示彩色日志
        # 这只影响控制台显示，不影响文件日志
        colorize=True,

        # 是否显示完整异常栈信息
        #
        # True 表示如果程序报错，会显示更完整的调用链
        # 方便定位错误是从哪里一路传过来的
        backtrace=True,

        # 是否在异常日志中显示变量值
        #
        # config.debug=True 时开启
        # 这样调试时可以看到报错位置附近的变量内容
        #
        # 注意：
        # diagnose=True 可能会暴露变量值，
        # 如果变量里有 API Key、密码等敏感信息，生产环境不建议开启
        diagnose=config.debug,
    )

    # =========================
    # 3. 添加文件日志输出
    # =========================

    # 再添加一个日志输出目标：日志文件
    #
    # 这样日志不仅会显示在控制台，还会保存到 logs 文件夹里
    logger.add(
        # 日志文件路径
        #
        # logs/app_{time:YYYY-MM-DD}.log
        # 表示按日期生成日志文件
        #
        # 例如：
        # logs/app_2026-04-28.log
        #
        # 如果 logs 文件夹不存在，Loguru 通常会自动创建
        "logs/app_{time:YYYY-MM-DD}.log",

        # 日志轮转规则
        #
        # rotation="00:00" 表示每天 0 点切割一个新的日志文件
        #
        # 例如：
        # 今天写入 app_2026-04-28.log
        # 到明天 0 点后写入 app_2026-04-29.log
        rotation="00:00",

        # 日志保留时间
        #
        # retention="7 days" 表示只保留最近 7 天的日志
        # 超过 7 天的旧日志会被自动清理
        retention="7 days",

        # 日志压缩方式
        #
        # compression="zip" 表示旧日志会被压缩成 zip 文件
        # 这样可以节省磁盘空间
        compression="zip",

        # 文件编码
        #
        # utf-8 可以避免中文日志乱码
        encoding="utf-8",

        # 是否异步写入日志
        #
        # enqueue=True 表示日志先进入队列，再由后台线程写入文件
        # 好处是减少文件 IO 对主程序的阻塞
        #
        # 在 Web 服务、Agent 服务中，一般建议开启
        enqueue=True,

        # 文件日志中也显示完整异常栈
        # 方便后期排查线上错误
        backtrace=True,

        # 文件日志中显示变量值
        #
        # 这里写的是 True，表示无论 debug 是否开启，文件日志都会记录变量诊断信息
        #
        # 注意：
        # 如果是生产环境，建议改成 diagnose=config.debug
        # 避免把 API Key、密码、Token 等敏感变量写进日志文件
        diagnose=True,

        # 文件日志级别
        #
        # level="INFO" 表示文件里只记录 INFO 及以上级别日志
        #
        # DEBUG 日志不会写入文件
        # INFO / WARNING / ERROR / CRITICAL 会写入文件
        level="INFO",

        # 文件日志格式
        #
        # 文件日志一般不需要颜色标签，
        # 所以这里没有 <green>、<level>、<cyan> 这些彩色标记
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level: <8} | "
            "{module}.{function}:{line} | "
            "{message}"
        ),
    )


# =========================
# 初始化日志系统
# =========================

# 文件被导入时，立即执行 setup_logger()
#
# 也就是说，只要其他模块执行：
#
# from app.logger import logger
#
# 或者导入了这个日志配置模块，
# 日志系统就会自动完成初始化。
setup_logger()