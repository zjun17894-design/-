# RAG Agent 执行流程详解（LangGraph + ChatQwen）

# 一、整体架构

这份 `rag_agent_service.py` 的本质是：

```text
一个基于 LangGraph 的 AI Agent 服务层
```

它负责：

- 大模型调用
- Tool Calling（工具调用）
- RAG 检索
- 流式输出
- 会话记忆
- MCP 工具扩展
- LangGraph 状态管理

整个系统链路：

```text
前端(Vue/React)
        ↓
FastAPI Router(chat.py)
        ↓
rag_agent_service.py
        ↓
LangGraph Agent
        ↓
ChatQwen
        ↓
Tool调用 / RAG检索
        ↓
流式返回结果
        ↓
SSE推送前端
```

---

# 二、用户请求完整生命周期

用户发送：

```text
“什么是 RAG？”
```

系统执行流程：

```text
用户请求
   ↓
FastAPI 接收请求
   ↓
调用 rag_agent_service.query_stream()
   ↓
初始化 LangGraph Agent
   ↓
构建 SystemMessage + HumanMessage
   ↓
加载历史会话(thread_id)
   ↓
Qwen 开始推理
   ↓
判断是否需要调用工具
   ↓
如果需要 → 调用 retrieve_knowledge
   ↓
Milvus 向量检索
   ↓
返回相关知识片段
   ↓
Qwen 基于知识生成回答
   ↓
Token 流式输出
   ↓
SSE 推送给前端
   ↓
前端逐字显示
```

---

# 三、文件顶部 Docstring

```python
"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""
```

这是模块级文档字符串。

Python 会自动保存为：

```python
module.__doc__
```

作用：

- IDE 提示
- 自动文档生成
- 团队协作说明

---

# 四、typing 类型系统

```python
from typing import Annotated, Any, AsyncGenerator, Dict, Sequence
```

这里是整个 LangGraph 状态系统的基础。

---

## 1. Annotated

```python
Annotated[Sequence[BaseMessage], add_messages]
```

含义：

```text
messages 是消息列表
更新时使用 add_messages 规则
```

即：

```text
追加消息
而不是覆盖消息
```

LangGraph 本质是：

```text
StateGraph（状态图）
```

每个节点都会返回状态更新。

例如：

NodeA：

```python
{
   "messages": [msg1]
}
```

NodeB：

```python
{
   "messages": [msg2]
}
```

如果没有 `add_messages`：

```text
msg1 会被覆盖
```

有了它：

```text
最终变成 [msg1, msg2]
```

---

## 2. Any

```python
Any
```

表示：

```text
放弃类型检查
```

AI Agent 系统数据结构动态变化很大：

- ToolCall
- Chunk
- Metadata
- StreamResult
- Message

很难完全静态定义。

---

## 3. AsyncGenerator

```python
AsyncGenerator
```

这是流式系统核心。

普通函数：

```python
def f():
    return 1
```

特点：

```text
一次执行结束
```

生成器：

```python
def f():
    yield 1
```

特点：

```text
执行到 yield 暂停
```

异步生成器：

```python
async def f():
    yield 1
```

支持：

```python
async for
```

适合：

```text
边收到 token
边向前端推送
```

---

# 五、LangChain Agent

```python
from langchain.agents import create_agent
```

这里是整个 AI Agent 的核心。

---

## create_agent 做了什么

很多人以为只是：

```text
创建一个对象
```

实际上它内部会自动构建：

```text
LLM节点
Tool节点
状态流
条件边
循环逻辑
Memory
```

本质：

```text
LangGraph 工作流
```

---

# 六、Message 消息体系

```python
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
```

LangChain 并不是字符串聊天。

而是：

```text
Message对象系统
```

---

## 1. BaseMessage

所有消息基类。

类似：

```text
Java里的 Object
```

---

## 2. HumanMessage

用户消息：

```python
HumanMessage(content="你好")
```

表示：

```text
role=user
```

---

## 3. SystemMessage

系统提示词：

```python
SystemMessage(content="你是AI助手")
```

作用：

```text
控制模型行为
```

---

## 4. RemoveMessage

这是 LangGraph 特殊消息。

作用：

```text
修改状态
```

不是给模型看的。

---

# 七、MemorySaver 会话记忆

```python
from langgraph.checkpoint.memory import MemorySaver
```

这是 LangGraph 的 Checkpoint 系统。

---

## 什么是 Checkpoint

本质：

```text
状态快照
```

每次 Agent 节点运行后：

```text
都会保存当前状态
```

---

## MemorySaver 本质

```python
self.checkpointer = MemorySaver()
```

内部类似：

```python
{
   thread_id: checkpoint
}
```

即：

```text
会话ID -> 历史消息
```

---

# 八、thread_id 会话隔离

```python
config_dict = {
    "configurable": {
        "thread_id": session_id
    }
}
```

这是多轮对话核心。

例如：

用户A：

```text
thread_id=abc
```

用户B：

```text
thread_id=xyz
```

二者历史不会串。

---

# 九、ChatQwen 初始化

```python
self.model = ChatQwen(
    model=self.model_name,
    api_key=config.dashscope_api_key,
    temperature=0.7,
    streaming=streaming,
)
```

这是大模型调用器。

---

## 1. model

指定模型：

```text
qwen-max
qwen-plus
```

---

## 2. temperature

控制随机性。

低 temperature：

```text
稳定、严谨
```

高 temperature：

```text
创造性更强
```

---

## 3. streaming=True

表示：

```text
开启真正 token 流式输出
```

不是：

```text
一次性生成全文
```

---

# 十、工具系统（Agent核心能力）

```python
self.tools = [retrieve_knowledge, get_current_time]
```

这里决定：

```text
Agent 能做什么
```

---

## retrieve_knowledge

RAG 检索工具。

作用：

```text
去向量数据库查知识
```

例如：

```text
Milvus
```

---

## get_current_time

获取真实时间。

因为：

```text
LLM 本身不知道当前时间
```

---

# 十一、Agent Tool Calling 执行流程

用户：

```text
现在几点？
```

执行流程：

```text
用户问题
   ↓
LLM推理
   ↓
发现自己不知道真实时间
   ↓
生成 ToolCall
   ↓
LangGraph 调用 get_current_time()
   ↓
工具返回真实时间
   ↓
LLM组织自然语言
   ↓
返回最终答案
```

---

# 十二、MCP 动态工具系统

```python
from app.agent.mcp_client import get_mcp_client_with_retry
```

MCP：

```text
Model Context Protocol
```

可以理解为：

```text
AI 插件协议
```

支持动态接入：

- GitHub
- 飞书
- 数据库
- 浏览器
- Jira
- 搜索引擎

---

# 十三、_initialize_agent()

```python
async def _initialize_agent(self):
```

这是真正 Agent 初始化过程。

---

## 为什么单独写？

因为：

```python
__init__
```

不能：

```python
await
```

而 MCP 工具获取：

```python
mcp_client = await get_mcp_client_with_retry()
```

必须异步。

所以：

```text
延迟初始化
```

---

## MCP 工具加载

```python
mcp_tools = await mcp_client.get_tools()
```

动态获取工具。

---

## 合并工具

```python
all_tools = self.tools + self.mcp_tools
```

最终 Agent 拥有：

```text
本地工具 + MCP工具
```

---

## create_agent()

```python
self.agent = create_agent(
    self.model,
    tools=all_tools,
    checkpointer=self.checkpointer,
)
```

这是整个系统最核心的一句。

它会自动构建：

```text
LLM节点
Tool节点
Memory
状态流转
循环
```

---

# 十四、系统提示词

```python
def _build_system_prompt(self) -> str:
```

构建 AI 行为规则。

作用：

```text
告诉模型：
你是谁
如何回答
什么时候调用工具
```

---

# 十五、query() 非流式调用

```python
async def query()
```

一次性返回完整结果。

---

## 执行流程

```text
初始化Agent
   ↓
构建消息
   ↓
恢复历史会话
   ↓
调用Agent
   ↓
执行Tool Calling
   ↓
生成最终答案
   ↓
返回完整文本
```

---

## ainvoke()

```python
result = await self.agent.ainvoke()
```

作用：

```text
完整执行整个 Agent
```

包括：

- LLM推理
- Tool调用
- 状态更新
- Memory保存

---

# 十六、query_stream() 流式输出

```python
async def query_stream()
```

这是整个 ChatGPT 打字效果核心。

---

## astream()

```python
async for token, metadata in self.agent.astream()
```

表示：

```text
边生成 token
边返回
```

---

# 十七、流式链路完整过程

```text
Qwen生成token
      ↓
LangChain Chunk
      ↓
LangGraph Stream
      ↓
query_stream yield
      ↓
FastAPI SSE
      ↓
浏览器 EventSource
      ↓
网页逐字显示
```

---

# 十八、content_blocks

```python
content_blocks = getattr(token, 'content_blocks', None)
```

这是 Qwen 原生流式结构。

不是 OpenAI 的：

```python
delta.content
```

而是：

```python
content_blocks
```

---

# 十九、yield 返回内容

```python
yield {
    "type": "content",
    "data": text_content,
    "node": node_name
}
```

这里：

```text
不是 return
而是 yield
```

因此：

```text
可以不断返回内容块
```

形成真正流式输出。

---

# 二十、get_session_history()

```python
def get_session_history(self, session_id: str)
```

作用：

```text
读取历史会话
```

---

## checkpointer.get()

```python
checkpoint_tuple = self.checkpointer.get(config)
```

获取：

```text
thread_id 对应状态
```

---

## 提取 messages

```python
messages = checkpoint_data.get("channel_values", {}).get("messages", [])
```

LangGraph 内部：

```text
所有消息都存储在 channel_values 中
```

---

# 二十一、clear_session()

```python
self.checkpointer.delete_thread(session_id)
```

作用：

```text
删除整个会话历史
```

即：

```text
清空 Memory
```

---

# 二十二、cleanup()

资源清理函数。

目前：

```text
MCP 客户端由全局管理器管理
```

因此无需手动关闭。

---

# 二十三、全局单例

```python
rag_agent_service = RagAgentService(streaming=True)
```

这是单例模式。

作用：

```text
整个项目只创建一个 Agent 服务实例
```

避免：

- 重复初始化模型
- 重复创建 Memory
- 重复创建 Tool
- 重复连接 MCP

---

# 二十四、整个系统真正技术栈

你这个项目实际上已经属于：

```text
企业级 AI Agent 架构
```

完整技术栈：

```text
FastAPI
+ LangChain
+ LangGraph
+ Qwen
+ MCP
+ RAG
+ SSE
+ Memory
+ Tool Calling
+ Milvus
```

已经不是普通聊天机器人。

而是：

```text
具备状态管理与工具调用能力的 AI Agent 系统
```

