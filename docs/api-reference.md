# API 参考

完整的 API 文档，包含所有公共类和方法。

## SuperReActAgent

主 Agent 类，实现 ReAct 架构。

### 初始化

```python
SuperReActAgent(
    agent_config: SuperAgentConfig,
    context_manager: Optional[ContextManager] = None,
    qa_handler: Optional[QAHandler] = None,
    tool_call_handler: Optional[ToolCallHandler] = None,
)
```

**参数**:
- `agent_config` (SuperAgentConfig): Agent 配置
- `context_manager` (ContextManager, optional): 上下文管理器
- `qa_handler` (QAHandler, optional): QA 处理器
- `tool_call_handler` (ToolCallHandler, optional): 工具调用处理器

### 主要方法

#### process_input

```python
async def process_input(
    self,
    question: str,
    **kwargs
) -> str
```

处理用户输入，执行 ReAct 循环。

**参数**:
- `question` (str): 用户问题
- `**kwargs`: 额外参数

**返回**:
- `str`: 最终答案

**示例**:

```python
result = await agent.process_input(
    "What is the capital of France?"
)
```

#### reset

```python
def reset(self) -> None
```

重置 Agent 状态（清除历史记录）。

#### cleanup

```python
async def cleanup(self) -> None
```

清理资源，关闭连接。

**示例**:

```python
# 使用完后清理
await agent.cleanup()
```

## SuperAgentConfig

Agent 配置类。

### 属性

```python
class SuperAgentConfig(ReActAgentConfig):
    agent_type: str = "main"                              # Agent 类型
    enable_question_hints: bool = False                   # 启用问题提示
    enable_extract_final_answer: bool = False             # 启用答案提取
    open_api_key: Optional[str] = None                    # OpenAI API 密钥
    reasoning_model: str = "o3"                           # 推理模型
    enable_context_limit_retry: bool = True               # 启用上下文重试
    keep_tool_result: int = -1                            # 保留工具结果数
    max_tool_calls_per_turn: int = 5                      # 每轮最大工具调用
    enable_todo_plan: bool = True                         # 启用计划追踪
    sub_agent_configs: Dict[str, SuperAgentConfig] = {}   # 子 Agent 配置
    task_guidance: str = ""                               # 任务指导
```

### 示例

```python
from agent.super_config import SuperAgentConfig

config = SuperAgentConfig(
    agent_type="main",
    description="Research agent",
    enable_question_hints=True,
    enable_extract_final_answer=True,
    max_iteration=15,
)
```

## SuperAgentFactory

Agent 配置工厂类。

### create_main_agent_config

```python
@staticmethod
def create_main_agent_config(
    agent_id: str,
    api_key: str,
    api_base: str,
    model_name: str,
    model_provider: str = "openrouter",
    description: str = "",
    max_iteration: int = 10,
    enable_question_hints: bool = False,
    enable_extract_final_answer: bool = False,
    reasoning_model: str = "o3",
    open_api_key: Optional[str] = None,
    task_guidance: str = "",
    sub_agent_configs: Optional[Dict[str, Any]] = None,
    **kwargs
) -> SuperAgentConfig
```

创建主 Agent 配置。

**参数**:
- `agent_id` (str): Agent ID
- `api_key` (str): API 密钥
- `api_base` (str): API 基础 URL
- `model_name` (str): 模型名称
- `model_provider` (str): 模型提供商，默认 "openrouter"
- `description` (str): Agent 描述
- `max_iteration` (int): 最大迭代次数
- `enable_question_hints` (bool): 启用问题提示
- `enable_extract_final_answer` (bool): 启用答案提取
- `reasoning_model` (str): 推理模型名称
- `open_api_key` (str, optional): OpenAI API 密钥
- `task_guidance` (str): 任务指导文本
- `sub_agent_configs` (Dict): 子 Agent 配置

**返回**:
- `SuperAgentConfig`: 配置对象

### create_sub_agent_config

```python
@staticmethod
def create_sub_agent_config(
    agent_name: str,
    agent_type: str,
    description: str = "",
    max_iteration: int = 10,
    **kwargs
) -> SuperAgentConfig
```

创建子 Agent 配置。

## ContextManager

上下文管理器，管理对话历史。

### 初始化

```python
ContextManager(
    llm: Optional[OpenRouterLLM] = None,
    max_history_length: int = 100
)
```

### 主要方法

#### add_message

```python
def add_message(
    self,
    role: str,
    content: Any,
    **kwargs
) -> None
```

添加消息到历史记录。

**参数**:
- `role` (str): 角色 (user/assistant/tool/system)
- `content` (Any): 消息内容
- `**kwargs`: 额外字段

#### get_history

```python
def get_history(self) -> List[Dict[str, Any]]
```

获取完整历史记录。

**返回**:
- `List[Dict]`: 消息列表

#### clear

```python
def clear(self) -> None
```

清空历史记录。

### 示例

```python
from agent.context_manager import ContextManager

context = ContextManager(max_history_length=50)

# 添加消息
context.add_message(
    role="user",
    content="Hello!"
)

# 获取历史
history = context.get_history()
```

## ToolCallHandler

工具调用处理器。

### 初始化

```python
ToolCallHandler(
    sub_agents: Optional[Dict[str, Any]] = None
)
```

### 主要方法

#### create_sub_agent_tool

```python
def create_sub_agent_tool(
    self,
    agent_name: str,
    sub_agent: Any
) -> LocalFunction
```

为子 Agent 创建工具包装器。

**参数**:
- `agent_name` (str): Agent 名称
- `sub_agent` (SuperReActAgent): 子 Agent 实例

**返回**:
- `LocalFunction`: 工具函数

#### execute_tool

```python
async def execute_tool(
    self,
    tool_name: str,
    arguments: Dict[str, Any]
) -> Any
```

执行工具调用。

## QAHandler

QA 处理器，用于推理模型调用。

### 初始化

```python
QAHandler(
    api_key: str,
    enable_message_ids: bool = True,
    reasoning_model: str = "o3"
)
```

### 主要方法

#### extract_hints

```python
@retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(1))
async def extract_hints(
    self,
    question: str
) -> str
```

提取问题提示。

**参数**:
- `question` (str): 用户问题

**返回**:
- `str`: 提取的提示

#### determine_answer_type

```python
async def determine_answer_type(
    self,
    question: str
) -> Tuple[str, str]
```

确定答案类型。

**返回**:
- `Tuple[str, str]`: (类型, 描述)

#### extract_final_answer

```python
async def extract_final_answer(
    self,
    conversation: str,
    answer_type: str
) -> str
```

提取最终答案。

## OpenRouterLLM

OpenRouter LLM 客户端。

### 初始化

```python
OpenRouterLLM(
    api_key: str,
    api_base: str = "https://openrouter.ai/api/v1",
    model_name: str = "anthropic/claude-3.5-sonnet",
    max_retries: int = 3,
    timeout: int = 600,
    temperature: float = 0.1,
    top_p: float = 1.0,
    max_tokens: int = 4096,
    **kwargs
)
```

### 主要方法

#### generate

```python
async def generate(
    self,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]] = None,
    **kwargs
) -> AIMessage
```

生成响应。

**参数**:
- `messages` (List[Dict]): 消息列表
- `tools` (List[Dict], optional): 工具定义

**返回**:
- `AIMessage`: AI 消息对象

#### stream_generate

```python
async def stream_generate(
    self,
    messages: List[Dict[str, Any]],
    **kwargs
) -> AsyncIterator[AIMessageChunk]
```

流式生成响应。

### 示例

```python
from llm.openrouter_llm import OpenRouterLLM

llm = OpenRouterLLM(
    api_key="your_key",
    model_name="anthropic/claude-3.7-sonnet",
    temperature=0.1
)

messages = [
    {"role": "user", "content": "Hello!"}
]

response = await llm.generate(messages)
print(response.content)
```

## AgentConstraints

Agent 约束配置。

```python
class AgentConstraints(BaseModel):
    max_iteration: int = 10              # 最大迭代次数
    max_tool_calls_per_turn: int = 5     # 每轮最大工具调用
    reserved_max_chat_rounds: int = 40   # 保留的最大对话轮数
```

### 示例

```python
from agent.super_config import AgentConstraints

constraints = AgentConstraints(
    max_iteration=15,
    max_tool_calls_per_turn=3
)
```

## 类型定义

### Message 类型

```python
Message = Dict[str, Any]
# {
#     "role": str,      # "user", "assistant", "tool", "system"
#     "content": str,
#     "tool_calls": Optional[List[Dict]],
#     "tool_call_id": Optional[str],
#     ...
# }
```

### ToolCall 类型

```python
ToolCall = Dict[str, Any]
# {
#     "id": str,
#     "type": "function",
#     "function": {
#         "name": str,
#         "arguments": str  # JSON 字符串
#     }
# }
```

## 异常类

### ContextLimitError

上下文限制超出错误。

```python
from llm.openrouter_llm import ContextLimitError

try:
    response = await llm.generate(messages)
except ContextLimitError as e:
    # 处理上下文超限
    messages = trim_messages(messages)
    response = await llm.generate(messages)
```

## 实用函数

### 工具创建辅助函数

```python
# 创建 MCP 工具调用包装器
def _make_mcp_call_coroutine(server_name: str, tool_name: str):
    """为 MCP 工具生成协程函数"""
    ...

# 标准化 MCP 服务器配置
def _normalize_mcp_server_config(
    client_type: str,
    params: Any
) -> Tuple[str, Dict]:
    """标准化配置参数"""
    ...
```

## 常量

```python
# 传输方式
AUTO_BROWSER_TRANSPORT = "sse"  # 或 "stdio"

# 默认端口
DEFAULT_SSE_PORT = 8930

# 超时设置
DEFAULT_TIMEOUT = 600
```

## 完整示例

### 基础使用

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def main():
    # 创建配置
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="example-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        description="Example agent",
        enable_question_hints=True,
    )
    
    # 创建 Agent
    agent = SuperReActAgent(agent_config=config)
    
    # 执行任务
    result = await agent.process_input(
        "What is the weather today?"
    )
    
    print(result)
    
    # 清理
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
```

### 高级使用

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentConfig, AgentConstraints
from agent.context_manager import ContextManager
from llm.openrouter_llm import OpenRouterLLM

load_dotenv()

async def advanced_example():
    # 创建 LLM 客户端
    llm = OpenRouterLLM(
        api_key=os.getenv("API_KEY"),
        model_name="anthropic/claude-3.7-sonnet",
        temperature=0.1,
        max_tokens=4096
    )
    
    # 创建上下文管理器
    context = ContextManager(
        llm=llm,
        max_history_length=50
    )
    
    # 创建约束
    constraints = AgentConstraints(
        max_iteration=20,
        max_tool_calls_per_turn=5
    )
    
    # 创建配置
    config = SuperAgentConfig(
        agent_type="main",
        description="Advanced research agent",
        enable_question_hints=True,
        enable_extract_final_answer=True,
        reasoning_model="o3",
        enable_context_limit_retry=True,
        keep_tool_result=10,
        enable_todo_plan=True,
        constraints=constraints.to_constrain_config()
    )
    
    # 创建 Agent
    agent = SuperReActAgent(
        agent_config=config,
        context_manager=context
    )
    
    # 执行复杂任务
    result = await agent.process_input(
        "Research the latest AI developments in 2025"
    )
    
    print(result)
    
    # 获取历史记录
    history = context.get_history()
    print(f"Total messages: {len(history)}")
    
    # 清理
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(advanced_example())
```
