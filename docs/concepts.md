# 核心概念

本文档介绍 OpenJiuwen DeepAgent 的核心架构和关键概念。

## Agent 架构

DeepAgent 采用增强的 ReAct (Reasoning + Acting) 架构，支持多层 Agent 编排。

### ReAct 循环

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Thought   │────▶│   Action    │────▶│ Observation │
└─────────────┘     └─────────────┘     └─────────────┘
       ▲                                         │
       └─────────────────────────────────────────┘
```

**循环流程：**

1. **Thought (思考)** - LLM 分析当前状态，决定下一步行动
2. **Action (行动)** - 执行工具调用（搜索、浏览、代码执行等）
3. **Observation (观察)** - 收集工具返回的结果
4. **迭代** - 重复直到获得最终答案或达到最大迭代次数

### 主 Agent 与子 Agent

```
                    Main Agent
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   Sub-Agent 1     Sub-Agent 2      Sub-Agent 3
   (Browser)       (Coder)          (Analyzer)
```

**主 Agent (Main Agent)**
- 负责任务分解和协调
- 调用子 Agent 执行特定任务
- 整合子 Agent 的结果

**子 Agent (Sub-Agent)**
- 专注于特定领域任务
- 独立执行 ReAct 循环
- 返回结构化结果给主 Agent

## 核心组件

### 1. SuperReActAgent

主 Agent 类，实现完整的 ReAct 逻辑。

**关键方法：**

```python
# 处理输入
result = await agent.process_input(question="...")

# 重置状态
agent.reset()

# 清理资源
await agent.cleanup()
```

**执行流程：**

```python
async def process_input(self, question: str) -> str:
    # 1. 提取任务提示（可选）
    hints = await self._extract_hints(question)
    
    # 2. ReAct 循环
    for iteration in range(max_iterations):
        # Thought
        response = await self._llm_call(messages)
        
        # Action
        if has_tool_calls:
            results = await self._execute_tools(tool_calls)
        
        # Observation
        self._add_observations(results)
        
        # 检查是否完成
        if is_final_answer:
            break
    
    # 3. 提取最终答案（可选）
    final_answer = await self._extract_final_answer()
    
    return final_answer
```

### 2. ContextManager

管理对话上下文和消息历史。

**功能：**
- 消息历史管理（添加、检索、修剪）
- Token 限制处理
- 自动摘要生成

**工作原理：**

```python
# 添加消息
context_manager.add_message(
    role="user",
    content="...",
    tool_calls=[...]
)

# 自动修剪（当超过限制时）
if total_tokens > max_context_length:
    # 生成历史摘要
    summary = await self._generate_summary()
    # 修剪旧消息
    self._trim_old_messages()
```

### 3. ToolCallHandler

处理工具调用和子 Agent 执行。

**功能：**
- 类型转换和验证
- 子 Agent 工具创建
- 并发工具执行

**子 Agent 工具：**

```python
# 创建子 Agent 工具
sub_agent_tool = tool_handler.create_sub_agent_tool(
    agent_name="browser-agent",
    sub_agent=browser_agent
)

# 子 Agent 执行
result = await sub_agent_tool.execute(task="...")
```

### 4. QAHandler

处理推理模型的调用（如 OpenAI O3）。

**功能：**
- 问题提示提取
- 答案类型判断
- 最终答案提取

```python
# 提取提示
hints = await qa_handler.extract_hints(question)

# 判断答案类型
answer_type = await qa_handler.determine_answer_type(question)

# 提取最终答案
final_answer = await qa_handler.extract_final_answer(
    conversation=messages,
    answer_type="date"
)
```

## 配置系统

### SuperAgentConfig

基于 Pydantic 的配置类。

**主要配置项：**

```python
class SuperAgentConfig(ReActAgentConfig):
    # Agent 类型
    agent_type: str = "main"  # "main" 或子 Agent 名称
    
    # 推理模型
    enable_question_hints: bool = True
    enable_extract_final_answer: bool = True
    reasoning_model: str = "o3"
    
    # 上下文管理
    enable_context_limit_retry: bool = True
    keep_tool_result: int = -1  # -1 = 保留所有
    
    # 工具限制
    max_tool_calls_per_turn: int = 5
    
    # 计划追踪
    enable_todo_plan: bool = True
    
    # 子 Agent 配置
    sub_agent_configs: Dict[str, SuperAgentConfig]
```

### 工厂方法

```python
from agent.super_config import SuperAgentFactory

# 创建主 Agent 配置
main_config = SuperAgentFactory.create_main_agent_config(
    agent_id="main-agent",
    api_key="...",
    api_base="...",
    model_name="...",
    enable_question_hints=True,
    max_iteration=15,
)

# 创建子 Agent 配置
browser_config = SuperAgentFactory.create_sub_agent_config(
    agent_name="browser-agent",
    agent_type="browser",
    description="Agent for web browsing",
    max_iteration=10,
)
```

## MCP 工具架构

### MCP (Model Context Protocol)

MCP 是连接 LLM 和外部工具的标准协议。

**传输方式：**
- **STDIO** - 标准输入输出（本地进程）
- **SSE** - Server-Sent Events（HTTP 流）

### 工具注册

```python
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig
from mcp import StdioServerParameters

# STDIO 方式
server_params = StdioServerParameters(
    command="python",
    args=["tool/mcp_servers/browser_use_mcp_server.py"],
    env={"API_KEY": "..."}
)

# SSE 方式
tool_config = ToolServerConfig(
    server_path="http://localhost:8930/sse",
    transport="sse"
)
```

### 内置工具

| 工具 | 描述 | 示例 |
|------|------|------|
| **Browser** | 浏览器自动化 | 网页浏览、表单填写、数据提取 |
| **Search** | Google 搜索 | 信息检索、新闻搜索 |
| **Python** | Python 代码执行 | 数据分析、计算、可视化 |
| **Vision** | 图像分析 | OCR、图像理解 |
| **Audio** | 音频处理 | 语音识别、音频分析 |

## 上下文管理策略

### Token 限制处理

当上下文超过模型限制时，系统自动处理：

1. **尝试摘要** - 生成历史对话摘要
2. **修剪消息** - 移除最早的消息
3. **重试调用** - 使用精简后的上下文重新调用

```python
async def _handle_context_limit(self, messages: List[Dict]) -> List[Dict]:
    # 1. 生成摘要
    summary = await self._generate_summary(messages)
    
    # 2. 修剪旧消息
    trimmed = self._trim_messages(messages, keep_recent=10)
    
    # 3. 添加摘要到系统消息
    trimmed.insert(0, {
        "role": "system",
        "content": f"Previous conversation summary: {summary}"
    })
    
    return trimmed
```

### 计划追踪

Agent 可以维护任务计划（todo.md）：

```python
# 创建计划
agent.create_plan(steps=[
    "Search for recent AI news",
    "Browse relevant articles",
    "Summarize findings"
])

# 更新进度
agent.update_progress(step=1, status="completed")
```

## 工作流示例

### 研究任务流程

```
用户提问
    │
    ▼
提取提示词
    │
    ▼
ReAct 循环
    │
    ├── Thought: 需要搜索信息
    │
    ├── Action: 调用搜索工具
    │
    ├── Observation: 获取搜索结果
    │
    ├── Thought: 需要浏览网页
    │
    ├── Action: 调用浏览器工具
    │
    ├── Observation: 获取网页内容
    │
    └── ... (重复直到完成)
    │
    ▼
提取最终答案
    │
    ▼
返回答案
```

## 最佳实践

### 1. 合理设置迭代次数

```python
# 简单任务
max_iteration = 5

# 复杂任务
max_iteration = 20

# 防止无限循环
max_iteration = 30  # 上限
```

### 2. 使用子 Agent 分解任务

```python
# 不好的做法：一个 Agent 做所有事
main_agent.process_input(complex_task)

# 好的做法：任务分解
browser_result = await browser_agent.process_input(search_task)
code_result = await coder_agent.process_input(coding_task)
final_result = await main_agent.synthesize(browser_result, code_result)
```

### 3. 配置合适的上下文长度

```python
# 根据任务复杂度调整
context_config = {
    "max_history_length": 50,  # 保留多少条消息
    "keep_tool_result": 10,    # 保留多少工具结果
}
```

### 4. 启用推理模型增强

```python
config = SuperAgentConfig(
    enable_question_hints=True,        # 提取任务提示
    enable_extract_final_answer=True,  # 格式化最终答案
    reasoning_model="o3",              # 使用推理模型
)
```
