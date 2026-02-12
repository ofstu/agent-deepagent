# 配置指南

本文档详细介绍 DeepAgent 的各种配置选项。

## 环境变量配置

### 必需变量

```bash
# LLM API 配置（必需）
API_BASE=https://openrouter.ai/api/v1        # LLM API 基础 URL
API_KEY=your_api_key                         # API 密钥
MODEL_NAME=anthropic/claude-3.7-sonnet       # 主模型名称
MODEL_PROVIDER=openrouter                    # 模型提供商

# 推理模型配置（用于 QA 处理）
REASONING_MODEL_NAME=o3                      # 推理模型（如 OpenAI O3）
```

### 可选变量

```bash
# 数据目录
DATA_DIR=./data                              # 数据文件存放目录

# 浏览器配置
CHROME_PATH=/usr/bin/google-chrome          # Chrome 浏览器路径
CHROME_USER_PROFILE_DIR=/path/to/profile    # Chrome 用户配置文件目录

# MCP 工具传输方式
AUTO_BROWSER_TRANSPORT=sse                  # 可选: sse 或 stdio
AUTO_BROWSER_SSE_URL=http://127.0.0.1:8930/sse  # SSE 服务地址
AUTO_BROWSER_START_SSE=true                 # 是否自动启动 SSE 服务
```

### MCP 工具 API 密钥

```bash
# LLM 提供商
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key

# Python 沙箱执行
E2B_API_KEY=your_e2b_key
E2B_TEMPLATE_ID=your_template_id            # E2B 模板 ID

# 搜索服务
SERPER_API_KEY=your_serper_key              # Google 搜索
JINA_API_KEY=your_jina_key                  # Jina 深度搜索
PERPLEXITY_API_KEY=your_perplexity_key      # Perplexity 搜索

# 音频处理（可选）
ACR_ACCESS_KEY=your_acr_key
ACR_ACCESS_SECRET=your_acr_secret
```

## Agent 配置

### SuperAgentConfig

主要配置类，继承自 `ReActAgentConfig`。

```python
from agent.super_config import SuperAgentConfig

config = SuperAgentConfig(
    # 基础配置
    agent_type="main",                          # Agent 类型: "main" 或子 Agent 名称
    description="Main agent for research",      # Agent 描述
    
    # 推理模型配置
    enable_question_hints=True,                 # 启用问题提示提取
    enable_extract_final_answer=True,           # 启用最终答案提取
    open_api_key="your_openai_key",             # OpenAI API 密钥
    reasoning_model="o3",                       # 推理模型名称
    
    # 上下文管理
    enable_context_limit_retry=True,            # 启用上下文限制重试
    keep_tool_result=-1,                        # 保留工具结果数量 (-1=全部)
    
    # 工具调用限制
    max_tool_calls_per_turn=5,                  # 每轮最大工具调用次数
    
    # 计划追踪
    enable_todo_plan=True,                      # 启用计划追踪
    
    # 任务指导
    task_guidance="",                           # 额外任务指导文本
)
```

### 约束配置

```python
from agent.super_config import AgentConstraints

constraints = AgentConstraints(
    max_iteration=10,                          # ReAct 循环最大迭代次数
    max_tool_calls_per_turn=5,                 # 每轮最大工具调用数
    reserved_max_chat_rounds=40,               # 保留的最大对话轮数
)
```

### 子 Agent 配置

```python
from agent.super_config import SuperAgentConfig

# 主 Agent 配置
main_config = SuperAgentConfig(
    agent_type="main",
    description="Main coordinator agent",
    sub_agent_configs={
        # 浏览器 Agent
        "browser-agent": SuperAgentConfig(
            agent_type="browser",
            description="Agent for web browsing",
            max_iteration=10,
            enable_question_hints=False,
        ),
        # 代码 Agent
        "coder-agent": SuperAgentConfig(
            agent_type="coder",
            description="Agent for code execution",
            max_iteration=15,
            enable_extract_final_answer=True,
        ),
    }
)
```

## 使用工厂方法创建配置

### SuperAgentFactory

```python
from agent.super_config import SuperAgentFactory

# 创建主 Agent 配置
main_config = SuperAgentFactory.create_main_agent_config(
    agent_id="main-agent",
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
    model_name="anthropic/claude-3.7-sonnet",
    model_provider="openrouter",
    
    # 可选参数
    description="Main research agent",
    max_iteration=15,
    enable_question_hints=True,
    enable_extract_final_answer=True,
    reasoning_model="o3",
    open_api_key=os.getenv("OPENAI_API_KEY"),
    task_guidance="Focus on accuracy and completeness",
)

# 创建子 Agent 配置
browser_config = SuperAgentFactory.create_sub_agent_config(
    agent_name="browser-agent",
    agent_type="browser",
    description="Agent for web browsing",
    max_iteration=10,
)
```

## LLM 配置

### OpenRouterLLM 配置

```python
from llm.openrouter_llm import OpenRouterLLM, OpenRouterConfig

# 方式 1: 直接使用配置类
config = OpenRouterConfig(
    api_key="your_api_key",
    api_base="https://openrouter.ai/api/v1",
    model_name="anthropic/claude-3.7-sonnet",
    max_retries=3,
    timeout=600,
    temperature=0.1,
    top_p=1.0,
    max_tokens=4096,
    max_context_length=200000,
    
    # 定价配置（可选）
    input_token_price=3.0,              # 每百万输入 tokens
    output_token_price=15.0,            # 每百万输出 tokens
    cache_input_token_price=0.3,
    
    # OpenRouter 特定配置
    openrouter_provider="anthropic",    # 提供商偏好
    disable_cache_control=False,
)

llm = OpenRouterLLM(
    api_key=config.api_key,
    api_base=config.api_base,
    model_name=config.model_name,
    **config.dict()
)

# 方式 2: 简化的初始化
llm = OpenRouterLLM(
    api_key="your_api_key",
    model_name="anthropic/claude-3.7-sonnet",
    temperature=0.1,
    max_tokens=4096,
)
```

### 支持的模型

```python
# Anthropic Claude
model_name = "anthropic/claude-3.7-sonnet"
model_name = "anthropic/claude-3.5-sonnet"
model_name = "anthropic/claude-3-opus"

# OpenAI
model_name = "openai/gpt-4o"
model_name = "openai/gpt-4o-mini"
model_name = "openai/o3-mini"

# Google Gemini
model_name = "google/gemini-2.5-flash"
model_name = "google/gemini-2.0-pro"

# 通过 OpenRouter
model_name = "anthropic/claude-3.7-sonnet:beta"
model_name = "openai/o3-mini:high"
```

## MCP 工具配置

### 浏览器工具

```python
from mcp import StdioServerParameters

# STDIO 方式
browser_params = StdioServerParameters(
    command="python",
    args=[
        "tool/mcp_servers/browser_use_mcp_server.py",
        "--transport", "stdio"
    ],
    env={
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "CHROME_PATH": os.getenv("CHROME_PATH"),
    }
)

# SSE 方式
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig

browser_config = ToolServerConfig(
    server_path="http://127.0.0.1:8930/sse",
    transport="sse"
)
```

### 搜索工具

```python
# Google 搜索
searching_params = StdioServerParameters(
    command="npx",
    args=["-y", "serper-search-scrape-mcp-server"],
    env={
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
)
```

### Python 执行工具

```python
python_params = StdioServerParameters(
    command="python",
    args=["tool/mcp_servers/python_server.py"],
    env={
        "E2B_API_KEY": os.getenv("E2B_API_KEY"),
        "E2B_TEMPLATE_ID": os.getenv("E2B_TEMPLATE_ID"),
    }
)
```

## 完整配置示例

### 研究 Agent 配置

```python
import os
from dotenv import load_dotenv
from agent.super_config import SuperAgentFactory
from agent.super_react_agent import SuperReActAgent

load_dotenv()

# 创建配置
config = SuperAgentFactory.create_main_agent_config(
    agent_id="research-agent",
    api_key=os.getenv("API_KEY"),
    api_base=os.getenv("API_BASE"),
    model_name="anthropic/claude-3.7-sonnet",
    model_provider="openrouter",
    
    description="Advanced research agent with browsing and analysis capabilities",
    max_iteration=20,
    
    # 启用推理增强
    enable_question_hints=True,
    enable_extract_final_answer=True,
    reasoning_model="o3",
    open_api_key=os.getenv("OPENAI_API_KEY"),
    
    # 上下文配置
    enable_context_limit_retry=True,
    keep_tool_result=15,
    max_tool_calls_per_turn=5,
    
    # 计划追踪
    enable_todo_plan=True,
    task_guidance="""
    When conducting research:
    1. Break down complex questions into sub-tasks
    2. Use browser tool for web searches
    3. Verify information from multiple sources
    4. Provide structured answers with citations
    """,
)

# 创建 Agent
agent = SuperReActAgent(agent_config=config)

# 使用 Agent
result = await agent.process_input(question="What are the latest developments in AI?")
```

### 多 Agent 系统配置

```python
from agent.super_config import SuperAgentConfig, SuperAgentFactory

# 主 Agent 配置
main_config = SuperAgentConfig(
    agent_type="main",
    description="Main coordinator",
    max_iteration=15,
    enable_question_hints=True,
    enable_extract_final_answer=True,
    
    # 子 Agent 配置
    sub_agent_configs={
        "browser-agent": SuperAgentConfig(
            agent_type="browser",
            description="Web browsing specialist",
            max_iteration=10,
            enable_todo_plan=True,
            max_tool_calls_per_turn=3,
        ),
        "coder-agent": SuperAgentConfig(
            agent_type="coder",
            description="Code execution specialist",
            max_iteration=15,
            enable_extract_final_answer=True,
        ),
        "analyzer-agent": SuperAgentConfig(
            agent_type="analyzer",
            description="Data analysis specialist",
            max_iteration=10,
            enable_question_hints=False,
        ),
    }
)

# 创建主 Agent
main_agent = SuperReActAgent(agent_config=main_config)
```

## 配置文件模板

### .env 文件模板

```bash
# ============================================================
# OpenJiuwen DeepAgent Environment Configuration
# ============================================================

# ------------------- LLM Configuration -------------------
API_BASE=https://openrouter.ai/api/v1
API_KEY=your_openrouter_api_key_here
MODEL_NAME=anthropic/claude-3.7-sonnet
MODEL_PROVIDER=openrouter

# Reasoning model (OpenAI O3)
REASONING_MODEL_NAME=o3

# ------------------- Browser Configuration -------------------
CHROME_PATH=/usr/bin/google-chrome
CHROME_USER_PROFILE_DIR=/home/user/.config/google-chrome

# MCP Tool transport (sse or stdio)
AUTO_BROWSER_TRANSPORT=sse
AUTO_BROWSER_SSE_URL=http://127.0.0.1:8930/sse
AUTO_BROWSER_START_SSE=true

# ------------------- API Keys -------------------
# LLM Providers
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
OPENROUTER_API_KEY=your_openrouter_key

# Python Sandbox
E2B_API_KEY=your_e2b_key
E2B_TEMPLATE_ID=your_template_id

# Search Services
SERPER_API_KEY=your_serper_key
JINA_API_KEY=your_jina_key
PERPLEXITY_API_KEY=your_perplexity_key

# Optional: Audio Recognition
ACR_ACCESS_KEY=your_acr_key
ACR_ACCESS_SECRET=your_acr_secret

# ------------------- Data Configuration -------------------
DATA_DIR=./data
```

## 配置最佳实践

### 1. 迭代次数设置

```python
# 根据任务复杂度调整
SIMPLE_TASK = 5         # 简单问答
RESEARCH_TASK = 15      # 研究任务
COMPLEX_TASK = 25       # 复杂多步任务
MAX_SAFE_LIMIT = 30     # 安全上限
```

### 2. 上下文管理

```python
# 保留适量的工具结果
config = SuperAgentConfig(
    keep_tool_result=10,     # 保留最近 10 个工具结果
    # 或
    keep_tool_result=-1,     # 保留所有（适合短对话）
)
```

### 3. 工具调用限制

```python
# 防止工具滥用
config = SuperAgentConfig(
    max_tool_calls_per_turn=3,   # 每轮最多 3 次调用
    # 或
    max_tool_calls_per_turn=5,   # 复杂任务放宽到 5 次
)
```

### 4. 环境变量管理

```python
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 验证必需变量
def validate_env():
    required = ["API_KEY", "API_BASE", "MODEL_NAME"]
    missing = [var for var in required if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing required environment variables: {missing}")

validate_env()
```
