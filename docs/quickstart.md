# 快速开始

本指南帮助你在 5 分钟内运行第一个 DeepAgent 任务。

## 环境要求

- **Python 3.11+** (主环境)
- **Python 3.12+** (MCP 工具环境，可选但推荐)
- **Node.js** (部分 MCP 工具需要)
- **Chrome/Chromium** (浏览器工具需要)

## 1. 安装主环境

```bash
# 进入项目目录
cd deepagent

# 使用 uv 安装依赖（推荐）
uv sync

# 额外安装 tiktoken
uv pip install tiktoken

# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows
```

## 2. 配置环境变量

```bash
# 创建 .env 文件
cp .env.example .env
```

编辑 `.env` 文件，填入必需变量：

```bash
# LLM API 配置（必需）
API_BASE=https://openrouter.ai/api/v1
API_KEY=your_openrouter_api_key
MODEL_NAME=anthropic/claude-3.7-sonnet
MODEL_PROVIDER=openrouter

# 推理模型配置（用于 QA 处理）
REASONING_MODEL_NAME=o3

# 浏览器工具配置
CHROME_PATH=/usr/bin/google-chrome  # Linux 示例
CHROME_USER_PROFILE_DIR=/home/user/.config/google-chrome

# MCP 工具 API 密钥（根据需要使用）
OPENROUTER_API_KEY=your_openrouter_api_key
GEMINI_API_KEY=your_gemini_api_key
E2B_API_KEY=your_e2b_api_key
SERPER_API_KEY=your_serper_api_key
```

## 3. 安装 MCP 工具环境（可选）

如果需要使用 MCP 工具（浏览器、搜索、Python 执行等）：

```bash
cd tool/

# 创建 Python 3.12 虚拟环境
uv venv .venv-tool --python 3.12

# 激活环境
source .venv-tool/bin/activate  # Linux/Mac
# .\.venv-tool\Scripts\Activate.ps1  # Windows

# 安装依赖
uv pip install --no-deps -r requirements.txt  # Linux
# uv pip install --no-deps -r requirements_mac.txt  # Mac
```

## 4. 运行示例

### 基础示例

```python
# test/super_react_agent_test_run.py
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

async def main():
    # 创建主 Agent 配置
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="main-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name=os.getenv("MODEL_NAME"),
        enable_question_hints=True,
        enable_extract_final_answer=True,
    )
    
    # 初始化 Agent
    agent = SuperReActAgent(agent_config=config)
    
    # 运行任务
    result = await agent.process_input(
        question="What is the capital of France?"
    )
    
    print(f"Answer: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

运行：

```bash
uv run python test/super_react_agent_test_run.py
```

## 5. 验证安装

### 测试 Agent 功能

```bash
# 运行所有测试
uv run pytest

# 运行特定测试
uv run pytest test/super_react_agent_test_run.py -v
```

### 检查代码风格

```bash
uv run ruff check .
```

## 6. 第一个任务

创建一个简单的任务文件 `my_task.py`：

```python
#!/usr/bin/env python
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

async def run_task():
    """运行一个研究任务"""
    
    # 配置
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="research-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name=os.getenv("MODEL_NAME"),
        description="A research agent that can search and browse the web",
        enable_question_hints=True,
        enable_extract_final_answer=True,
        max_iteration=15,
    )
    
    # 创建 Agent
    agent = SuperReActAgent(agent_config=config)
    
    # 研究问题
    question = """
    What are the latest developments in AI agents in 2025?
    Please search for recent news and provide a summary.
    """
    
    print(f"🤔 Task: {question}")
    print("-" * 50)
    
    # 执行任务
    result = await agent.process_input(question=question)
    
    print("-" * 50)
    print(f"✅ Result: {result}")
    
    # 清理资源
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(run_task())
```

运行任务：

```bash
uv run python my_task.py
```

## 下一步

- 查看 [核心概念](./concepts.md) 了解 Agent 工作原理
- 阅读 [配置指南](./configuration.md) 自定义 Agent 行为
- 探索 [MCP 工具](./mcp-tools.md) 扩展能力
- 参考 [示例](./examples.md) 学习更多用法

## 常见问题

### Q: 运行时提示缺少 tiktoken？

```bash
uv pip install tiktoken
```

### Q: MCP 工具无法启动？

确保：
1. 工具环境已正确安装（Python 3.12+）
2. 环境变量已正确配置
3. Node.js 已安装（部分工具需要）

### Q: 浏览器工具报错？

检查 `CHROME_PATH` 和 `CHROME_USER_PROFILE_DIR` 是否正确设置。

```bash
# Linux 查找 Chrome 路径
which google-chrome
# 或
which chromium-browser
```
