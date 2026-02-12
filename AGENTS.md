# openJiuwen DeepAgent - Agent 开发指南

## 1. 项目概览

**openJiuwen DeepAgent** 是一个灵活的 AI Agent 开发框架，支持复杂任务的多层 Agent 编排。

### 核心架构
- **主 Agent** (`agent/super_react_agent.py`): 增强版 ReAct Agent，支持子 Agent 调用
- **配置管理** (`agent/super_config.py`): Pydantic-based 配置系统
- **上下文管理** (`agent/context_manager.py`): 对话历史与摘要生成
- **工具调用** (`agent/tool_call_handler.py`): 工具执行与类型转换
- **MCP 服务器** (`tool/mcp_servers/`): 浏览器、搜索、Python 执行等工具

### 目录结构
```
agent/                  # 核心 Agent 逻辑
├── super_react_agent.py    # 主 ReAct Agent 实现
├── super_config.py         # Agent 配置类
├── context_manager.py      # 上下文管理
├── tool_call_handler.py    # 工具调用处理
└── qa_handler.py           # QA 处理

llm/                    # LLM 客户端
└── openrouter_llm.py       # OpenRouter 客户端

tool/mcp_servers/       # MCP 服务器
├── browser_use_mcp_server.py
├── searching_mcp_server.py
└── python_server.py
```

## 2. 构建与测试命令

### 环境设置（双环境）
```bash
# 1. 主环境（Python 3.11+）
uv sync                                    # 安装主环境依赖
uv pip install tiktoken

# 2. MCP 工具环境（Python 3.12+）
cd tool/
uv venv .venv-tool --python 3.12
source .venv-tool/bin/activate
uv pip install --no-deps -r requirements.txt
```

### 测试命令
```bash
# 运行所有测试
uv run pytest

# 运行单个测试文件
uv run pytest test/super_react_agent_test_run.py

# 运行特定测试函数
uv run pytest test/super_react_agent_test_run.py::test_function_name -v

# 生成 HTML 报告
uv run pytest --html=report/index.html --self-contained-html
```

### 代码检查
```bash
# 使用 ruff 检查
uv run ruff check .

# 自动修复
uv run ruff check . --fix

# 格式化代码
uv run ruff format .
```

### 运行 Agent
```bash
source .venv/bin/activate
uv run python test/super_react_agent_test_run.py
```

## 3. 代码风格规范

### 导入顺序
```python
# 1. 标准库
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# 2. 第三方库
from pydantic import BaseModel, Field
from openai import AsyncOpenAI

# 3. openjiuwen 框架
from openjiuwen.core.agent.agent import BaseAgent
from openjiuwen.core.common.logging import logger

# 4. 项目内部模块
from agent.super_config import SuperAgentConfig
```

### 命名规范
- **类名**: `PascalCase` (如 `SuperReActAgent`)
- **函数/方法**: `snake_case` (如 `create_sub_agent_tool`)
- **私有方法**: 下划线前缀 (如 `_generate_message_id`)
- **常量**: 大写 (如 `AUTO_BROWSER_TRANSPORT`)
- **模块名**: `snake_case`

### 类型注解
- 所有函数参数和返回值必须添加类型注解
- 使用 `Optional[T]` 而非 `T | None`（Python 3.11 兼容）

```python
async def extract_hints(self, question: str) -> str:
    ...

def add_message(self, role: str, content: Any, **kwargs) -> None:
    ...
```

### 错误处理与日志
```python
from tenacity import retry, stop_after_attempt, wait_exponential
from openjiuwen.core.common.logging import logger

@retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(1))
async def extract_hints(self, question: str) -> str:
    try:
        # 业务逻辑
    except Exception as e:
        logger.error(f"Failed to extract hints: {e}")
        raise
```

## 4. Agent 编排与执行流程

### 主 Agent 执行循环
1. **初始化**: `SuperReActAgent.__init__()` 加载配置和工具
2. **任务处理**: `process_input()` 处理输入并提取提示
3. **ReAct 循环**: Thought → Action → Observation
4. **上下文管理**: 自动处理 token 限制，生成摘要
5. **终止条件**: 达到最大迭代次数或获得最终答案

### 子 Agent 调用
- 主 Agent 通过 `ToolCallHandler` 创建子 Agent 工具
- 子 Agent 复用相同的 `SuperReActAgent` 类
- 配置通过 `sub_agent_configs` 字典传递

### MCP 工具集成
- 工具服务器配置使用 `StdioServerParameters`
- 支持 SSE 和 STDIO 两种传输方式
- 工具调用通过 `Runner.run_tool()` 统一执行

## 5. 环境变量配置

**必需变量**:
- `API_BASE`, `API_KEY`, `MODEL_NAME`: LLM API 配置
- `REASONING_MODEL_NAME`: 推理模型（如 o3）

**MCP 工具密钥**:
- `OPENAI_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`
- `E2B_API_KEY`: Python 沙箱执行
- `SERPER_API_KEY`: Google 搜索

**浏览器配置**:
- `CHROME_PATH`: 浏览器路径
- `CHROME_USER_PROFILE_DIR`: 用户配置目录

## 6. 开发检查清单

- [ ] 通过 `uv run ruff check .` 无错误
- [ ] 类型注解完整
- [ ] 新增功能有对应测试
- [ ] 日志记录适当
- [ ] 导入顺序符合规范
- [ ] 文档字符串完整
