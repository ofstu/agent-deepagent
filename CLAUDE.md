# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概览

openJiuwen DeepAgent 是一个灵活、强大且易用的 AI agent 开发框架，支持 ToC 和 ToB 各种场景。该项目基于 openjiuwen 核心框架，使用增强的 ReAct agent 架构和 MCP（Model Context Protocol）工具服务器。

## 环境设置

### 双环境架构
项目需要两个独立的 Python 环境：

1. **主框架环境**（Python 3.11+）：
   ```bash
   cd ./deepagent
   uv sync  # 创建/更新虚拟环境
   uv pip install tiktoken  # 单独安装
   ```

2. **MCP 工具环境**（Python 3.12+）：
   ```bash
   cd ./tool
   uv venv .venv-tool --python 3.12
   # Windows: .\.venv-tool\Scripts\Activate.ps1
   # macOS/Linux: source .venv-tool/bin/activate
   # Windows: uv pip install --no-deps -r requirements.txt
   # macOS/Linux: uv pip install --no-deps -r requirements_mac.txt
   ```

### 环境变量
在 `.env` 文件中配置以下变量：

**主 Agent 配置**：
- `API_BASE`, `API_KEY`, `MODEL_NAME`, `MODEL_PROVIDER`
- `REASONING_MODEL_NAME`

**API Keys**：
- `OPENAI_API_KEY` - audio, vision, doubter 服务器
- `GEMINI_API_KEY` - vision, browser_use_mcp, doubter, searching 服务器
- `OPENROUTER_API_KEY` - vision, reasoning, doubter, searching 服务器
- `E2B_API_KEY`, `E2B_TEMPLATE_ID` - Python 沙盒执行
- `SERPER_API_KEY`, `JINA_API_KEY`, `PERPLEXITY_API_KEY` - 搜索服务

**其他**：
- `CHROME_PATH`, `CHROME_USER_PROFILE_DIR` - 浏览器配置
- `DATA_DIR` - 数据路径

## 常用命令

### 运行测试
```bash
deactivate  # 如果当前在 tool 环境中
cd ./deepagent  # 跳到 DeepAgent 文件夹
source ./.venv/bin/activate  # macOS/Linux
uv run ./test/super_react_agent_test_run.py
```

### 测试
```bash
pytest  # 运行所有测试
pytest tests/  # 运行特定目录
```

### 代码格式化与检查
```bash
ruff check .  # 检查代码风格
ruff format .  # 格式化代码
```

## 核心架构

### Agent 架构层次

```
SuperReActAgent (主 Agent)
├── SuperAgentConfig (配置)
├── ContextManager (上下文管理)
├── ToolCallHandler (工具调用处理)
├── QAHandler (问题提示与答案提取)
├── PlanTracker (计划跟踪)
└── Sub-Agents (子 Agent)
    ├── agent-browsing (浏览专家)
    └── agent-coding (编码专家)
```

### MCP 工具服务器架构

项目使用 MCP (Model Context Protocol) 来组织和管理工具：

**工具组**（在 `test/super_react_agent_test_run.py` 中定义）：
- `tool-autobrowser` - 浏览器自动化（browser-use-server）
- `tool-transcribe` - 音频转录（audio-mcp-server）
- `tool-reasoning` - 推理能力（reasoning-mcp-server）
- `tool-reading` - 文档阅读（reading-mcp-server）
- `tool-searching` - 网络搜索（searching-mcp-server）
- `tool-vqa` - 视觉问答（vision-mcp-server）
- `tool-code` - Python 代码执行（e2b-python-interpreter）

**工具分配策略**：
- 主 Agent：仅使用 `tool-reasoning`（协调子 Agent）
- Browsing Agent：`tool-searching`, `tool-vqa`, `tool-reading`, `tool-autobrowser`, `tool-transcribe`
- Coding Agent：`tool-code`, `tool-vqa`, `tool-reading`

### 关键组件

**SuperReActAgent** (`agent/super_react_agent.py`):
- 主/子 agent 共用同一个类，通过 `agent_type` 区分
- 支持 MCP 工具的动态注册和调用
- 内置上下文限制处理和重试逻辑
- ReAct 循环：`max_iteration` 控制最大迭代次数

**ContextManager** (`agent/context_manager.py`):
- 自定义上下文管理，替代默认的 ContextEngine
- 支持上下文摘要生成（当接近限制时）
- 维护消息历史：user, assistant, tool, system

**PlanTracker** (`agent/super_react_agent.py`):
- 从 LLM 输出中提取 #PLAN# 块
- 跟踪步骤进度并写入 `todo.md`
- 支持显式的 `<TODO_PLAN>` 和 `<TODO_STATUS>` 块
- 将计划摘要注入上下文

**ToolCallHandler** (`agent/tool_call_handler.py`):
- 处理工具调用的执行
- 管理子 agent 的调用（`agent-*` 前缀的工具自动路由到子 agent）

**QAHandler** (`agent/qa_handler.py`):
- 使用推理模型提取问题提示
- 提取并格式化最终答案（支持多种答案类型）

### MCP 工具注册流程

1. 定义 MCP server 配置（`StdioServerParameters` 或 SSE URL）
2. 调用 `agent.create_mcp_tools(server_name, client_type, params)`
3. 返回的 `LocalFunction` 列表添加到 agent：`agent.add_tools(tools)`
4. MCP 工具调用通过 `Runner.run_tool(tool_id, kwargs)` 执行

### 子 Agent 注册

```python
# 创建子 agent
sub_agent = SuperReActAgent(agent_config=sub_config, ...)

# 注册到主 agent（自动创建同名工具）
main_agent.register_sub_agent("agent-browsing", sub_agent)
```

调用子 agent 时，LLM 会调用名为 `agent-browsing` 的工具，ToolCallHandler 会自动路由到对应的子 agent 实例。

### 提示模板（Prompt Templates）

提示模板位于 `agent/prompt_templates.py`：
- `get_main_agent_system_prompt()` - 主 agent 系统提示
- `get_browsing_agent_system_prompt()` - 浏览 agent 系统提示
- `get_coding_agent_system_prompt()` - 编码 agent 系统提示
- `process_input()` - 处理 GAIA 数据集输入
- `get_task_instruction_prompt()` - 任务指令提示

### LLM 集成

项目使用自定义的 `OpenRouterLLM` (`llm/openrouter_llm.py`) 封装，支持：
- 多种 LLM 提供商（OpenRouter, OpenAI 等）
- 工具调用（function calling）
- 上下文限制检测（`ContextLimitError`）
- 重试逻辑

## 数据处理

### GAIA 数据集格式
数据位于 `data/test.jsonl`，每行一个 JSON 对象：
```json
{
  "task_question": "问题文本",
  "file_path": "文件路径或 null",
  "label_answer": "标准答案"
}
```

## 配置模式

### SuperAgentConfig 关键参数
- `agent_type`: "main" 或子 agent 名称
- `max_iteration`: ReAct 循环最大迭代次数（默认 10）
- `max_tool_calls_per_turn`: 每轮最大工具调用数（默认 5）
- `enable_question_hints`: 启用问题提示提取（仅主 agent）
- `enable_extract_final_answer`: 启用最终答案提取（仅主 agent）
- `enable_context_limit_retry`: 启用上下文限制重试
- `enable_todo_plan`: 启用计划跟踪到 todo.md

### 工厂方法
使用 `SuperAgentFactory` 创建配置：
- `create_main_agent_config()` - 主 agent
- `create_sub_agent_config()` - 子 agent

## 重要约定

1. **工具命名**：子 agent 工具必须以 `agent-` 开头（如 `agent-browsing`）才能被自动路由
2. **上下文清理**：在处理新查询前调用 `agent._context_manager.clear()` 清空上下文
3. **MCP 环境隔离**：MCP 服务器运行在独立的 Python 环境中（.venv-tool）
4. **Node.js 依赖**：某些 MCP 工具需要 Node.js（如 serper-search-scrape-mcp-server）

## 测试与评估

测试脚本会：
1. 从 `data/test.jsonl` 读取 GAIA 数据集
2. 对每个问题运行主 agent
3. 提取预测答案并与标准答案比较
4. 生成评估报告（`evaluation_results_[timestamp].txt` 和 `.json`）

## 常见问题

- **SSL 验证**：设置 `LLM_SSL_VERIFY=false` 禁用 SSL 验证（如有需要）
- **浏览器传输**：AUTO_BROWSER_TRANSPORT 可设为 "sse" 或 "stdio"（SSE 更稳定）
- **上下文限制**：启用 `enable_context_limit_retry` 自动处理上下文溢出
- **工具依赖**：确保 MCP 工具环境中安装了所有依赖（requirements.txt/requirements_mac.txt）
