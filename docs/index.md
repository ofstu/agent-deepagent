# OpenJiuwen DeepAgent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

OpenJiuwen DeepAgent 是一个灵活的 AI Agent 开发框架，支持复杂任务的多层 Agent 编排，旨在为 ToC 和 ToB 场景提供强大且易用的 Agent 开发能力。

## 核心特性

- **多层 Agent 编排** - 主 Agent 可调用子 Agent，实现任务分解与协作
- **ReAct 架构** - Thought → Action → Observation 循环，支持复杂推理
- **上下文管理** - 自动处理 Token 限制，智能生成对话摘要
- **MCP 工具生态** - 内置浏览器、搜索、Python 执行等多种工具
- **灵活配置** - Pydantic-based 配置系统，支持多种 LLM 提供商

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                    SuperReActAgent                          │
│                    (主 Agent)                                │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  Context     │   ToolCall   │     QA       │   Plan/ToDo    │
│  Manager     │   Handler    │   Handler    │   Tracking     │
└──────┬───────┴──────┬───────┴──────┬───────┴────────────────┘
       │              │              │
       ▼              ▼              ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Sub-Agent  │  │  MCP Tools   │  │  Reasoning   │
│   Tools      │  │  (Browser,   │  │  Model (O3)  │
│              │  │  Search,     │  │              │
│              │  │  Python)     │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 快速开始

```bash
# 1. 克隆仓库
git clone <repository-url>
cd deepagent

# 2. 安装主环境依赖
uv sync
uv pip install tiktoken

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env 文件，填入 API 密钥

# 4. 运行示例
source .venv/bin/activate
uv run python test/super_react_agent_test_run.py
```

## 支持的 MCP 工具

| 工具 | 描述 | 环境变量 |
|------|------|----------|
| **Browser** | 浏览器自动化 | `CHROME_PATH`, `OPENROUTER_API_KEY` |
| **Search** | Google 搜索 | `SERPER_API_KEY` |
| **Python** | Python 代码执行 | `E2B_API_KEY` |
| **Vision** | 图像分析 | `OPENROUTER_API_KEY` |
| **Audio** | 音频处理 | `OPENAI_API_KEY` |

## 文档导航

- [快速开始](./quickstart.md) - 5 分钟上手
- [核心概念](./concepts.md) - 理解 Agent 架构
- [配置指南](./configuration.md) - 详细配置说明
- [MCP 工具](./mcp-tools.md) - 工具使用指南
- [API 参考](./api-reference.md) - 完整 API 文档
- [开发指南](./development.md) - 贡献代码
- [示例](./examples.md) - 实战案例
- [故障排查](./troubleshooting.md) - 常见问题

## 项目结构

```
deepagent/
├── agent/                  # 核心 Agent 逻辑
│   ├── super_react_agent.py    # 主 ReAct Agent
│   ├── super_config.py         # 配置系统
│   ├── context_manager.py      # 上下文管理
│   ├── tool_call_handler.py    # 工具调用处理
│   └── qa_handler.py           # QA 处理
├── llm/                    # LLM 客户端
│   └── openrouter_llm.py       # OpenRouter 实现
├── tool/                   # MCP 工具
│   └── mcp_servers/            # 工具服务器
│       ├── browser_use_mcp_server.py
│       ├── searching_mcp_server.py
│       └── python_server.py
├── test/                   # 测试文件
└── docs/                   # 文档
```

## 许可证

MIT License - 详见 [LICENSE](../LICENSE) 文件

## 贡献

欢迎贡献代码！请查看 [开发指南](./development.md) 了解如何参与项目。

## 支持

如有问题，请：
- 查看 [故障排查](./troubleshooting.md)
- 提交 GitHub Issue
- 查看现有 Issues
