# 开发指南

本文档介绍如何为 DeepAgent 项目贡献代码。

## 开发环境设置

### 1. 克隆仓库

```bash
git clone <repository-url>
cd deepagent
```

### 2. 安装主环境

```bash
# 使用 uv 安装依赖
uv sync

# 额外安装 tiktoken
uv pip install tiktoken

# 激活虚拟环境
source .venv/bin/activate
```

### 3. 安装 MCP 工具环境（可选）

```bash
cd tool/

# 创建 Python 3.12 虚拟环境
uv venv .venv-tool --python 3.12

# 激活并安装依赖
source .venv-tool/bin/activate
uv pip install --no-deps -r requirements.txt  # Linux
# uv pip install --no-deps -r requirements_mac.txt  # Mac
```

### 4. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件填入 API 密钥
```

## 代码规范

### 导入顺序

```python
# 1. 标准库
import json
import asyncio
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

```python
# 所有函数必须添加类型注解
async def extract_hints(self, question: str) -> str:
    ...

def add_message(self, role: str, content: Any, **kwargs) -> None:
    ...

# 使用 Optional[T] 而非 T | None
from typing import Optional

def get_config(key: str) -> Optional[SuperAgentConfig]:
    ...
```

### 文档字符串

```python
class ContextManager:
    """
    管理对话上下文：
    - 消息历史管理（添加、检索、修剪）
    - 摘要生成与上下文溢出处理
    """

    def add_message(self, role: str, content: Any, **kwargs):
        """
        添加消息到历史记录

        Args:
            role: 消息角色 (user, assistant, tool, system)
            content: 消息内容
            **kwargs: 额外字段 (tool_calls, tool_call_id 等)
        """
```

### 错误处理

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(wait=wait_exponential(multiplier=15), stop=stop_after_attempt(1))
async def extract_hints(self, question: str) -> str:
    try:
        # 业务逻辑
        result = await self._call_llm(prompt)
        return result
    except ContextLimitError as e:
        logger.error(f"Context limit exceeded: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to extract hints: {e}")
        raise
```

## 测试规范

### 编写测试

```python
# test/test_example.py
import pytest
from agent.super_react_agent import SuperReActAgent

@pytest.mark.asyncio
async def test_agent_basic():
    """测试 Agent 基本功能"""
    agent = create_test_agent()
    
    result = await agent.process_input("Hello")
    
    assert result is not None
    assert len(result) > 0
    
    await agent.cleanup()

@pytest.mark.asyncio
async def test_context_manager():
    """测试上下文管理"""
    from agent.context_manager import ContextManager
    
    context = ContextManager(max_history_length=10)
    
    context.add_message("user", "Hello")
    context.add_message("assistant", "Hi!")
    
    history = context.get_history()
    assert len(history) == 2
```

### 运行测试

```bash
# 运行所有测试
uv run pytest

# 运行特定测试文件
uv run pytest test/test_example.py

# 运行特定测试函数
uv run pytest test/test_example.py::test_agent_basic -v

# 生成覆盖率报告
uv run pytest --cov=agent --cov-report=html
```

## 代码检查

### Ruff 检查

```bash
# 检查代码
uv run ruff check .

# 自动修复问题
uv run ruff check . --fix

# 格式化代码
uv run ruff format .
```

### 类型检查（可选）

```bash
# 使用 mypy 进行类型检查
uv pip install mypy
mypy agent/
```

## Git 工作流

### 分支命名

```bash
# 功能分支
feature/add-custom-tool

# 修复分支
bugfix/fix-context-trim

# 文档分支
docs/update-api-reference

# 重构分支
refactor/simplify-handler
```

### 提交信息

```bash
# 格式: <type>(<scope>): <subject>

feat(agent): add support for custom constraints
fix(context): resolve memory leak in history trimming
docs(readme): update installation instructions
refactor(tools): simplify MCP tool registration
test(agent): add unit tests for QA handler
style(format): apply ruff formatting
```

### 提交前检查清单

- [ ] 代码通过 ruff 检查
- [ ] 所有测试通过
- [ ] 类型注解完整
- [ ] 文档字符串更新
- [ ] 提交信息符合规范

## 添加新功能

### 添加新 MCP 工具

**步骤 1**: 创建工具文件

```python
# tool/mcp_servers/my_tool.py
from fastmcp import FastMCP

mcp = FastMCP("my-tool")

@mcp.tool()
async def my_function(param: str) -> str:
    """工具描述"""
    return f"Result: {param}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

**步骤 2**: 添加测试

```python
# test/test_my_tool.py
import pytest

@pytest.mark.asyncio
async def test_my_tool():
    from tool.mcp_servers.my_tool import my_function
    
    result = await my_function("test")
    assert "Result: test" in result
```

**步骤 3**: 更新文档

在 `docs/mcp-tools.md` 中添加新工具的说明。

### 添加新 Agent 功能

**步骤 1**: 修改核心代码

```python
# agent/super_react_agent.py
class SuperReActAgent:
    async def new_feature(self, data: str) -> str:
        """新功能描述"""
        # 实现逻辑
        return result
```

**步骤 2**: 添加配置选项

```python
# agent/super_config.py
class SuperAgentConfig:
    enable_new_feature: bool = Field(
        default=False,
        description="启用新功能"
    )
```

**步骤 3**: 添加测试和文档

## 调试技巧

### 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(level=logging.DEBUG)

# 或在代码中
from openjiuwen.core.common.logging import logger
logger.setLevel(logging.DEBUG)
```

### 调试 Agent 执行

```python
# 添加断点
async def process_input(self, question: str):
    import pdb; pdb.set_trace()  # 设置断点
    
    # 或者使用 IPython
    from IPython import embed; embed()
    
    # 继续执行
    ...
```

### 检查上下文

```python
# 打印上下文状态
print(f"History length: {len(context.get_history())}")
print(f"Last message: {context.get_history()[-1]}")

# 保存上下文到文件
import json
with open("context_debug.json", "w") as f:
    json.dump(context.get_history(), f, indent=2)
```

## 性能优化

### 分析性能瓶颈

```python
import cProfile
import pstats

# 性能分析
profiler = cProfile.Profile()
profiler.enable()

# 运行代码
result = await agent.process_input(question)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # 打印前 20 个
```

### 内存分析

```python
import tracemalloc

tracemalloc.start()

# 运行代码
result = await agent.process_input(question)

# 获取内存使用
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024 / 1024:.2f} MB")
print(f"Peak: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

## 发布流程

### 版本号规范

遵循 [SemVer](https://semver.org/)：

- **MAJOR**: 不兼容的 API 变更
- **MINOR**: 向后兼容的功能添加
- **PATCH**: 向后兼容的问题修复

### 发布步骤

1. **更新版本号**

```bash
# pyproject.toml
version = "0.2.0"
```

2. **更新 CHANGELOG**

```markdown
## [0.2.0] - 2025-02-11

### Added
- 新功能 A
- 新功能 B

### Fixed
- 修复问题 C
```

3. **创建 Git Tag**

```bash
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0
```

4. **构建发布包**

```bash
uv build
```

## 贡献规范

### Pull Request 流程

1. **Fork 仓库**

```bash
git clone <your-fork-url>
git checkout -b feature/my-feature
```

2. **进行更改**

```bash
# 编写代码
# 添加测试
# 更新文档
```

3. **提交前检查**

```bash
# 运行测试
uv run pytest

# 代码检查
uv run ruff check .

# 格式化
uv run ruff format .
```

4. **提交 PR**

- 清晰的 PR 标题
- 详细的描述
- 关联相关 Issue

### Code Review 标准

- **正确性**: 代码逻辑正确
- **可读性**: 易于理解和维护
- **测试**: 有足够的测试覆盖
- **文档**: 文档完整清晰
- **性能**: 没有明显性能问题

## 联系方式

如有问题或建议：

- 提交 GitHub Issue
- 查看现有 Issues
- 查看文档

## 许可证

项目使用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。
