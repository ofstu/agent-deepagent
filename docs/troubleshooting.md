# 故障排查

本文档帮助解决使用 DeepAgent 时遇到的常见问题。

## 安装问题

### uv sync 失败

**问题**: `uv sync` 命令失败

**解决方案**:

```bash
# 1. 确保 uv 已安装
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 清除缓存并重试
uv cache clean
uv sync

# 3. 检查 Python 版本（需要 3.11+）
python --version

# 4. 手动创建虚拟环境
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r pyproject.toml
```

### tiktoken 安装失败

**问题**: `uv pip install tiktoken` 失败

**解决方案**:

```bash
# 1. 确保在虚拟环境中
source .venv/bin/activate

# 2. 使用 pip 安装
pip install tiktoken

# 3. 或从源码安装
pip install git+https://github.com/openai/tiktoken.git

# 4. 检查 Rust 编译器（某些平台需要）
rustc --version
```

### MCP 工具环境安装失败

**问题**: 工具环境依赖安装失败

**解决方案**:

```bash
cd tool/

# 1. 确保 Python 3.12+
python3.12 --version

# 2. 创建干净的虚拟环境
rm -rf .venv-tool
uv venv .venv-tool --python 3.12

# 3. 激活环境
source .venv-tool/bin/activate

# 4. 安装依赖
uv pip install --no-deps -r requirements.txt

# 5. 如果失败，尝试不使用 --no-deps
uv pip install -r requirements.txt
```

## 运行问题

### 导入错误

**问题**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:

```bash
# 1. 确保在项目根目录
cd /path/to/deepagent

# 2. 检查 PYTHONPATH
export PYTHONPATH=/path/to/deepagent:$PYTHONPATH

# 3. 重新安装依赖
uv sync

# 4. 检查虚拟环境
which python
# 应该指向 .venv/bin/python
```

### 环境变量缺失

**问题**: 提示缺少 API_KEY 或其他环境变量

**解决方案**:

```bash
# 1. 检查 .env 文件是否存在
ls -la .env

# 2. 创建 .env 文件
cp .env.example .env

# 3. 编辑并填入密钥
nano .env

# 4. 加载环境变量
export $(cat .env | xargs)

# 5. 或在代码中显式加载
from dotenv import load_dotenv
load_dotenv()  # 确保在文件开头调用
```

### Agent 无法启动

**问题**: Agent 初始化失败

**常见原因和解决方案**:

```python
# 问题 1: 配置错误
try:
    config = SuperAgentFactory.create_main_agent_config(...)
except Exception as e:
    print(f"Config error: {e}")
    # 检查所有必需参数

# 问题 2: API 密钥无效
try:
    agent = SuperReActAgent(agent_config=config)
except Exception as e:
    print(f"Agent init error: {e}")
    # 验证 API 密钥
```

## MCP 工具问题

### 浏览器工具无法启动

**问题**: Chrome 浏览器无法启动或连接

**解决方案**:

```bash
# 1. 检查 Chrome 安装
which google-chrome
# 或
which chromium-browser

# 2. 设置正确的路径
export CHROME_PATH=/usr/bin/google-chrome

# 3. 安装 Playwright
playwright install chromium

# 4. 检查 Chrome 版本
google-chrome --version  # 需要较新版本

# 5. 尝试无头模式（如果支持）
export HEADLESS=true
```

### MCP 工具超时

**问题**: 工具调用超时

**解决方案**:

```python
# 1. 增加超时配置
config = SuperAgentConfig(
    timeout=1200,  # 20 分钟
)

# 2. 检查工具服务是否运行
# SSE 模式
curl http://localhost:8930/sse

# 3. 检查日志
tail -f logs/mcp_server.log

# 4. 简化任务
# 将复杂任务分解为多个简单任务
```

### 搜索工具无结果

**问题**: 搜索工具返回空结果

**解决方案**:

```bash
# 1. 检查 API 密钥
echo $SERPER_API_KEY

# 2. 测试 API 是否工作
curl -X POST https://google.serper.dev/search \
  -H "X-API-KEY: $SERPER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"q": "test"}'

# 3. 检查网络连接
ping google.com

# 4. 使用备用搜索引擎
# 配置 JINA_API_KEY 或 PERPLEXITY_API_KEY
```

### Python 执行工具失败

**问题**: Python 代码执行失败

**解决方案**:

```bash
# 1. 检查 E2B API 密钥
echo $E2B_API_KEY

# 2. 验证网络连接（需要访问 E2B 服务）
ping e2b.dev

# 3. 检查模板 ID
echo $E2B_TEMPLATE_ID

# 4. 增加超时
export PYTHON_EXEC_TIMEOUT=1200
```

## LLM 问题

### API 调用失败

**问题**: LLM API 调用返回错误

**常见错误和解决方案**:

```python
# 错误 1: 401 Unauthorized
# 解决方案: 检查 API 密钥
api_key = os.getenv("API_KEY")
if not api_key:
    raise ValueError("API_KEY not set")

# 错误 2: 429 Rate Limit
# 解决方案: 添加延迟或重试
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=10),
    stop=stop_after_attempt(3)
)
async def call_with_retry():
    return await llm.generate(messages)

# 错误 3: 503 Service Unavailable
# 解决方案: 更换模型或等待
config = SuperAgentConfig(
    model_name="anthropic/claude-3.5-sonnet",  # 备用模型
)
```

### 上下文超出限制

**问题**: `ContextLimitError` 上下文超出限制

**解决方案**:

```python
# 1. 启用上下文重试
config = SuperAgentConfig(
    enable_context_limit_retry=True,
)

# 2. 减少保留的历史消息数
config = SuperAgentConfig(
    keep_tool_result=5,  # 只保留最近 5 个工具结果
)

# 3. 减少最大迭代次数
config = SuperAgentConfig(
    max_iteration=10,
)

# 4. 手动清理历史
agent.reset()  # 清除所有历史
```

### 模型输出质量差

**问题**: LLM 输出不符合预期

**解决方案**:

```python
# 1. 调整温度参数
llm = OpenRouterLLM(
    api_key="...",
    temperature=0.0,  # 降低随机性
)

# 2. 使用更强的模型
config = SuperAgentConfig(
    model_name="anthropic/claude-3-opus",  # 更强的模型
)

# 3. 添加任务指导
config = SuperAgentConfig(
    task_guidance="""
    Be precise and thorough in your answers.
    Always verify facts before stating them.
    """,
)

# 4. 启用推理增强
config = SuperAgentConfig(
    enable_question_hints=True,
    enable_extract_final_answer=True,
)
```

## 性能问题

### Agent 执行缓慢

**问题**: Agent 执行任务时间过长

**优化建议**:

```python
# 1. 限制迭代次数
config = SuperAgentConfig(
    max_iteration=10,  # 减少最大迭代次数
)

# 2. 限制工具调用
config = SuperAgentConfig(
    max_tool_calls_per_turn=3,  # 减少每轮工具调用
)

# 3. 使用更快的模型
config = SuperAgentConfig(
    model_name="anthropic/claude-3.5-sonnet",  # 更快的模型
)

# 4. 禁用不必要的功能
config = SuperAgentConfig(
    enable_question_hints=False,  # 如果不需要
    enable_todo_plan=False,
)
```

### 内存占用过高

**问题**: 程序占用内存过多

**解决方案**:

```python
# 1. 限制上下文长度
context = ContextManager(
    max_history_length=30,  # 减少历史长度
)

# 2. 定期清理
async def process_batch(questions):
    for i, question in enumerate(questions):
        result = await agent.process_input(question)
        
        # 每 10 个问题清理一次
        if i % 10 == 0:
            agent.reset()

# 3. 使用生成器而不是列表
# 不好
results = [await process(q) for q in questions]

# 好
async for result in process_generator(questions):
    yield result
```

## 调试技巧

### 启用详细日志

```python
import logging

# 设置日志级别
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 或只启用特定模块
logging.getLogger('agent.super_react_agent').setLevel(logging.DEBUG)
logging.getLogger('agent.context_manager').setLevel(logging.DEBUG)
```

### 跟踪 Agent 执行

```python
# 添加自定义回调
class DebugCallback:
    async def on_thought(self, thought: str):
        print(f"[THOUGHT] {thought}")
    
    async def on_action(self, action: str):
        print(f"[ACTION] {action}")
    
    async def on_observation(self, observation: str):
        print(f"[OBSERVATION] {observation}")

# 在 Agent 中使用
callback = DebugCallback()
agent.register_callback(callback)
```

### 保存执行状态

```python
import json

# 保存对话历史
history = agent.context_manager.get_history()
with open("conversation.json", "w") as f:
    json.dump(history, f, indent=2)

# 保存 Agent 状态
state = agent.get_state()
with open("agent_state.json", "w") as f:
    json.dump(state, f, indent=2)
```

### 使用 PDB 调试

```python
# 在代码中添加断点
async def process_input(self, question: str):
    import pdb; pdb.set_trace()
    
    # 或使用 IPython
    from IPython import embed; embed()
    
    # 继续执行
    ...
```

## 常见错误代码

### Error Code 参考

| 错误代码 | 描述 | 解决方案 |
|---------|------|---------|
| E001 | API 密钥缺失 | 检查环境变量 |
| E002 | 上下文超限 | 启用重试或减少历史 |
| E003 | 工具调用超时 | 增加超时设置 |
| E004 | MCP 连接失败 | 检查服务状态 |
| E005 | 模型不可用 | 更换模型或稍后重试 |

### 获取详细错误信息

```python
import traceback

try:
    result = await agent.process_input(question)
except Exception as e:
    print(f"Error: {e}")
    print(f"Traceback:\n{traceback.format_exc()}")
    
    # 检查特定错误类型
    if isinstance(e, ContextLimitError):
        print("Context limit exceeded - try reducing history")
    elif "timeout" in str(e).lower():
        print("Timeout - try increasing timeout settings")
```

## 获取帮助

### 自助排查清单

在提交 Issue 之前，请检查：

- [ ] 所有环境变量已正确设置
- [ ] Python 版本符合要求（3.11+）
- [ ] 依赖包已正确安装
- [ ] 虚拟环境已激活
- [ ] API 密钥有效且有额度
- [ ] 网络连接正常
- [ ] 查看日志获取详细信息

### 提交 Issue

如果问题无法解决，请提交 GitHub Issue 并包含：

1. **环境信息**
   - Python 版本: `python --version`
   - 操作系统: `uname -a`
   - 项目版本: `git log --oneline -1`

2. **配置文件**（去除敏感信息）
   ```bash
   cat .env | grep -v KEY | grep -v SECRET
   ```

3. **复现步骤**
   - 最小复现代码
   - 执行的命令
   - 预期行为 vs 实际行为

4. **错误日志**
   - 完整错误信息
   - 相关日志输出

### 社区支持

- 查看 [GitHub Issues](https://github.com/your-repo/issues)
- 阅读 [文档](./index.md)
- 参考 [示例](./examples.md)
