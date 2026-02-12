# MCP 工具指南

本文档介绍 DeepAgent 支持的所有 MCP (Model Context Protocol) 工具及其使用方法。

## MCP 概述

MCP (Model Context Protocol) 是连接 LLM 和外部工具的标准协议，支持两种传输方式：

- **STDIO** - 标准输入输出（本地进程通信）
- **SSE** - Server-Sent Events（HTTP 流通信）

## 工具列表

### 1. 浏览器工具 (Browser)

**文件**: `tool/mcp_servers/browser_use_mcp_server.py`

**功能**: 自动化浏览器操作，支持网页浏览、表单填写、数据提取等。

**支持的模型**: Gemini、GPT、Claude

**使用方法**:

```python
from mcp import StdioServerParameters

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
```

**工具调用示例**:

```python
# Agent 会自动调用
result = await agent.process_input(
    "Search for 'latest AI news 2025' and summarize the first 3 articles"
)
```

**环境变量**:

```bash
CHROME_PATH=/usr/bin/google-chrome
CHROME_USER_PROFILE_DIR=/path/to/profile
OPENROUTER_API_KEY=your_key
GEMINI_API_KEY=your_key
```

### 2. 搜索工具 (Search)

**文件**: `tool/mcp_servers/searching_mcp_server.py`

**功能**: 多搜索引擎支持，包括 Google、Bing、Perplexity 等。

**支持的搜索服务**:
- Google Search (Serper)
- Jina Deep Search
- Perplexity Search

**使用方法**:

```python
searching_params = StdioServerParameters(
    command="npx",
    args=["-y", "serper-search-scrape-mcp-server"],
    env={
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY"),
        "JINA_API_KEY": os.getenv("JINA_API_KEY"),
        "PERPLEXITY_API_KEY": os.getenv("PERPLEXITY_API_KEY"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
    }
)
```

**环境变量**:

```bash
SERPER_API_KEY=your_serper_key
JINA_API_KEY=your_jina_key
PERPLEXITY_API_KEY=your_perplexity_key
```

### 3. Python 执行工具 (Python)

**文件**: `tool/mcp_servers/python_server.py`

**功能**: 在沙箱环境中执行 Python 代码，支持数据分析和计算。

**依赖**: E2B Code Interpreter

**使用方法**:

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

**配置选项**:

```python
# Python 服务器配置
DEFAULT_TIMEOUT = 1200          # 默认超时（秒）
COMMAND_MAX_RETRIES = 5         # 命令重试次数
NETWORK_TIMEOUT_SECONDS = 20    # 网络超时
SANDBOX_HOME = "/home/user"     # 沙箱主目录
```

**环境变量**:

```bash
E2B_API_KEY=your_e2b_key
E2B_TEMPLATE_ID=your_template_id
```

### 4. 视觉工具 (Vision)

**文件**: `tool/mcp_servers/vision_mcp_server.py`

**功能**: 图像分析和 OCR 识别。

**使用方法**:

```python
# Agent 会自动处理图像输入
result = await agent.process_input(
    "Analyze this image and describe what you see",
    image_path="path/to/image.jpg"
)
```

**支持的模型**: GPT-4 Vision、Claude Vision、Gemini Vision

### 5. 音频工具 (Audio)

**文件**: `tool/mcp_servers/audio_mcp_server.py`

**功能**: 音频处理和语音识别。

**使用方法**:

```python
# Agent 会自动处理音频输入
result = await agent.process_input(
    "Transcribe this audio file",
    audio_path="path/to/audio.mp3"
)
```

**环境变量**:

```bash
OPENAI_API_KEY=your_openai_key
ACR_ACCESS_KEY=your_acr_key        # 可选
ACR_ACCESS_SECRET=your_acr_secret  # 可选
```

### 6. 阅读工具 (Reading)

**文件**: `tool/mcp_servers/reading_mcp_server.py`

**功能**: 文档解析，支持 PDF、Word、PPT 等格式。

**使用方法**:

```python
# Agent 会自动处理文档
result = await agent.process_input(
    "Extract the key points from this PDF",
    document_path="path/to/document.pdf"
)
```

### 7. 推理工具 (Reasoning)

**文件**: `tool/mcp_servers/reasoning_mcp_server.py`

**功能**: 高级推理能力，用于复杂问题求解。

**使用方法**:

```python
# 通过 QAHandler 自动调用
# 不需要手动配置
```

### 8. Doubter 工具

**文件**: `tool/mcp_servers/doubter.py`

**功能**: 验证和质疑 Agent 的输出，提高准确性。

**使用方法**:

```python
# 集成在主 Agent 中
# 自动验证工具调用结果
```

## 工具配置模式

### STDIO 模式

适合本地运行的工具，通过标准输入输出通信。

**优点**:
- 简单直接
- 适合本地开发
- 无需网络配置

**配置示例**:

```python
from mcp import StdioServerParameters

stdio_params = StdioServerParameters(
    command="python",
    args=["path/to/server.py", "--arg1", "value1"],
    env={"KEY": "value"},
    cwd="/working/directory"
)
```

### SSE 模式

适合长期运行的服务，通过 HTTP SSE 通信。

**优点**:
- 可独立部署
- 支持远程调用
- 更好的性能

**配置示例**:

```python
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig

sse_config = ToolServerConfig(
    server_path="http://localhost:8930/sse",
    transport="sse"
)
```

## 在 Agent 中使用工具

### 自动工具发现

```python
from agent.super_react_agent import SuperReActAgent

# Agent 自动发现和注册可用的 MCP 工具
agent = SuperReActAgent(agent_config=config)

# 执行时会自动选择合适的工具
result = await agent.process_input("Search for Python tutorials")
```

### 手动工具配置

```python
from agent.super_react_agent import SuperReActAgent
from openjiuwen.core.utils.tool.mcp.base import ToolServerConfig

# 手动配置工具
tool_configs = [
    ToolServerConfig(
        name="browser",
        server_path="http://localhost:8930/sse",
        transport="sse"
    ),
    ToolServerConfig(
        name="search",
        server_path="stdio:python tool/mcp_servers/searching_mcp_server.py",
        transport="stdio"
    )
]

agent = SuperReActAgent(
    agent_config=config,
    tool_configs=tool_configs
)
```

## 工具开发指南

### 创建自定义 MCP 工具

**步骤 1**: 创建工具文件

```python
# tool/mcp_servers/my_custom_tool.py
from fastmcp import FastMCP

mcp = FastMCP("my-custom-tool")

@mcp.tool()
async def my_function(param1: str, param2: int) -> str:
    """
    工具描述
    
    Args:
        param1: 参数1描述
        param2: 参数2描述
    
    Returns:
        结果描述
    """
    # 工具逻辑
    result = f"Processed {param1} with {param2}"
    return result

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse", host=args.host, port=args.port)
```

**步骤 2**: 注册工具

```python
from mcp import StdioServerParameters

my_tool_params = StdioServerParameters(
    command="python",
    args=["tool/mcp_servers/my_custom_tool.py"],
    env={}
)

# 添加到 Agent 配置
agent = SuperReActAgent(
    agent_config=config,
    tool_configs=[my_tool_params]
)
```

### 工具最佳实践

1. **清晰的描述**: 工具描述应该清晰说明用途
2. **参数验证**: 在工具内部验证输入参数
3. **错误处理**: 返回友好的错误信息
4. **超时处理**: 设置合理的超时时间
5. **日志记录**: 记录工具执行情况

```python
@mcp.tool()
async def safe_search(query: str, max_results: int = 10) -> str:
    """
    安全的网络搜索工具
    
    Args:
        query: 搜索查询（不能为空）
        max_results: 最大结果数（1-50）
    
    Returns:
        JSON 格式的搜索结果
    """
    try:
        # 参数验证
        if not query or not query.strip():
            return json.dumps({"error": "Query cannot be empty"})
        
        max_results = max(1, min(50, max_results))
        
        # 执行搜索
        results = await perform_search(query, max_results)
        
        return json.dumps({"results": results})
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return json.dumps({"error": str(e)})
```

## 故障排查

### 工具无法启动

**问题**: MCP 工具进程无法启动

**解决方案**:
1. 检查 Python 版本（工具环境需要 Python 3.12+）
2. 验证依赖安装: `uv pip install --no-deps -r requirements.txt`
3. 检查环境变量配置
4. 查看工具日志

### 工具调用超时

**问题**: 工具调用超时

**解决方案**:
1. 增加超时配置: `timeout=1200`
2. 优化工具逻辑
3. 检查网络连接（SSE 模式）

### 工具返回错误

**问题**: 工具返回错误信息

**解决方案**:
1. 查看详细错误日志
2. 验证 API 密钥
3. 检查输入参数格式

## 环境要求

### 浏览器工具

- Chrome/Chromium 浏览器
- Playwright: `playwright install`
- 足够的内存（推荐 4GB+）

### Python 执行工具

- E2B API 密钥
- 网络连接（连接 E2B 沙箱）

### 搜索工具

- 至少一个搜索引擎 API 密钥
- Node.js（用于 npx 工具）

## 性能优化

### 工具缓存

```python
# 启用工具结果缓存
config = SuperAgentConfig(
    keep_tool_result=10,  # 缓存最近 10 个结果
)
```

### 并发执行

```python
# 并发调用多个工具
results = await asyncio.gather(
    tool1.execute(),
    tool2.execute(),
    tool3.execute()
)
```

### 资源清理

```python
# 及时清理资源
await agent.cleanup()
```
