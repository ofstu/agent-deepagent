# 示例

本文档提供各种使用场景的实际示例。

## 基础示例

### 简单的问答 Agent

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def simple_qa():
    """简单的问答示例"""
    
    # 创建配置
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="qa-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
    )
    
    # 创建 Agent
    agent = SuperReActAgent(agent_config=config)
    
    # 问答
    questions = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = await agent.process_input(question)
        print(f"A: {answer}")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(simple_qa())
```

### 启用推理增强

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def enhanced_qa():
    """使用推理模型增强的 QA"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="enhanced-qa",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        
        # 启用推理增强
        enable_question_hints=True,
        enable_extract_final_answer=True,
        reasoning_model="o3",
        open_api_key=os.getenv("OPENAI_API_KEY"),
    )
    
    agent = SuperReActAgent(agent_config=config)
    
    # 复杂问题
    question = """
    A farmer has 17 sheep and all but 9 die.
    How many sheep are left?
    """
    
    result = await agent.process_input(question)
    print(f"Answer: {result}")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(enhanced_qa())
```

## 研究 Agent

### 网络研究

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def web_research():
    """使用浏览器工具进行网络研究"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="research-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        
        description="Research agent with web browsing capabilities",
        max_iteration=20,
        enable_question_hints=True,
        enable_extract_final_answer=True,
        enable_todo_plan=True,
        
        task_guidance="""
        When researching:
        1. Use search tool to find relevant information
        2. Browse official sources for accuracy
        3. Cross-verify facts from multiple sources
        4. Cite sources in your answer
        """,
    )
    
    agent = SuperReActAgent(agent_config=config)
    
    # 研究任务
    question = """
    Research the latest developments in quantum computing in 2025.
    Focus on:
    - Recent breakthroughs
    - Commercial applications
    - Key companies and researchers
    """
    
    print("🔍 Starting research...")
    result = await agent.process_input(question)
    
    print("\n" + "="*50)
    print("RESEARCH RESULT:")
    print("="*50)
    print(result)
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(web_research())
```

### 数据分析

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def data_analysis():
    """使用 Python 工具进行数据分析"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="data-analyst",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        
        description="Data analysis agent with Python execution",
        max_iteration=15,
        enable_extract_final_answer=True,
    )
    
    agent = SuperReActAgent(agent_config=config)
    
    # 数据分析任务
    question = """
    Calculate the first 50 Fibonacci numbers and find:
    1. The ratio of consecutive numbers (approaching golden ratio)
    2. How many are prime numbers
    3. Visualize the growth pattern
    """
    
    print("📊 Analyzing data...")
    result = await agent.process_input(question)
    
    print("\nAnalysis Result:")
    print(result)
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(data_analysis())
```

## 多 Agent 系统

### 协作 Agent

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentConfig, SuperAgentFactory
from agent.tool_call_handler import ToolCallHandler

load_dotenv()

async def multi_agent_collaboration():
    """多个 Agent 协作完成任务"""
    
    # 创建子 Agent
    browser_config = SuperAgentFactory.create_sub_agent_config(
        agent_name="browser-agent",
        agent_type="browser",
        description="Expert in web browsing and information extraction",
        max_iteration=10,
    )
    browser_agent = SuperReActAgent(agent_config=browser_config)
    
    coder_config = SuperAgentFactory.create_sub_agent_config(
        agent_name="coder-agent",
        agent_type="coder",
        description="Expert in Python programming and data analysis",
        max_iteration=15,
        enable_extract_final_answer=True,
    )
    coder_agent = SuperReActAgent(agent_config=coder_config)
    
    # 创建工具处理器
    tool_handler = ToolCallHandler()
    
    # 为子 Agent 创建工具
    browser_tool = tool_handler.create_sub_agent_tool(
        agent_name="browser-agent",
        sub_agent=browser_agent
    )
    
    coder_tool = tool_handler.create_sub_agent_tool(
        agent_name="coder-agent",
        sub_agent=coder_agent
    )
    
    # 创建主 Agent
    main_config = SuperAgentConfig(
        agent_type="main",
        description="Main coordinator agent",
        max_iteration=15,
        sub_agent_configs={
            "browser-agent": browser_config,
            "coder-agent": coder_config,
        }
    )
    
    main_agent = SuperReActAgent(
        agent_config=main_config,
        tool_call_handler=tool_handler
    )
    
    # 复杂任务
    question = """
    I need to analyze the stock market trends for tech companies in 2025.
    
    Please:
    1. Search for recent stock data of major tech companies
    2. Extract key metrics and trends
    3. Perform statistical analysis
    4. Create visualizations
    5. Provide investment recommendations
    """
    
    print("🎯 Starting multi-agent collaboration...")
    result = await main_agent.process_input(question)
    
    print("\n" + "="*60)
    print("FINAL RESULT:")
    print("="*60)
    print(result)
    
    # 清理
    await main_agent.cleanup()
    await browser_agent.cleanup()
    await coder_agent.cleanup()

if __name__ == "__main__":
    asyncio.run(multi_agent_collaboration())
```

## 高级用例

### 自定义工具集成

```python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory
from openjiuwen.core.utils.tool.function.function import LocalFunction
from openjiuwen.core.utils.tool.param import Param

load_dotenv()

# 定义自定义工具
def custom_calculator(expression: str) -> str:
    """
    安全计算器工具
    
    Args:
        expression: 数学表达式
        
    Returns:
        计算结果
    """
    try:
        # 只允许安全操作
        allowed_names = {
            "abs": abs,
            "max": max,
            "min": min,
            "pow": pow,
            "round": round,
        }
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

# 创建 LocalFunction 包装
calculator_tool = LocalFunction(
    name="calculator",
    description="Safe calculator for mathematical expressions",
    params=[
        Param(
            name="expression",
            type="string",
            description="Mathematical expression to evaluate",
            required=True
        )
    ],
    function=custom_calculator
)

async def custom_tools_example():
    """使用自定义工具"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="custom-tools-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        
        description="Agent with custom calculator tool",
        max_iteration=10,
    )
    
    # 创建 Agent 并添加自定义工具
    agent = SuperReActAgent(agent_config=config)
    agent.add_tool(calculator_tool)
    
    # 使用自定义工具
    question = """
    Calculate:
    1. The factorial of 10
    2. 2 raised to the power of 20
    3. The absolute value of -123.456
    """
    
    result = await agent.process_input(question)
    print(f"Result: {result}")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(custom_tools_example())
```

### 批量处理

```python
import os
import asyncio
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def process_single(agent: SuperReActAgent, question: str) -> dict:
    """处理单个问题"""
    try:
        result = await agent.process_input(question)
        return {
            "question": question,
            "answer": result,
            "status": "success"
        }
    except Exception as e:
        return {
            "question": question,
            "answer": str(e),
            "status": "error"
        }

async def batch_processing():
    """批量处理多个问题"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="batch-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
        max_iteration=10,
    )
    
    agent = SuperReActAgent(agent_config=config)
    
    # 批量问题
    questions = [
        "What is machine learning?",
        "Explain neural networks",
        "What is deep learning?",
        "How does GPT work?",
        "What is reinforcement learning?",
    ]
    
    print(f"Processing {len(questions)} questions...")
    
    # 并发处理
    tasks = [process_single(agent, q) for q in questions]
    results = await asyncio.gather(*tasks)
    
    # 输出结果
    print("\n" + "="*60)
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result['status'].upper()}")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer'][:200]}...")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(batch_processing())
```

### 流式输出

```python
import os
import asyncio
from dotenv import load_dotenv

from llm.openrouter_llm import OpenRouterLLM

load_dotenv()

async def streaming_example():
    """流式生成响应"""
    
    # 创建 LLM 客户端
    llm = OpenRouterLLM(
        api_key=os.getenv("API_KEY"),
        model_name="anthropic/claude-3.7-sonnet",
        temperature=0.1
    )
    
    # 准备消息
    messages = [
        {"role": "user", "content": "Tell me a story about AI"}
    ]
    
    print("Generating story...\n")
    
    # 流式生成
    full_response = ""
    async for chunk in llm.stream_generate(messages):
        content = chunk.content
        full_response += content
        print(content, end="", flush=True)
    
    print(f"\n\nTotal length: {len(full_response)} characters")

if __name__ == "__main__":
    asyncio.run(streaming_example())
```

## 实用脚本

### 交互式 Agent

```python
#!/usr/bin/env python
import os
import asyncio
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

class InteractiveAgent:
    """交互式 Agent 会话"""
    
    def __init__(self):
        self.config = SuperAgentFactory.create_main_agent_config(
            agent_id="interactive-agent",
            api_key=os.getenv("API_KEY"),
            api_base=os.getenv("API_BASE"),
            model_name="anthropic/claude-3.7-sonnet",
            enable_question_hints=True,
        )
        self.agent = SuperReActAgent(agent_config=self.config)
    
    async def chat(self):
        """开始交互会话"""
        print("🤖 DeepAgent Interactive Mode")
        print("Type 'quit' to exit, 'reset' to clear history\n")
        
        while True:
            try:
                # 获取用户输入
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'reset':
                    self.agent.reset()
                    print("History cleared.\n")
                    continue
                elif not user_input:
                    continue
                
                # 处理输入
                print("\n🤔 Thinking...")
                response = await self.agent.process_input(user_input)
                
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")
        
        await self.agent.cleanup()

async def main():
    session = InteractiveAgent()
    await session.chat()

if __name__ == "__main__":
    asyncio.run(main())
```

### 任务评估

```python
import os
import asyncio
import json
from dotenv import load_dotenv

from agent.super_react_agent import SuperReActAgent
from agent.super_config import SuperAgentFactory

load_dotenv()

async def evaluate_agent():
    """评估 Agent 性能"""
    
    config = SuperAgentFactory.create_main_agent_config(
        agent_id="eval-agent",
        api_key=os.getenv("API_KEY"),
        api_base=os.getenv("API_BASE"),
        model_name="anthropic/claude-3.7-sonnet",
    )
    
    agent = SuperReActAgent(agent_config=config)
    
    # 测试数据集
    test_cases = [
        {
            "question": "What is 15 + 27?",
            "expected": "42"
        },
        {
            "question": "Who is the president of the USA?",
            "expected": None  # 开放性问题
        },
        {
            "question": "What is the capital of Japan?",
            "expected": "Tokyo"
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Testing: {test['question']}")
        
        response = await agent.process_input(test['question'])
        
        # 评估
        if test['expected']:
            passed = test['expected'].lower() in response.lower()
        else:
            passed = len(response) > 50  # 开放性问题至少有内容
        
        results.append({
            "question": test['question'],
            "expected": test['expected'],
            "response": response,
            "passed": passed
        })
        
        status = "✅" if passed else "❌"
        print(f"{status} Response: {response[:100]}...")
    
    # 统计
    passed_count = sum(1 for r in results if r['passed'])
    total = len(results)
    accuracy = passed_count / total * 100
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Passed: {passed_count}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")
    
    # 保存结果
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")
    
    await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(evaluate_agent())
```

## 更多示例

查看 `examples/` 目录获取更多完整示例：

- `basic_qa.py` - 基础问答
- `web_research.py` - 网络研究
- `multi_agent.py` - 多 Agent 协作
- `custom_tools.py` - 自定义工具
- `batch_processing.py` - 批量处理

运行示例：

```bash
# 基础问答
uv run python examples/basic_qa.py

# 网络研究
uv run python examples/web_research.py

# 多 Agent
uv run python examples/multi_agent.py
```
