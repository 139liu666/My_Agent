import re
from llm_client import AgentsLLM
from tools import ToolExecutor, search

REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

class ReActAgent:
    def __init__(self, llm_client: AgentsLLM, tool_executor: ToolExecutor, max_steps: int = 5):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(tools=tools_desc, question=question, history=history_str)

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                return None

            thought, action = self._parse_output(response_text)
            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("警告：未能解析出有效的Action，流程终止。")
                return None

            tool_name, tool_input = self._parse_action(action)
            if tool_name == "Finish" and tool_input:
                separator = "═" * 60
                print(f"\n{separator}\n✅ 最终回答\n{separator}\n\n{tool_input}\n\n{separator}\n")
                return tool_input

            if not tool_name or not tool_input:
                self.history.append(f"Action: {action}")
                self.history.append("Observation: 无效的Action格式，请检查。")
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            observation = self.tool_executor.execute(tool_name, tool_input)

            print(f"👀 观察: {observation}")
            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"^\s*Thought:\s*(.*?)(?=\n\s*Action:|\Z)", text, re.DOTALL | re.MULTILINE)
        # Action: 匹配到文本末尾
        action_match = re.search(r"^\s*Action:\s*(.*?)\Z", text, re.DOTALL | re.MULTILINE)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.fullmatch(r"(\w+)\[(.*)\]", action_text.strip(), re.DOTALL)
        return (match.group(1), match.group(2).strip()) if match else (None, None)


def main():
    llm = AgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    print("请输入问题，输入 exit、quit 或 退出结束。每个问题独立处理。")
    while True:
        try:
            question = input("\n你的问题：").strip()
            if question.lower() in {"exit", "quit", "退出"}:
                print("已退出。")
                break
            if not question:
                continue
            agent.run(question)
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
if __name__ == '__main__':
    main()
