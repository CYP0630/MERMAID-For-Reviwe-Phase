import os
import sys
import asyncio
import logging
import json
from typing import Dict, List, Tuple
from pathlib import Path

from src.llm_provider import ChatBackend, OpenAIBackend, TogetherAIBackend
from dotenv import load_dotenv

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils import setup_logging


load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
together_key = os.getenv("TOGETHER_API_KEY")

    
    
# System prompt for the decomp agent that analyzes and breaks down user input claims
Decomp_SYSTEM_PROMPT = (
    "You are the claim decompose agent in a hierarchical AI Fact-Checking system.\n" 
    "A user will input a claim and please decompose and analysis it.\n"
    "**First**: extract the triplets from the claim.\n"
    "**Second**: analyze the topic of the claim. \n"
    "Reply ONLY in JSON with the schema:\n"
    "{ \"triplets\": [ {\"subject\": \"\", \"relation\": \"\", \"object\": \"\"} ], \"topic\": \"\" }\n"
)

ReAct_SYSTEM_PROMPT = """You are a ReAct (Reasoning and Acting) agent tasked with verifying the following query:

    Query: {query}
    
    Your task is to reason about the claim step by step, using external tools when needed, and ultimately determine whether it is factually **True** or **False**.

    You may also consider relevant context or decomposed background knowledge:
    {background}
    
    You may consult previous reasoning steps and tool results:
    {history}
    
    Available tools: 
    {tools}
    
    Instructions:
    1. Analyze the claim carefully using any available background knowledge, previous reasoning steps, and observations.
    2. Decide whether to take an action (e.g., use a tool) or provide a final answer.
    3. Respond in the following **strict JSON format**:
    
    If you need to use a tool:
        {{
            "thought": "Your detailed reasoning about what to do next",
            "action": {{
                "name": "Tool name (wikipedia, google, or none)",
                "reason": "Explanation of why you chose this tool for this step",
                "input": "Your specific query to the tool (e.g., a sub-claim or entity from triplets)"
            }}
        }}
    
    If you have enough evidence to make a final judgment:
    {{
        "thought": ""Your final reasoning based on the evidence gathered so far",
        "answer": "True or False"
    }}
    
    Remember:
        - Be explicit and logical in your reasoning.
        - Use tools when you need more information.
        - Always base your reasoning on the actual observations from tool use.
        - If a tool returns no results or fails, acknowledge this and consider using a different tool or approach.
        - Provide a final answer only when you're confident you have sufficient information.
        - If you cannot find the necessary information after using available tools, admit that you don't have enough information to answer the query confidently.
    """


class CheckAgent:

    def __init__(self, meta_model: str, exec_model: str):
        """
        Initialize the hierarchical client.

        Args:
            meta_model: Model name for the decompose_agent
            exec_model: Model name for the executor agent
        """
        self.max_react_cycles = 5
        # Select provider per model; OpenAI models use OpenAIBackend, others TogetherAIBackend
        if meta_model in ["gpt-4o", "gpt-5-mini"]:
            self.decompose_agent = OpenAIBackend(
                meta_model,
                key=openai_key,
                url="https://api.openai.com/v1",
            )
        else:
            self.decompose_agent = OpenAIBackend(
                meta_model,
                key=together_key,
                url="https://api.together.xyz/v1",
            )

        if exec_model in ["gpt-4o", "gpt-5-mini"]:
            self.react_agent = OpenAIBackend(
                exec_model,
                key=openai_key,
                url="https://api.openai.com/v1",
            )
        else:
            self.react_agent = OpenAIBackend(
                exec_model,
                key=together_key,
                url="https://api.together.xyz/v1",
            )

        self.exec_model = exec_model
        self.sessions: Dict[str, ClientSession] = {} # tool name in MCP -> session
        self.shared_history: List[Dict[str, str]] = [] # memory management
    
        
    # ---------- Tool management ----------
    async def connect_to_servers(self, scripts: List[str]):
        """
        Connect to MCP tool servers specified by script paths.

        Args:
            scripts: List of paths to tool server scripts

        Raises:
            RuntimeError: If duplicate tool names are found
        """
        self.exit_stack = AsyncExitStack()

        for script in scripts:
            path = Path(script)
            # Determine command based on file extension
            cmd = "python" if path.suffix == ".py" else "node"
            params = StdioServerParameters(command=cmd, args=[str(path)])

            # Create stdio client and session
            stdio, write = await self.exit_stack.enter_async_context(stdio_client(params))
            session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
            await session.initialize()

            # Register tools from this session
            for tool in (await session.list_tools()).tools:
                if tool.name in self.sessions:
                    raise RuntimeError(f"Duplicate tool name '{tool.name}'.")
                self.sessions[tool.name] = session

        logging.info(f"Connected tools: {list(self.sessions.keys())}")


    async def _tools_schema(self) -> List[Dict]:
        """
        Get the schema of all available tools in OpenAI function format.

        Returns:
            List of tool definitions with name, description, and parameters
        """
        all_tools, cached = [], {}
        for session in self.sessions.values():
            tools_resp = cached.get(id(session)) or await session.list_tools()
            cached[id(session)] = tools_resp
            for tool in tools_resp.tools:
                all_tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        },
                    }
                )
        return all_tools


    async def _call_tool(self, tool_name: str, arguments: Dict) -> str:
        """
        Call a tool via its MCP session.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as a string
        """
        if tool_name not in self.sessions:
            return f"Error: Tool '{tool_name}' not found."

        session = self.sessions[tool_name]
        called_tool = await session.call_tool(tool_name, arguments)
        return str(called_tool.content)


    # ---------- Decompose agent ----------
    async def decompose(self, claim: str) -> Dict:
        """
        Decompose a claim into triplets and analyze its topic.

        Args:
            claim: The claim to decompose

        Returns:
            Dictionary containing triplets and topic
        """
        messages = [
            {"role": "system", "content": Decomp_SYSTEM_PROMPT},
            {"role": "user", "content": claim}
        ]

        response = self.decompose_agent.chat(messages)
        content = response.get("content", "")

        # Parse JSON from response
        try:
            # Strip markdown code fences if present
            content = content.strip()
            if content.startswith("```"):
                import re
                content = re.sub(r"^```[^\n]*\n", "", content)
                content = re.sub(r"\n?```$", "", content)
                content = content.strip()

            return json.loads(content)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse decompose response: {content}. Error: {e}")
            return {"triplets": [], "topic": "unknown"}


    # ---------- ReAct loop ----------
    async def react_loop(self, query: str, background: Dict) -> Tuple[str, List[str]]:
        """
        Execute the ReAct (Reasoning and Acting) loop for fact verification.

        Args:
            query: The query/claim to verify
            background: Background information from decompose step (triplets, topic)

        Returns:
            Tuple of (final answer from the ReAct agent, reasoning/action history)
        """
        
        tools_schema = await self._tools_schema()
        tool_names = ", ".join(self.sessions.keys())

        history = []  # Reasoning history for this query
        
        for current_step in range(self.max_react_cycles):
            logging.info(f"ReAct iteration {current_step + 1}/{self.max_react_cycles}")

            # Format the prompt with current state
            history_str = "\n".join(history) if history else "None yet."
            background_str = json.dumps(background, ensure_ascii=False)

            prompt = ReAct_SYSTEM_PROMPT.format(
                query=query,
                history=history_str,
                tools=tool_names,
                background=background_str
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Continue reasoning about: {query}"}
            ]


            # Get response from ReAct agent
            response = self.react_agent.chat(messages, tools=tools_schema)
            content = response.get("content", "")
            tool_calls = response.get("tool_calls")


            # Handle tool calls (OpenAI function calling style)
            if tool_calls:
                for call in tool_calls:
                    func = call.get("function", {})
                    t_name = func.get("name", "")
                    t_args = json.loads(func.get("arguments", "{}"))

                    logging.info(f"Calling tool: {t_name} with args: {t_args}")
                    result = await self._call_tool(t_name, t_args)

                    history.append(f"Action: Used {t_name} with input {t_args}")
                    history.append(f"Observation: {result[:2000]}")  # Truncate long results
                continue


            # Parse JSON response for ReAct decision
            if content:
                try:
                    # Strip markdown code fences if present
                    cleaned = content.strip()
                    if cleaned.startswith("```"):
                        import re
                        cleaned = re.sub(r"^```[^\n]*\n", "", cleaned)
                        cleaned = re.sub(r"\n?```$", "", cleaned)
                        cleaned = cleaned.strip()

                    parsed = json.loads(cleaned)
                    thought = parsed.get("thought", "")

                    # Check if we have a final answer
                    if "answer" in parsed:
                        logging.info(f"ReAct completed with answer: {parsed['answer']}")
                        return parsed["answer"], history

                    # Handle action decision
                    if "action" in parsed:
                        action = parsed["action"]
                        tool_name = action.get("name", "").lower()
                        tool_input = action.get("input", query)

                        history.append(f"Thought: {thought}")

                        if tool_name == "none" or not tool_name:
                            history.append("Action: No tool needed, continuing reasoning.")
                            continue

                        # Map tool names to actual MCP tool names
                        tool_mapping = {
                            "wikipedia": "search_wikipedia",
                            "google": "search_google",
                            "search_wikipedia": "search_wikipedia",
                            "search_google": "search_google",
                        }

                        actual_tool = tool_mapping.get(tool_name, tool_name)

                        if actual_tool in self.sessions:
                            logging.info(f"Calling tool: {actual_tool} with input: {tool_input}")
                            result = await self._call_tool(actual_tool, {"query": tool_input})
                            history.append(f"Action: Used {actual_tool}")
                            history.append(f"Observation: {result[:2000]}")
                        else:
                            history.append(f"Action: Tool '{tool_name}' not found. Available: {tool_names}")
                    else:
                        history.append(f"Thought: {thought}")

                except json.JSONDecodeError:
                    # If response isn't valid JSON, treat as thought
                    history.append(f"Thought: {content}")

        # Max iterations reached
        logging.warning("ReAct loop reached maximum iterations.")
        return (
            f"Could not determine answer after {self.max_react_cycles} iterations. "
            f"History: {'; '.join(history[-4:])}",
            history,
        )


    # ---------- Main processing ----------
    async def process_claim(self, claim: str) -> Dict:
        """
        Process a claim through the full fact-checking pipeline.

        Args:
            claim: The claim to verify

        Returns:
            Dictionary with decomposition results and verification answer
        """
        logging.info(f"Processing claim: {claim}")

        # Step 1: Decompose the claim
        decomposition = await self.decompose(claim)
        logging.info(f"Decomposition: {decomposition}")

        # Step 2: Run ReAct loop with decomposition as background
        answer, history = await self.react_loop(claim, decomposition)
        logging.info("ReAct history: %s", " | ".join(history))

        return {
            "claim": claim,
            "decomposition": decomposition,
            "answer": answer,
            "history": history,
        }


    async def cleanup(self):
        """Clean up resources and close MCP sessions."""
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()


# ---------- Main entry point ----------

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Fact-Checking Multi-Agent System")
    parser.add_argument("-c", "--claim", type=str, help="Claim to verify")
    parser.add_argument("-m", "--meta_model", type=str, default="gpt-4o",
                        help="Model for decompose agent")
    parser.add_argument("-e", "--exec_model", type=str, default="gpt-4o",
                        help="Model for ReAct agent")
    parser.add_argument("--log-file", type=str, default="results/fact_check.log",
                        help="Path to write detailed logs")
    args = parser.parse_args()

    setup_logging(args.log_file)
    
    # Paths to MCP tool server scripts
    server_path = [
        "src/server/search_tool.py",
        "src/server/wikipedia_tool.py",
    ]

    agent = CheckAgent(args.meta_model, args.exec_model)
    await agent.connect_to_servers(server_path)

    try:
        if args.claim:
            result = await agent.process_claim(args.claim)
            print("\n" + "="*60)
            print("FACT-CHECK RESULT")
            print("="*60)
            print(f"Claim: {result['claim']}")
            print(f"Decomposition: {json.dumps(result['decomposition'], indent=2)}")
            print(f"Answer: {result['answer']}")
            print("ReAct Trace:")
            for item in result.get("history", []):
                print(f"- {item}")
        else:
            print("Fact-Checking Multi-Agent System")
            print("Enter 'exit' to quit.\n")
            while True:
                claim = input("Enter claim to verify: ").strip()
                if claim.lower() in {"exit", "quit", "q"}:
                    break
                if claim:
                    result = await agent.process_claim(claim)
                    print("\n" + "-"*40)
                    print(f"Decomposition: {json.dumps(result['decomposition'], indent=2)}")
                    print(f"Answer: {result['answer']}")
                    print("ReAct Trace:")
                    for item in result.get("history", []):
                        print(f"- {item}")
                    print("-"*40 + "\n")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
