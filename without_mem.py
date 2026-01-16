import os
import sys
import argparse
import asyncio
import logging
import json
from typing import Dict, List, Tuple
from pathlib import Path
from contextlib import AsyncExitStack
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from src.llm_provider import ChatBackend, OpenAIBackend, TogetherAIBackend
from utils import setup_logging, load_dataset, calculate_metrics


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

ReAct_SYSTEM_PROMPT = """You are a ReAct (Reasoning and Acting) agent for fact verification.

Query: {query}

Background: {background}

Previous steps: {history}

Available tools: {tools}

CRITICAL: You MUST respond with ONLY a valid JSON object. No explanations, no markdown, no text before or after the JSON.

If using a tool:
{{"thought": "your reasoning", "action": {{"name": "tool_name", "reason": "why", "input": "query"}}}}

If providing final answer:
{{"thought": "your reasoning", "answer": "True"}}
or
{{"thought": "your reasoning", "answer": "False"}}

Rules:
- Output ONLY the JSON object, nothing else
- Do NOT wrap JSON in markdown code blocks
- Do NOT add any text before or after the JSON
- When you have sufficient evidence, provide final answer immediately
- Current iteration: this may be your last chance to answer"""


def extract_json_from_text(text: str) -> dict:
    """
    Extract JSON object from text that may contain non-JSON content.

    Tries multiple strategies:
    1. Parse the entire text as JSON
    2. Remove markdown code fences and parse
    3. Find JSON object with 'answer' key using regex
    4. Find any JSON object using regex

    Args:
        text: Text that may contain a JSON object

    Returns:
        Parsed JSON dict, or None if no valid JSON found
    """
    import re

    text = text.strip()

    # Strategy 1: Try parsing the entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Remove markdown code fences
    if text.startswith("```"):
        cleaned = re.sub(r"^```[^\n]*\n", "", text)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find JSON with 'answer' key (highest priority for final answers)
    # Match JSON objects that contain "answer" key
    answer_pattern = r'\{[^{}]*"answer"\s*:\s*"[^"]*"[^{}]*\}'
    match = re.search(answer_pattern, text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find JSON with 'action' key
    action_pattern = r'\{[^{}]*"action"\s*:\s*\{[^{}]*\}[^{}]*\}'
    match = re.search(action_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Strategy 5: Find any complete JSON object (greedy match for nested objects)
    # This handles more complex nested structures
    brace_count = 0
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                json_candidate = text[start_idx:i+1]
                try:
                    return json.loads(json_candidate)
                except json.JSONDecodeError:
                    start_idx = None
                    continue

    return None


class CheckAgent:

    def __init__(self, meta_model: str, exec_model: str):
        """
        Initialize the hierarchical client.

        Args:
            meta_model: Model name for the decompose_agent
            exec_model: Model name for the executor agent
        """
        self.max_react_cycles = 5
        
        if meta_model in ["gpt-4o", "gpt-5-mini", "gpt-5.2"]:
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

        if exec_model in ["gpt-4o", "gpt-5-mini", "gpt-5.2"]:
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
        #self.shared_history: List[Dict[str, str]] = [] # working memory management
    
        
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
        
        tool_names = ", ".join(self.sessions.keys())
        tool_cal_num = 0
        
        history = []  # Reasoning history for this query
        
        for current_step in range(self.max_react_cycles):
            logging.info(f"{'='*50}")
            logging.info(f"ReAct iteration {current_step + 1}/{self.max_react_cycles}")

            # Format the prompt with current state
            history_str = "\n".join(history) if history else "None yet."
            background_str = json.dumps(background, ensure_ascii=False)

            prompt = ReAct_SYSTEM_PROMPT.format(
                query=query,
                history=history_str,
                tools=tool_names,
                background=background_str,
                max_iterations=self.max_react_cycles,
                
            )

            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Continue reasoning about: {query}"}
            ]

            response = self.react_agent.chat(messages)
            content = response.get("content", "")

            if content:
                parsed = extract_json_from_text(content)

                if parsed:
                    thought = parsed.get("thought", "")

                    # Log the thought from LLM response
                    if thought:
                        logging.info(f"Thought: {thought[:1000]}...")

                    # Check if we have a final answer (highest priority)
                    if "answer" in parsed:
                        history.append(f"Thought: {thought}")
                        history.append(f"Answer: {parsed['answer']}")
                        logging.info(f"ReAct completed with answer: {parsed['answer']}")
                        return parsed["answer"], history, tool_cal_num

                    # Handle action decision
                    if "action" in parsed:
                        action = parsed["action"]
                        tool_name = action.get("name", "").lower()
                        tool_input = action.get("input", query)
                        action_reason = action.get("reason", "")

                        history.append(f"Thought: {thought}")

                        if tool_name == "none" or not tool_name:
                            logging.info("Action: No tool needed, continuing reasoning.")
                            history.append("Action: No tool needed, continuing reasoning.")
                            continue

                        # Error hanlde. Sometimes, tool from LLM response may not match exactly
                        tool_mapping = {
                            "wikipedia": "search_wikipedia",
                            "google": "search_google",
                            "search_wikipedia": "search_wikipedia",
                            "search_google": "search_google",
                        }

                        actual_tool = tool_mapping.get(tool_name, tool_name)

                        if actual_tool in self.sessions:
                            logging.info(f"Action: Calling tool '{actual_tool}' with input: {tool_input}")
                            tool_cal_num += 1
                            if action_reason:
                                logging.info(f"Reason: {action_reason}")
                            result = await self._call_tool(actual_tool, {"query": tool_input})
                            observation_truncated = result[:2000]
                            history.append(f"Action: Used {actual_tool} with input: {tool_input}")
                            history.append(f"Observation: {observation_truncated}")
                            logging.info(f"Observation: {observation_truncated[:1000]}...")
                        else:
                            logging.warning(f"Tool '{tool_name}' not found. Available: {tool_names}")
                            history.append(f"Action: Tool '{tool_name}' not found. Available: {tool_names}")
                    else:
                        # No action specified, just thought
                        history.append(f"Thought: {thought}")
                        logging.info("No action in response, continuing to next iteration.")
                else:
                    # Could not extract valid JSON - treat content as thought
                    logging.warning(f"Failed to extract JSON from response: {content[:500]}...")
                    history.append(f"Thought: {content[:1000]}")

        # Max iterations reached
        logging.warning("ReAct loop reached maximum iterations.")
        
        return "True", history, tool_cal_num


    # ---------- Main processing ----------
    async def process_claim(self, idx, claim: str) -> Dict:
        """
        Process a claim through the full fact-checking pipeline.

        Args:
            claim: The claim to verify

        Returns:
            Dictionary with decomposition results and verification answer
        """
        logging.info(f"{'@'*50}")
        logging.info(f"Processing claim {idx}: {claim}")

        # Step 1: Decompose the claim
        decomposition = await self.decompose(claim)
        logging.info(f"Decomposition: {decomposition}")

        # Step 2: Run ReAct loop with decomposition as background
        answer, history, tool_cal_num = await self.react_loop(claim, decomposition)
        logging.info("ReAct history: %s", " | ".join(history))

        return {
            "claim": claim,
            "decomposition": decomposition,
            "history": history,
            "answer": answer,
            "tool_calls": tool_cal_num
        }


    async def cleanup(self):
        """Clean up resources and close MCP sessions."""
        if hasattr(self, "exit_stack"):
            await self.exit_stack.aclose()




# ---------- Main entry point ----------

async def main():
    
    parser = argparse.ArgumentParser(description="Fact-Checking Multi-Agent System")
    parser.add_argument("--dataset", default="factcheckbench", 
                        choices=["factscore", "factool_qa", "factcheckbench", "bingcheck"], help="Dataset")
    parser.add_argument("-m", "--meta_model", type=str, default="gpt-4o",
                        help="Model for decompose agent")
    parser.add_argument("-e", "--exec_model", type=str, default="Qwen/Qwen2.5-72B-Instruct-Turbo",
                        help="Model for ReAct agent")
    parser.add_argument("--log-file", type=str, default="logs/",
                        help="Path to write detailed logs")
    args = parser.parse_args()
    
    # Initialize logging
    log_file_path = os.path.join(args.log_file, f"{args.dataset}.log")
    setup_logging(log_file_path)
    
    # Load dataset
    all_data, label_mapping = load_dataset(args.dataset)

    # Paths to MCP tool server scripts
    server_path = [
        "src/server/search_tool.py",
        "src/server/wikipedia_tool.py",
    ]

    # Initialize Agent
    agent = CheckAgent(args.meta_model, args.exec_model)
    await agent.connect_to_servers(server_path)

    predictions = []
    labels = []
    total_tool_calls = 0
    for i in range(len(all_data)):
        
        claim = all_data[i]['claim']
        label = all_data[i]['label']
        result = await agent.process_claim(i, claim)
        # print("\n" + "="*60)
        # print("FACT-CHECK RESULT")
        # print("="*60)
        # print(f"Claim: {result['claim']}")
        # print(f"Decomposition: {json.dumps(result['decomposition'], indent=2)}")
        # print(f"Answer: {result['answer']}")
        # print("ReAct Trace:")
        # for item in result.get("history", []):
        #     print(f"- {item}")
        
        all_data[i]['decomposition'] = result['decomposition']
        all_data[i]['prediction'] = result['answer']
        all_data[i]['react_trace'] = result.get("history", [])
        
        total_tool_calls += result.get("tool_calls")
        predictions.append(result['answer'])
        labels.append(label)

    await agent.cleanup()
    
    print(f"\nAverage tool calls per claim: {total_tool_calls / len(all_data):.2f}")
    print(f"Total tool calls: {total_tool_calls}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{args.dataset}_qwen-72b_without_memory.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    

if __name__ == "__main__":
    asyncio.run(main())