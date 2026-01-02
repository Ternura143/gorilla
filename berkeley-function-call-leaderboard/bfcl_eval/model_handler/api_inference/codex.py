"""
Codex handler for BFCL evaluation.

This handler uses OpenAI's Codex CLI tool to solve function calling tasks.
It operates in prompting mode, constructing prompts that include function definitions
and expecting the model to output function calls in Python format.

Two implementations:
1. CodexHandler - Direct text parsing (original approach)
2. CodexFileHandler - File-based approach (similar to Harbor's EvoEval adapter)
"""

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from bfcl_eval.constants.enums import ModelStyle, ReturnFormat
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.utils import (
    default_decode_ast_prompting,
    default_decode_execute_prompting,
    format_execution_results_prompting,
    system_prompt_pre_processing_chat_model,
)


class CodexHandler(BaseHandler):
    """
    Handler for OpenAI Codex CLI integration with BFCL.
    
    Codex is a coding agent that can execute shell commands and write code.
    This handler bridges BFCL's function calling evaluation with Codex's capabilities
    by constructing prompts that describe the available functions and expected output format.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OSSMODEL  # Use open source model style for prompting
        
        # Verify codex CLI is available
        self._verify_codex_installation()
    
    def _verify_codex_installation(self) -> None:
        """Check if codex CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                print(f"Warning: codex CLI returned non-zero exit code: {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                "Codex CLI is not installed. Please install it with: "
                "npm install -g @openai/codex"
            )
        except subprocess.TimeoutExpired:
            print("Warning: codex version check timed out")

    def _build_codex_prompt(self, system_prompt: str, user_message: str) -> str:
        """
        Build a comprehensive prompt for Codex that includes:
        - System instructions (function definitions, output format)
        - User query
        
        The prompt instructs Codex to output function calls in Python format
        that BFCL can parse.
        """
        prompt = f"""You are a function calling assistant. Your task is to analyze the user's request and output the appropriate function call(s).

{system_prompt}

User Request: {user_message}

IMPORTANT: Your response must be ONLY the function call(s) in Python format, nothing else.
For example: [function_name(param1="value1", param2=123)]

Do not include any explanation, just output the function call(s) as a Python list."""
        
        return prompt

    def _run_codex(self, prompt: str) -> tuple[str, float]:
        """
        Execute Codex CLI with the given prompt.
        
        Returns:
            tuple: (output_text, latency_seconds)
        """
        # Get the model name (strip provider prefix if present)
        model = self.model_name
        if "/" in model:
            model = model.split("/")[-1]
        
        # Escape the prompt for shell
        escaped_prompt = shlex.quote(prompt)
        
        # Build the codex command
        # Using exec mode with bypass flags for automation
        # Note: Codex CLI does not support --temperature flag
        cmd = [
            "codex", "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model", model,
            "--json",
            "--",
            prompt,
        ]
        
        # Set up environment with API key
        env = os.environ.copy()
        if "OPENAI_API_KEY" not in env:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=300,  # 5 minute timeout
            )
            
            end_time = time.time()
            latency = end_time - start_time
            
            # Parse the output - codex outputs JSON when using --json flag
            output = result.stdout.strip()
            
            # Try to extract the actual response from JSON output
            response_text = self._parse_codex_output(output)
            
            if result.returncode != 0 and not response_text:
                print(f"Codex error (exit code {result.returncode}): {result.stderr}")
                return result.stderr or "Error executing codex", latency
            
            return response_text, latency
            
        except subprocess.TimeoutExpired:
            return "Error: Codex execution timed out", time.time() - start_time
        except Exception as e:
            return f"Error: {str(e)}", time.time() - start_time

    def _parse_codex_output(self, raw_output: str) -> str:
        """
        Parse the raw output from Codex CLI.
        
        Codex with --json flag outputs structured JSON with message events.
        We need to extract the actual text response.
        
        Expected format (one JSON per line):
        {"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"[function_call()]"}}
        """
        if not raw_output:
            return ""
        
        lines = raw_output.strip().split("\n")
        
        # Collect all text content from the output
        text_parts = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to parse as JSON (codex --json output)
            try:
                data = json.loads(line)
                
                if not isinstance(data, dict):
                    continue
                
                # Handle item.completed event (main response)
                if data.get("type") == "item.completed":
                    item = data.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            text_parts.append(text)
                
                # Handle older format with 'output' field
                elif "output" in data:
                    output_data = data["output"]
                    if isinstance(output_data, list):
                        for item in output_data:
                            if isinstance(item, dict) and item.get("type") == "message":
                                content = item.get("content", [])
                                for c in content:
                                    if isinstance(c, dict) and c.get("type") in ("output_text", "text"):
                                        text_parts.append(c.get("text", ""))
                    elif isinstance(output_data, str):
                        text_parts.append(output_data)
                
                # Handle 'content' field (streaming output)
                elif "content" in data:
                    content = data["content"]
                    if isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict) and "text" in c:
                                text_parts.append(c["text"])
                    elif isinstance(content, str):
                        text_parts.append(content)
                
                # Handle 'item' field with text (alternative format)
                elif "item" in data:
                    item = data["item"]
                    if isinstance(item, dict) and "text" in item:
                        text_parts.append(item["text"])
                            
            except json.JSONDecodeError:
                # Not JSON, might be plain text output
                # Check if line looks like a function call
                if re.match(r'^\[.*\]$', line) or re.match(r'^\w+[\.\(]', line):
                    text_parts.append(line)
        
        result = "".join(text_parts).strip()
        
        # Remove markdown code block wrappers if present
        result = self._strip_markdown_code_blocks(result)
        
        # If no structured output found, try to find function calls in raw output
        if not result:
            cleaned_output = self._strip_markdown_code_blocks(raw_output)
            
            # Try to find function call list pattern [...]
            fc_match = re.search(r'\[.*?\(.*?\).*?\]', cleaned_output, re.DOTALL)
            if fc_match:
                return fc_match.group(0)
            
            # Look for single function call
            fc_match = re.search(r'\w+[\.\w]*\([^)]*\)', cleaned_output, re.DOTALL)
            if fc_match:
                return f"[{fc_match.group(0)}]"
            
            return cleaned_output if cleaned_output else raw_output
        
        return result
    
    def _strip_markdown_code_blocks(self, text: str) -> str:
        """
        Remove markdown code block wrappers like ```python ... ``` or ```json ... ```
        """
        if not text:
            return text
        
        # Pattern to match ```language ... ``` blocks
        pattern = r'```(?:python|json|javascript|bash)?\s*\n?(.*?)\n?```'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Also handle single backtick wrapped content
        if text.startswith('`') and text.endswith('`') and not text.startswith('```'):
            return text[1:-1].strip()
        
        return text

    def decode_ast(self, result: Any, language: ReturnFormat, has_tool_call_tag: bool) -> list[dict]:
        """Decode the model response to AST format for evaluation."""
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result: Any, has_tool_call_tag: bool) -> list[str]:
        """Decode the model response to executable format."""
        return default_decode_execute_prompting(result, has_tool_call_tag)

    #### Prompting methods ####

    def _query_prompting(self, inference_data: dict) -> tuple[str, float]:
        """
        Execute the query using Codex CLI.
        
        Returns:
            tuple: (response_text, latency)
        """
        # Build the complete prompt
        messages = inference_data.get("message", [])
        
        # Extract system prompt and user message
        system_prompt = ""
        user_message = ""
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_message = content
            elif role == "assistant":
                # For multi-turn, include previous assistant responses
                user_message += f"\n\nPrevious response: {content}"
        
        # Build the final prompt
        full_prompt = self._build_codex_prompt(system_prompt, user_message)
        
        # Log the inference input
        inference_data["inference_input_log"] = {
            "prompt": full_prompt,
        }
        
        # Run codex
        return self._run_codex(full_prompt)

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the test entry before sending to Codex.
        This adds function documentation to the system prompt.
        """
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        # Add function docs to system prompt
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": []}

    def _parse_query_response_prompting(self, api_response: str) -> dict:
        """
        Parse the response from Codex CLI.
        
        Args:
            api_response: The response text from Codex
            
        Returns:
            dict with model_responses and token counts (estimated)
        """
        return {
            "model_responses": api_response,
            "model_responses_message_for_chat_history": [
                {"role": "assistant", "content": api_response}
            ],
            "input_token": 0,  # Codex CLI doesn't expose token counts
            "output_token": 0,
        }

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Add the first turn message to the chat history."""
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """Add next turn user message to the chat history."""
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """Add assistant message to the chat history."""
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """Add execution results to the chat history for multi-turn."""
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data


class CodexFileHandler(BaseHandler):
    """
    File-based Codex handler for BFCL evaluation.
    
    This handler is similar to Harbor's EvoEval adapter approach:
    - Instructs Codex to execute shell commands to write function calls to a file
    - Reads the result file to get the function call output
    - This approach is closer to how Harbor evaluates agents
    
    Comparison with CodexHandler:
    - CodexHandler: Codex outputs text → parse text
    - CodexFileHandler: Codex writes file → read file → parse content
    """

    def __init__(
        self,
        model_name: str,
        temperature: float,
        registry_name: str,
        is_fc_model: bool,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OSSMODEL
        
        # Verify codex CLI is available
        self._verify_codex_installation()
    
    def _verify_codex_installation(self) -> None:
        """Check if codex CLI is installed and accessible."""
        try:
            result = subprocess.run(
                ["codex", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                print(f"Warning: codex CLI returned non-zero exit code: {result.returncode}")
        except FileNotFoundError:
            raise RuntimeError(
                "Codex CLI is not installed. Please install it with: "
                "npm install -g @openai/codex"
            )
        except subprocess.TimeoutExpired:
            print("Warning: codex version check timed out")

    def _build_codex_file_prompt(self, system_prompt: str, user_message: str) -> str:
        """
        Build a prompt for Codex to output JSON format directly to stdout.
        
        This avoids file path issues and simplifies the evaluation process.
        """
        prompt = f"""# Task

{user_message}

## Available Functions

{system_prompt}

## Output

Analyze the request and determine the appropriate function call(s). Output ONLY a JSON array.

Format:
- If a function applies: [{{"function_name": {{"param1": "value1"}}}}]
- If no function applies: []

Example output:
[{{"get_weather": {{"city": "NYC"}}}}]

IMPORTANT: Output ONLY the JSON array, nothing else."""
        
        return prompt

    def _run_codex_with_file(self, prompt: str, work_dir: Path) -> tuple[str, float]:
        """
        Execute Codex CLI and parse JSON output from stdout.
        
        Parses JSON directly from stdout without file dependency for reliability.
        
        Args:
            prompt: The prompt to send to Codex
            work_dir: Working directory (not used in this implementation)
        
        Returns:
            tuple: (json_output, latency_seconds)
        """
        # Get the model name (strip provider prefix if present)
        model = self.model_name
        if "/" in model:
            model = model.split("/")[-1]
        
        # Construct codex command
        cmd = [
            "codex",
            "exec",
            "--dangerously-bypass-approvals-and-sandbox",
            "--skip-git-repo-check",
            "--model", model,
            "--json",
            "--",
            prompt,
        ]
        
        start_time = time.time()
        
        try:
            # Run codex
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                env={**os.environ, "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "")},
            )
            
            latency = time.time() - start_time
            
            # Parse JSON directly from stdout (no file needed)
            return result.stdout, latency
                
        except subprocess.TimeoutExpired:
            latency = time.time() - start_time
            return json.dumps({"error": "Codex execution timed out"}), latency
        except Exception as e:
            latency = time.time() - start_time
            return json.dumps({"error": str(e)}), latency

    def _parse_result_file(self, content: str) -> str:
        """
        Parse Codex stdout to extract JSON function calls and convert to BFCL format.
        
        Extracts JSON from stdout events and converts to Python function call format.
        
        Args:
            content: Codex stdout (containing JSON events)
        
        Returns:
            String representation of function calls in BFCL format (e.g., "[func1(), func2()]")
        """
        # First, try to extract JSON from Codex's stdout
        # Codex outputs JSON events line by line
        json_output = None
        
        for line in content.strip().split('\n'):
            if not line.strip():
                continue
            
            try:
                event = json.loads(line)
                # Look for agent_message with JSON content
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        # Try to extract JSON array from text
                        # Look for pattern like [{...}]
                        import re
                        match = re.search(r'\[.*?\]', text, re.DOTALL)
                        if match:
                            json_str = match.group(0)
                            try:
                                json_output = json.loads(json_str)
                                break
                            except:
                                continue
            except json.JSONDecodeError:
                continue
        
        if json_output is None:
            return "[]"
        
        # Convert from JSON format to BFCL Python format
        # JSON: [{"function_name": {"param": "value"}}]
        # BFCL: "[function_name(param='value')]"
        if isinstance(json_output, list):
            result = []
            for call in json_output:
                if isinstance(call, dict) and len(call) == 1:
                    func_name = list(call.keys())[0]
                    params = call[func_name]
                    if isinstance(params, dict):
                        param_str = ", ".join([f"{k}={repr(v)}" for k, v in params.items()])
                        result.append(f"{func_name}({param_str})")
                    else:
                        result.append(f"{func_name}()")
            
            return "[" + ", ".join(result) + "]"
        
        return "[]"

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Preprocess the test entry before sending to Codex.
        This adds function documentation to the system prompt.
        """
        functions: list = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        # Add function docs to system prompt
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0], functions, test_entry_id
        )

        return {"message": test_entry["question"], "function": functions}

    def _parse_query_response_prompting(self, api_response: str) -> dict:
        """
        Parse the response from Codex file execution.
        
        Args:
            api_response: The response string from Codex (e.g., "[func1(), func2()]")
            
        Returns:
            dict with model_responses and token counts
        """
        return {
            "model_responses": api_response,
            "model_responses_message_for_chat_history": [
                {"role": "assistant", "content": api_response}
            ],
            "input_token": 0,
            "output_token": 0,
        }

    def inference_single_turn_prompting(
        self, test_entry: dict, include_input_log: bool
    ) -> tuple[any, dict]:
        """
        Single-turn inference using file-based approach.
        """
        # Pre-process the query
        inference_data: dict = self._pre_query_processing_prompting(test_entry)
        
        # Add first turn message (this converts test_entry format to message format)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry["question"][0]
        )
        
        # Extract prompts from messages
        messages = inference_data.get("message", [])
        system_prompt = ""
        user_message = ""
        
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_message = content
        
        # Build Codex prompt with file writing instructions
        prompt = self._build_codex_file_prompt(system_prompt, user_message)
        
        # Create temporary working directory
        with tempfile.TemporaryDirectory() as temp_dir:
            work_dir = Path(temp_dir)
            
            # Run Codex
            result_content, latency = self._run_codex_with_file(prompt, work_dir)
            
            # Parse result file (returns string format like "[func()]")
            api_response = self._parse_result_file(result_content)
        
        # Parse the response using the same method as direct parsing
        model_response_data = self._parse_query_response_prompting(api_response)
        
        # Process metadata
        metadata = {}
        if include_input_log:
            metadata["inference_log"] = [
                {
                    "role": "inference_input",
                    "content": inference_data.get("inference_input_log", ""),
                }
            ]
        metadata["input_token_count"] = model_response_data["input_token"]
        metadata["output_token_count"] = model_response_data["output_token"]
        metadata["latency"] = latency
        
        return model_response_data["model_responses"], metadata

    def decode_ast(self, result, language, has_tool_call_tag):
        """Decode AST for prompting mode."""
        return default_decode_ast_prompting(result, language, has_tool_call_tag)

    def decode_execute(self, result, has_tool_call_tag):
        """Decode execute for prompting mode."""
        return default_decode_execute_prompting(result, has_tool_call_tag)

    def add_first_turn_message_prompting(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        """Add the first turn message to the chat history."""
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def _add_next_turn_user_message_prompting(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        """Add next turn user message to the chat history."""
        inference_data["message"].extend(user_message)
        return inference_data

    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        """Add assistant message to the chat history."""
        inference_data["message"].extend(
            model_response_data["model_responses_message_for_chat_history"]
        )
        return inference_data

    def _add_execution_results_prompting(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """Add execution results to the chat history for multi-turn."""
        formatted_results_message = format_execution_results_prompting(
            inference_data, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data

