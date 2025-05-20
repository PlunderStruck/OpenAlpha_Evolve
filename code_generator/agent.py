# code_generator/agent.py
import httpx # For making asynchronous HTTP requests
import json # For handling JSON data
from typing import Optional, Dict, Any # For type hinting
import logging # For logging messages
import asyncio # For asynchronous operations
import time # For time-related functions (though not directly used in this version's core logic)
import re # For regular expressions (used in diff application)

# Import interfaces and settings from the project's core and config modules
from core.interfaces import CodeGeneratorInterface, BaseAgent, Program
from config import settings

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class CodeGeneratorAgent(CodeGeneratorInterface):
    """
    An agent responsible for generating code using a Large Language Model (LLM),
    specifically adapted to interact with an LM Studio server.
    It can generate full code snippets or diffs to modify existing code.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the CodeGeneratorAgent.

        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary (not actively used in this version).
        """
        super().__init__(config)
        # Configure for LM Studio API
        self.api_base_url = settings.LMSTUDIO_API_BASE_URL
        self.model_name = settings.LMSTUDIO_MODEL_NAME # This might be a placeholder if the model is pre-selected in LM Studio
        # self.api_key = settings.LMSTUDIO_API_KEY # Uncomment if your LM Studio instance requires an API key

        # Standard LLM parameters for generation
        self.temperature = 1.0  # Controls randomness: higher values (e.g., 1.0) make output more random, lower values (e.g., 0.2) make it more deterministic.
        self.top_p = 0.9        # Nucleus sampling: considers the smallest set of tokens whose cumulative probability exceeds top_p.
        self.max_tokens = 2048  # Maximum number of tokens to generate in the response.

        logger.info(f"CodeGeneratorAgent initialized for LM Studio at: {self.api_base_url}")

    async def generate_code(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code") -> str:
        """
        Generates code or a diff based on the provided prompt using the LM Studio API.

        Args:
            prompt (str): The prompt to send to the LLM.
            model_name (Optional[str]): The specific model to use (overrides the default if provided).
            temperature (Optional[float]): The temperature for generation (overrides the default if provided).
            output_format (str): The desired output format ("code" for full code, "diff" for a diff).

        Returns:
            str: The generated code or diff string, or an empty string if generation fails.
        """
        effective_model_name = model_name if model_name else self.model_name
        effective_temperature = temperature if temperature is not None else self.temperature
        
        logger.info(f"Attempting to generate code using LM Studio model: {effective_model_name}, output_format: {output_format}, temperature: {effective_temperature}")

        # Augment prompt with diff instructions if 'diff' format is requested
        if output_format == "diff":
            prompt += '''

I need you to provide your changes as a sequence of diff blocks in the following format:

<<<<<<< SEARCH
# Original code block to be found and replaced (COPY EXACTLY from original)
=======
# New code block to replace the original
>>>>>>> REPLACE

IMPORTANT DIFF GUIDELINES:
1. The SEARCH block MUST be an EXACT copy of code from the original - match whitespace, indentation, and line breaks precisely.
2. Each SEARCH block should be large enough (3-5 lines minimum) to uniquely identify where the change should be made.
3. Include context around the specific line(s) you want to change.
4. Make multiple separate diff blocks if you need to change different parts of the code.
5. For each diff, the SEARCH and REPLACE blocks must be complete, valid code segments.

Example of a good diff:
<<<<<<< SEARCH
def calculate_sum(numbers):
    result = 0
    for num in numbers:
        result += num
    return result
=======
def calculate_sum(numbers):
    if not numbers:
        return 0
    result = 0
    for num in numbers:
        result += num
    return result
>>>>>>> REPLACE

Make sure your diff can be applied correctly!
'''
        logger.debug(f"Received prompt for code generation (format: {output_format}):\n--PROMPT START--\n{prompt}\n--PROMPT END--")

        headers = {
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {self.api_key}" # Uncomment if API key is needed
        }

        # Payload for the LM Studio API (OpenAI-compatible chat completions endpoint)
        payload = {
            "model": effective_model_name, # This might be overridden by the LM Studio server's loaded model
            "messages": [{"role": "user", "content": prompt}],
            "temperature": effective_temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            # "stream": False, # Ensure non-streaming response for simpler handling
        }

        retries = settings.API_MAX_RETRIES
        delay = settings.API_RETRY_DELAY_SECONDS
        
        # Use EVALUATION_TIMEOUT_SECONDS as a general timeout for the HTTP client
        # This might need to be adjusted if LLM responses are very slow.
        async with httpx.AsyncClient(timeout=settings.EVALUATION_TIMEOUT_SECONDS) as client:
            for attempt in range(retries):
                try:
                    logger.debug(f"API Call Attempt {attempt + 1} of {retries} to {self.api_base_url}/chat/completions.")
                    response = await client.post(
                        f"{self.api_base_url}/chat/completions", # Standard OpenAI-compatible endpoint
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status() # Raises an HTTPStatusError for bad responses (4xx or 5xx)

                    response_data = response.json()
                    
                    # Validate the structure of the response
                    if not response_data.get("choices") or \
                       not isinstance(response_data["choices"], list) or \
                       not response_data["choices"][0].get("message") or \
                       not isinstance(response_data["choices"][0]["message"], dict) or \
                       response_data["choices"][0]["message"].get("content") is None:
                        logger.warning(f"LM Studio API returned an unexpected or incomplete response structure: {response_data}")
                        if response_data.get("error"):
                            logger.error(f"LM Studio API Error in response: {response_data.get('error')}")
                            # Potentially raise an exception or handle more gracefully
                        return "" # Return empty if critical parts are missing

                    generated_text = response_data["choices"][0]["message"]["content"]
                    logger.debug(f"Raw response from LM Studio API:\n--RESPONSE START--\n{generated_text}\n--RESPONSE END--")
                    
                    if output_format == "code":
                        cleaned_code = self._clean_llm_output(generated_text)
                        logger.debug(f"Cleaned code:\n--CLEANED CODE START--\n{cleaned_code}\n--CLEANED CODE END--")
                        return cleaned_code
                    else: # output_format == "diff"
                        logger.debug(f"Returning raw diff text:\n--DIFF TEXT START--\n{generated_text}\n--DIFF TEXT END--")
                        return generated_text
                        
                except httpx.HTTPStatusError as e:
                    logger.warning(f"LM Studio API HTTP error on attempt {attempt + 1}/{retries}: {e.response.status_code} - {e.response.text}. Retrying in {delay}s...")
                except httpx.RequestError as e: 
                    logger.warning(f"LM Studio API request error on attempt {attempt + 1}/{retries}: {type(e).__name__} - {e}. Retrying in {delay}s...")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode JSON response from LM Studio on attempt {attempt + 1}/{retries}: {e}. Response text: {response.text if 'response' in locals() else 'N/A'}")
                except Exception as e: 
                    logger.error(f"An unexpected error occurred during code generation with LM Studio on attempt {attempt + 1}/{retries}: {e}", exc_info=True)
                    # For truly unexpected errors, we might not want to retry, or retry with caution.
                
                if attempt < retries - 1:
                    await asyncio.sleep(delay)
                    delay *= 2 # Exponential backoff
                else:
                    logger.error(f"LM Studio API call failed after {retries} retries.")
            
            logger.error(f"Code generation failed for LM Studio model {effective_model_name} after all retries.")
            return "" # Fallback if all retries fail

    def _clean_llm_output(self, raw_code: str) -> str:
        """
        Cleans the raw output from the LLM, typically by removing markdown code fences
        (e.g., ```python\ncode\n``` becomes `code`).

        Args:
            raw_code (str): The raw string output from the LLM.

        Returns:
            str: The cleaned code string.
        """
        logger.debug(f"Attempting to clean raw LLM output. Input length: {len(raw_code)}")
        code = raw_code.strip()
        
        # Regex to find ```python ... ``` or ``` ... ```
        match = re.search(r"```(?:python\n)?(.*?)```", code, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
            logger.debug("Cleaned markdown fences using regex.")
            return cleaned
        
        # Fallback for simple cases if regex fails or isn't comprehensive
        if code.startswith("```python") and code.endswith("```"):
            cleaned = code[len("```python"): -len("```")].strip()
            logger.debug("Cleaned Python markdown fences (simple string strip).")
            return cleaned
        elif code.startswith("```") and code.endswith("```"):
            cleaned = code[len("```"): -len("```")].strip()
            logger.debug("Cleaned generic markdown fences (simple string strip).")
            return cleaned
            
        logger.debug("No markdown fences found by common patterns, returning stripped raw code.")
        return code

    def _apply_diff(self, parent_code: str, diff_text: str) -> str:
        """
        Applies a diff in the custom format to the parent code.
        The diff format is:
        <<<<<<< SEARCH
        # Original code block
        =======
        # New code block
        >>>>>>> REPLACE

        Args:
            parent_code (str): The original code to apply the diff to.
            diff_text (str): The diff string from the LLM.

        Returns:
            str: The modified code after applying the diff, or the original code if no changes were applied.
        """
        logger.info("Attempting to apply diff.")
        modified_code = parent_code
        # Regex to find all diff blocks
        # Using re.DOTALL so that . matches newlines within the SEARCH and REPLACE blocks
        diff_pattern = re.compile(r"<<<<<<< SEARCH\s*?\n(.*?)\n=======\s*?\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
        
        patches_applied = 0
        last_match_end = 0
        current_diff_text = diff_text # Work on a copy to analyze remaining diff text

        while True:
            match = diff_pattern.search(current_diff_text)
            if not match:
                break

            search_block = match.group(1)
            replace_block = match.group(2)
            
            # Normalize line endings for the search_block to improve matching robustness against parent_code
            # Parent code's line endings are preserved during replacement.
            search_block_normalized_for_matching = search_block.replace('\r\n', '\n').replace('\r', '\n')

            # Attempt to find and replace the search_block in the *current* state of modified_code
            try:
                # Find the first occurrence of the normalized search block
                # To do this robustly, we might need to normalize `modified_code` for searching too,
                # or search line by line if direct string replacement is too brittle.
                # For now, let's try a direct replacement of the normalized version.
                
                # Create a version of modified_code with normalized line endings for searching
                modified_code_normalized_for_search = modified_code.replace('\r\n', '\n').replace('\r', '\n')
                
                start_index = modified_code_normalized_for_search.find(search_block_normalized_for_matching)

                if start_index != -1:
                    # Found the block. We need to map start_index and end_index back to the original `modified_code`
                    # This is complex if `modified_code` had mixed EOLs.
                    # A simpler, though less robust approach for this example, is to assume `replace` handles it.
                    # A truly robust diff apply would use a proper diffing library or more careful segment reconstruction.

                    # For this implementation, we'll replace the first occurrence in the original `modified_code`
                    # This assumes the LLM provides SEARCH blocks that are unique enough or appear in order.
                    
                    # Try replacing the search block (with its original EOLs from diff)
                    temp_code_orig_eol = modified_code.replace(search_block, replace_block, 1)
                    if temp_code_orig_eol != modified_code:
                        modified_code = temp_code_orig_eol
                        patches_applied += 1
                        logger.debug(f"Applied one diff block (using original EOL from diff). SEARCH:\n{search_block}\nREPLACE:\n{replace_block}")
                    else:
                        # Try replacing the search block (normalized EOL) if original EOL version didn't change anything
                        # This covers cases where parent_code EOLs might differ from diff's search_block EOLs
                        temp_code_norm_eol = modified_code.replace(search_block_normalized_for_matching, replace_block, 1)
                        if temp_code_norm_eol != modified_code:
                            modified_code = temp_code_norm_eol
                            patches_applied += 1
                            logger.debug(f"Applied one diff block (using normalized EOL for search). SEARCH:\n{search_block_normalized_for_matching}\nREPLACE:\n{replace_block}")
                        else:
                            logger.warning(f"Diff application: SEARCH block was found by normalized search, but direct replacement attempts failed to change code. SEARCH block:\n{search_block}")
                else:
                    logger.warning(f"Diff application: SEARCH block not found in current code state:\n{search_block_normalized_for_matching}")
            
            except Exception as e:
                logger.error(f"Error during a specific diff block application: {e}", exc_info=True)
                # Continue to the next diff block if one fails

            # Move to the text after the current match to find subsequent diffs
            current_diff_text = current_diff_text[match.end():]

        if patches_applied > 0:
             logger.info(f"Diff successfully applied, {patches_applied} changes made.")
        elif "=======" in diff_text: # Check if it looked like a diff format was intended
             logger.warning("Diff text was provided (contained '======='), but no changes were applied. Check SEARCH blocks, diff format, or LLM output.")
        else: # No '=======' implies it probably wasn't a diff or was empty
             logger.info("No valid diff content provided or diff was empty, code unchanged.")
             
        return modified_code

    async def execute(self, prompt: str, model_name: Optional[str] = None, temperature: Optional[float] = None, output_format: str = "code", parent_code_for_diff: Optional[str] = None) -> str:
        """
        Main execution method for the agent.
        If output_format is 'diff', it generates a diff and applies it to parent_code_for_diff.
        Otherwise, it generates full code.

        Args:
            prompt (str): The prompt for the LLM.
            model_name (Optional[str]): Specific model name to use.
            temperature (Optional[float]): Specific temperature for generation.
            output_format (str): "code" or "diff".
            parent_code_for_diff (Optional[str]): The original code if output_format is "diff".

        Returns:
            str: The generated (and potentially modified) code, or raw diff if application fails.
        """
        logger.debug(f"CodeGeneratorAgent.execute called. Output format: {output_format}")
        
        generated_output = await self.generate_code(
            prompt=prompt, 
            model_name=model_name, 
            temperature=temperature,
            output_format=output_format
        )

        if output_format == "diff":
            if not parent_code_for_diff:
                logger.error("Output format is 'diff' but no parent_code_for_diff provided. Returning raw diff output.")
                return generated_output 
            
            # Check if the generated output actually looks like a diff and is not empty
            if not generated_output.strip() or "=======" not in generated_output:
                 logger.info("Generated output for diff is empty or not in expected format. Returning parent code unchanged.")
                 return parent_code_for_diff # Return original code if diff is invalid

            try:
                logger.info("Applying generated diff to parent code.")
                modified_code = self._apply_diff(parent_code_for_diff, generated_output)
                return modified_code
            except Exception as e:
                # If diff application itself throws an unexpected error, log it and return the raw diff.
                logger.error(f"Error applying diff: {e}. Returning raw diff text as fallback.", exc_info=True)
                return generated_output # Return the raw diff for inspection
        else: # output_format == "code"
            return generated_output

# Main block for testing (can be run with `python -m code_generator.agent`)
if __name__ == '__main__':
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Test for the _apply_diff method
    async def test_diff_application():
        agent = CodeGeneratorAgent()
        parent = """Line 1
Line 2 to be replaced
Line 3
Another block
To be changed
End of block
Final line"""

        # Test case 1: Standard diff
        diff1 = """Some preamble text from LLM...
<<<<<<< SEARCH
Line 2 to be replaced
=======
Line 2 has been successfully replaced
>>>>>>> REPLACE

Some other text...

<<<<<<< SEARCH
Another block
To be changed
End of block
=======
This
Entire
Block
Is New
>>>>>>> REPLACE
Trailing text..."""
        expected_output1 = """Line 1
Line 2 has been successfully replaced
Line 3
This
Entire
Block
Is New
Final line"""
        
        print("\n--- Testing _apply_diff directly (Test Case 1) ---")
        result1 = agent._apply_diff(parent, diff1)
        print("Result of diff application (Test Case 1):")
        print(f"Expected:\n{expected_output1}\nGot:\n{result1}")
        assert result1.strip() == expected_output1.strip(), "Direct diff application failed for Test Case 1."
        print("_apply_diff Test Case 1 passed.")

        # Test case 2: Diff with slightly different line endings in search block
        diff2 = """<<<<<<< SEARCH
Line 2 to be replaced\r
=======
Line 2 replaced (CR EOL in search)
>>>>>>> REPLACE"""
        expected_output2 = """Line 1
Line 2 replaced (CR EOL in search)
Line 3
Another block
To be changed
End of block
Final line"""
        print("\n--- Testing _apply_diff directly (Test Case 2 - EOL Mismatch) ---")
        result2 = agent._apply_diff(parent, diff2)
        print(f"Expected:\n{expected_output2}\nGot:\n{result2}")
        assert result2.strip() == expected_output2.strip(), "Direct diff application failed for Test Case 2."
        print("_apply_diff Test Case 2 passed.")


        print("\n--- Testing execute with output_format='diff' (mocked LLM) ---")
        async def mock_generate_code_for_diff_test(prompt, model_name, temperature, output_format):
            # This mock should return the 'diff1' string for this test
            return diff1
        
        original_generate_code_method = agent.generate_code # Save original method
        agent.generate_code = mock_generate_code_for_diff_test # Mock the method
        
        result_execute_diff = await agent.execute(
            prompt="mocked_prompt_for_diff", 
            parent_code_for_diff=parent,
            output_format="diff"
        )
        agent.generate_code = original_generate_code_method # Restore original method

        print("Result of execute with diff (mocked LLM):")
        print(result_execute_diff)
        assert result_execute_diff.strip() == expected_output1.strip(), "Execute with diff (mocked) failed."
        print("Execute with diff (mocked LLM) test passed.")

    # Test for LM Studio code generation (requires a running LM Studio server)
    async def test_lm_studio_generation():
        # This agent will use LMStudio settings from config/settings.py by default
        agent = CodeGeneratorAgent() 
        
        print("\n--- Testing Code Generation with LM Studio (LIVE TEST) ---")
        # Simple prompt for full code generation
        test_prompt_full_code = "Write a very simple Python function that takes two numbers, x and y, and returns their sum. Name the function 'add_two_numbers'."
        
        try:
            # Test full code generation
            generated_full_code = await agent.execute(test_prompt_full_code, temperature=0.5, output_format="code")
            print("\n--- Generated Full Code (via LM Studio) ---")
            print(generated_full_code)
            print("-------------------------------------------")
            assert "def add_two_numbers" in generated_full_code, "LM Studio full code generation failed to produce the expected function name."
            print("LM Studio full code generation test passed (manual check of output recommended).")

            # Test diff generation and application with LM Studio
            parent_code_for_llm_diff = '''
def greet(name: str) -> str:
    """Greets the person."""
    return f"Hello, {name}!"

def calculate_stuff(data: list) -> int:
    # TODO: Implement actual calculation
    result = sum(data) # Current simple sum
    return result * 2 
'''
            test_prompt_diff_gen = f'''
Current Python code:
```python
{parent_code_for_llm_diff}
```
Task:
1. Modify the `greet` function to return "Greetings, {name}!!" instead.
2. Modify the `calculate_stuff` function to subtract 5 from the sum before multiplying by 2.
Provide the changes ONLY in the specified diff format.
'''
            # This call will first generate a diff, then apply it
            modified_code_via_diff = await agent.execute(
                prompt=test_prompt_diff_gen,
                output_format="diff",
                parent_code_for_diff=parent_code_for_llm_diff,
                temperature=0.7 # Slightly higher temp for potentially more creative diffs
            )
            print("\n--- Code After Applying Diff (from LM Studio) ---")
            print(modified_code_via_diff)
            print("-------------------------------------------------")
            assert "Greetings, {name}!!" in modified_code_via_diff, "LM Studio diff for `greet` function was not applied correctly."
            assert "sum(data) - 5" in modified_code_via_diff or "(sum(data) - 5)" in modified_code_via_diff, "LM Studio diff for `calculate_stuff` was not applied correctly."
            assert modified_code_via_diff != parent_code_for_llm_diff, "LM Studio diff generation and application resulted in no change to the code."
            print("LM Studio diff generation and application test passed (manual check of output recommended).")

        except httpx.RequestError as e:
            print(f"LIVE LM Studio Test FAILED: Could not connect to LM Studio at {agent.api_base_url}. "
                  f"Please ensure LM Studio is running with a model loaded and the server started. Error: {e}")
        except Exception as e:
            print(f"LIVE LM Studio Test FAILED with an unexpected error: {e}", exc_info=True)

    # Main function to run the tests
    async def main_tests_for_agent():
        await test_diff_application() # Test the local _apply_diff logic
        
        # Control whether to run live tests against an LM Studio instance
        run_live_lm_studio_tests = True # Set to False to skip live tests
        
        if run_live_lm_studio_tests:
            print("\nIMPORTANT: The following tests will attempt to connect to a local LM Studio server.")
            print(f"Please ensure LM Studio is running, a model is loaded, and the server is active at: {settings.LMSTUDIO_API_BASE_URL}.")
            await test_lm_studio_generation()
        else:
            print("\nSkipping LIVE LM Studio generation tests as per `run_live_lm_studio_tests` flag.")
                                                                                     
        print("\nAll selected tests for CodeGeneratorAgent completed.")

    # Run the asyncio event loop for the tests
    asyncio.run(main_tests_for_agent())
