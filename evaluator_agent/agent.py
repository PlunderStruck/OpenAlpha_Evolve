# evaluator_agent/agent.py
import time
import logging
import traceback
import subprocess
import tempfile
import os
import ast
import json
import asyncio
import sys
import math # Ensure math is imported for isclose, isnan, isinf
from typing import Optional, Dict, Any, Tuple, Union, List

from core.interfaces import EvaluatorAgentInterface, Program, TaskDefinition, BaseAgent
from config import settings

logger = logging.getLogger(__name__)

class EvaluatorAgent(EvaluatorAgentInterface, BaseAgent):
    """
    Agent responsible for evaluating the correctness and performance of generated programs.
    It performs syntax checks and executes code against predefined test cases in a sandboxed environment.
    """
    def __init__(self, task_definition: Optional[TaskDefinition] = None):
        """
        Initializes the EvaluatorAgent.

        Args:
            task_definition (Optional[TaskDefinition]): The definition of the task
                against which programs will be evaluated.
        """
        super().__init__()
        self.task_definition = task_definition
        # This identifier was previously for a Gemini model. Now it can be a generic
        # identifier if an LLM is ever used for evaluation tasks by this agent.
        # Currently, this agent focuses on execution-based evaluation.
        self.evaluation_model_identifier = settings.LMSTUDIO_MODEL_NAME # Or a specific eval model if configured
        self.evaluation_timeout_seconds = settings.EVALUATION_TIMEOUT_SECONDS
        
        logger.info(f"EvaluatorAgent initialized. Evaluation timeout: {self.evaluation_timeout_seconds}s.")
        # Log if a specific model identifier is set (and not just the default placeholder from LM Studio config)
        if self.evaluation_model_identifier and self.evaluation_model_identifier != "local-model":
             logger.info(f"EvaluatorAgent: Placeholder for potential LLM-based evaluation using model/identifier: {self.evaluation_model_identifier}")
        if self.task_definition:
            logger.info(f"EvaluatorAgent is configured for task: {self.task_definition.id}")

    def _check_syntax(self, code: str) -> List[str]:
        """
        Checks the Python syntax of the given code string.

        Args:
            code (str): The Python code to check.

        Returns:
            List[str]: A list of error messages if syntax errors are found, otherwise an empty list.
        """
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            error_msg = f"SyntaxError: {e.msg} at line {e.lineno}, offset {e.offset}."
            if e.text: # Add the line with the error if available
                error_msg += f"\nError in line: '{e.text.strip()}'"
            errors.append(error_msg)
            logger.warning(f"Syntax error found: {error_msg}")
        except Exception as e:
            # Catch any other potential errors during parsing, though SyntaxError is the most common.
            errors.append(f"Unexpected error during syntax check: {str(e)}")
            logger.error(f"Unexpected syntax check error: {e}", exc_info=True)
        return errors

    async def _execute_code_safely(
        self, 
        code: str, 
        task_for_examples: TaskDefinition,
        timeout_seconds: Optional[int] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """
        Executes the provided Python code in a separate subprocess against test cases.

        Args:
            code (str): The Python code string to execute.
            task_for_examples (TaskDefinition): The task definition containing I/O examples.
            timeout_seconds (Optional[int]): Execution timeout. Defaults to self.evaluation_timeout_seconds.

        Returns:
            Tuple[Optional[Dict[str, Any]], Optional[str]]: 
                A tuple. The first element is a dictionary with execution results 
                (test outputs, runtimes, etc.) or None if a critical error occurred.
                The second element is an error message string if an error occurred, else None.
        """
        timeout = timeout_seconds if timeout_seconds is not None else self.evaluation_timeout_seconds
        # Initialize a comprehensive results structure
        execution_summary = {
            "test_outputs": [], 
            "average_runtime_ms": 0.0, 
            "total_successful_tests": 0, 
            "total_failed_tests": 0,
            "harness_script_error": None # To capture errors from the harness itself
        }
        
        if not task_for_examples.input_output_examples:
            logger.warning(f"No input/output examples for task '{task_for_examples.id}'. Cannot execute for correctness.")
            execution_summary["harness_script_error"] = "No test cases provided."
            # Return the summary indicating no tests were run, not a critical failure of execution itself.
            return execution_summary, "No test cases to run for correctness evaluation."

        if not task_for_examples.function_name_to_evolve:
            logger.error(f"Task '{task_for_examples.id}' is missing 'function_name_to_evolve'. Cannot execute code.")
            return None, "Task definition is missing 'function_name_to_evolve'."

        try:
            # Use TemporaryDirectory for robust cleanup of the script and its directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_path = os.path.join(temp_dir, "temp_script.py")

                # Prepare test cases string, handling special float values for JSON and Python interpretation
                test_cases_str = json.dumps(task_for_examples.input_output_examples)
                test_cases_str = test_cases_str.replace('"Infinity"', 'float("inf")').replace('"-Infinity"', 'float("-inf")')
                test_cases_str = test_cases_str.replace('"NaN"', 'float("nan")')
                test_cases_str = test_cases_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')

                # Construct the test harness script
                test_harness_code = f"""
import json
import time
import sys
import math # For float('inf'), float('nan') and other math functions
import traceback # For detailed error reporting within the harness

# --- User's code (function to be tested) ---
{code}
# --- End of User's code ---

# Initialize results structure within the harness
harness_results_data = {{
    "test_outputs": [],
    "average_runtime_ms": 0.0,
    "total_successful_tests": 0,
    "total_failed_tests": 0,
    "harness_script_error": None
}}
total_execution_time_ms_harness = 0.0
num_tests_run_harness = 0

# Define special float constants that might be in test cases (already handled by eval in Python)
Infinity = float('inf') 
NaN = float('nan')

# Load test cases (already processed for Python compatibility)
test_cases_harness = {test_cases_str} 
function_to_test_name_harness = "{task_for_examples.function_name_to_evolve}"

# Ensure the target function is available
target_function = globals().get(function_to_test_name_harness)

if not callable(target_function):
    # Attempt to find it if it was defined inside a class (common for LLM output)
    # This is a simple heuristic and might need refinement for complex class structures.
    for name, obj in list(globals().items()): # Iterate over a copy
        if isinstance(obj, type): # Check if it's a class
            if hasattr(obj, function_to_test_name_harness):
                method = getattr(obj, function_to_test_name_harness)
                if callable(method):
                    # This makes the method available globally. If it's an instance method,
                    # it might not work as expected without an instance.
                    # For static/class methods, or simple classes, it might be okay.
                    target_function = method
                    break
    if not callable(target_function):
        harness_results_data["harness_script_error"] = f"Function '{{function_to_test_name_harness}}' not found or not callable."
        # Output JSON and exit if the function isn't found
        print(json.dumps(harness_results_data, default=str)) 
        sys.exit(0) # Exit gracefully with current data, main script will see this error

# Iterate through test cases
for i, test_case_entry in enumerate(test_cases_harness):
    input_args_harness = test_case_entry.get("input")
    # The 'output' from test_case_entry is the expected_output, used for logging here
    expected_output_harness = test_case_entry.get("output") 
    
    actual_output_harness = None
    error_info_harness = None
    runtime_ms_harness = -1.0 
    status_harness = "error" 

    try:
        # Handle functions that might require specific call signatures (e.g., callables in input)
        processed_input_args = []
        is_callable_input_case = False
        if isinstance(input_args_harness, list):
            for arg in input_args_harness:
                if isinstance(arg, str):
                    if arg.startswith("lambda "): # Handle lambda strings
                        try:
                            processed_input_args.append(eval(arg, {{ "math": math }})) # Provide math module to lambda
                            is_callable_input_case = True
                        except Exception as e_eval:
                            raise ValueError(f"Failed to eval lambda string '{{arg}}': {{e_eval}}")
                    elif arg.startswith("math."): # Handle math functions like "math.sin"
                         try:
                            func_name = arg.split('.')[1]
                            processed_input_args.append(getattr(math, func_name))
                            is_callable_input_case = True
                         except Exception as e_math_lookup:
                            raise ValueError(f"Failed to lookup math function '{{arg}}': {{e_math_lookup}}")
                    else:
                        processed_input_args.append(arg)
                else:
                    processed_input_args.append(arg)
        else: # Single argument or other non-list input structure
            # This part might need more specific handling if non-list inputs can also be callables
            processed_input_args = input_args_harness


        start_time_harness = time.perf_counter()
        # Call the target function with processed arguments
        if isinstance(processed_input_args, list):
            actual_output_harness = target_function(*processed_input_args)
        # Add other input structures if necessary (e.g., dict for kwargs, None for no-arg functions)
        elif isinstance(processed_input_args, dict) and processed_input_args:
             actual_output_harness = target_function(**processed_input_args)
        elif processed_input_args is None and target_function.__code__.co_argcount == 0:
             actual_output_harness = target_function()
        else: # Default to passing as a single argument
            actual_output_harness = target_function(processed_input_args)
            
        end_time_harness = time.perf_counter()
        runtime_ms_harness = (end_time_harness - start_time_harness) * 1000
        total_execution_time_ms_harness += runtime_ms_harness
        num_tests_run_harness += 1
        status_harness = "success"
        # Tentatively increment successful tests; _assess_correctness will make the final call
        # harness_results_data["total_successful_tests"] += 1 
    except Exception as e_test:
        end_time_harness = time.perf_counter() 
        if 'start_time_harness' in locals(): # Check if start_time was set
             runtime_ms_harness = (end_time_harness - start_time_harness) * 1000
        
        error_info_harness = {{
            "message": str(e_test), 
            "type": type(e_test).__name__,
            "traceback": traceback.format_exc() # Include full traceback for debugging
        }}
        status_harness = "error"
        harness_results_data["total_failed_tests"] += 1 # Count tests that errored during execution

    # Store detailed output for this test case
    test_output_detail_harness = {{
        "test_case_id": i,
        "input_original": test_case_entry.get("input"), # Log original input for clarity
        "expected_output": expected_output_harness, 
        "actual_output": actual_output_harness, 
        "runtime_ms": round(runtime_ms_harness, 4), 
        "status": status_harness
    }}
    if error_info_harness:
        test_output_detail_harness["error_info"] = error_info_harness
    
    harness_results_data["test_outputs"].append(test_output_detail_harness)

if num_tests_run_harness > 0:
    harness_results_data["average_runtime_ms"] = round(total_execution_time_ms_harness / num_tests_run_harness, 4)

# Custom JSON serializer for special float values and other non-standard types
def custom_json_serializer_harness(obj):
    if isinstance(obj, float):
        if math.isinf(obj): return str(obj) # 'Infinity' or '-Infinity'
        if math.isnan(obj): return 'NaN'
    if callable(obj): # Represent callables as strings
        return f"<function {{obj.__name__ if hasattr(obj, '__name__') else str(obj)}}>"
    # Fallback for other unhandled types to prevent serialization errors
    try:
        json.dumps(obj) # Check if it's directly serializable
        return obj
    except TypeError:
        return repr(obj) # Return a string representation

try:
    # Output the final results as a JSON string
    print(json.dumps(harness_results_data, default=custom_json_serializer_harness))
except Exception as dump_error:
    # Fallback if harness_results_data itself is problematic for JSON dumping
    fallback_error_harness = {{
        "harness_script_error": f"FATAL: Failed to serialize execution results: {{str(dump_error)}}. Traceback: {{traceback.format_exc()}}",
        "test_outputs": harness_results_data.get("test_outputs", []) # Try to include test_outputs if available
    }}
    print(json.dumps(fallback_error_harness, default=str)) # Use basic str for fallback
"""
                # Write the harness script to the temporary file
                with open(temp_file_path, "w", encoding="utf-8") as f:
                    f.write(test_harness_code)

                cmd = [sys.executable, temp_file_path]
                proc = None 
                logger.debug(f"Executing code evaluation script: {' '.join(cmd)} in {temp_dir}")
                
                start_run_time = time.monotonic()
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=temp_dir 
                )
                
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                duration_run = time.monotonic() - start_run_time
                logger.debug(f"Code evaluation script finished in {duration_run:.2f}s. Exit code: {proc.returncode}")

                stdout_str = stdout_bytes.decode('utf-8', errors='replace').strip()
                stderr_str = stderr_bytes.decode('utf-8', errors='replace').strip()

                if stderr_str: # Log any stderr from the script, could indicate issues
                    logger.warning(f"Stderr from evaluation script ({temp_file_path}):\n{stderr_str}")

                # Try to parse the JSON output from stdout
                if stdout_str:
                    try:
                        def json_loads_with_special_floats(s_load): # Helper to parse special floats
                            s_load = s_load.replace('"Infinity"', 'float("inf")').replace('"-Infinity"', 'float("-inf")')
                            s_load = s_load.replace('"NaN"', 'float("nan")')
                            return json.loads(s_load)
                        
                        parsed_output = json_loads_with_special_floats(stdout_str)

                        # Check if the harness script itself reported an error
                        if parsed_output.get("harness_script_error"):
                            harness_error = parsed_output["harness_script_error"]
                            logger.error(f"Error reported by evaluation harness script: {harness_error}")
                            # Return the partially filled execution_summary if available, plus the harness error
                            execution_summary.update(parsed_output) # Update with any partial data
                            return execution_summary, f"Harness Script Error: {harness_error}"
                        
                        # Basic validation of parsed_output structure
                        if not isinstance(parsed_output, dict) or "test_outputs" not in parsed_output:
                            err_msg = f"Parsed output from script is not a valid results dictionary. Raw stdout: '{stdout_str[:500]}...'"
                            logger.error(err_msg)
                            return None, err_msg # Critical parsing failure
                            
                        logger.debug(f"Successfully parsed execution output for task '{task_for_examples.id}'.")
                        return parsed_output, None # Success, return parsed results

                    except json.JSONDecodeError as e_json:
                        error_message = f"Failed to decode JSON output from script: {e_json}. Raw stdout: '{stdout_str[:1000]}...'"
                        logger.error(error_message)
                        return None, error_message # Critical parsing failure
                else: # No stdout
                    err_msg = f"Execution script produced no stdout. Exit code: {proc.returncode}. Stderr: '{stderr_str[:500]}...'"
                    logger.warning(err_msg)
                    # If exit code was non-zero, it implies an error the script couldn't report via JSON
                    if proc.returncode != 0:
                        return None, err_msg
                    else: # Exit code 0 but no stdout is strange, treat as harness error
                        execution_summary["harness_script_error"] = "Script produced no output but exited cleanly."
                        return execution_summary, execution_summary["harness_script_error"]
        
        except asyncio.TimeoutError:
            if proc and proc.returncode is None: 
                try:
                    proc.kill()
                    await proc.wait() 
                except ProcessLookupError:
                    logger.debug("Process already terminated when timeout kill was attempted.")
                except Exception as e_kill_timeout:
                    logger.error(f"Error trying to kill timed-out process: {e_kill_timeout}")
            logger.warning(f"Code execution timed out after {timeout} seconds for function '{task_for_examples.function_name_to_evolve}'.")
            return None, f"Execution timed out after {timeout} seconds."
        except Exception as e_outer:
            logger.error(f"An unexpected error occurred during sandboxed code execution setup or handling: {e_outer}", exc_info=True)
            return None, f"Unexpected execution setup/handling error: {str(e_outer)}"


    def _compare_outputs(self, actual: Any, expected: Any, rel_tol: float = 1e-9, abs_tol: float = 0.0) -> bool:
        """
        Compares actual and expected outputs, with special handling for floats,
        lists/tuples of floats, and dictionaries containing floats.

        Args:
            actual: The actual output from the executed code.
            expected: The expected output defined in the task.
            rel_tol: Relative tolerance for float comparisons (see math.isclose).
            abs_tol: Absolute tolerance for float comparisons (see math.isclose).

        Returns:
            bool: True if outputs are considered close enough, False otherwise.
        """
        # logger.debug(f"Comparing outputs. Actual: {actual} (type: {type(actual)}), Expected: {expected} (type: {type(expected)})")

        # Handle cases where one is None and the other is not
        if actual is None and expected is not None: return False
        if actual is not None and expected is None: return False
        if actual is None and expected is None: return True

        # If types are fundamentally different (and not a case of int vs float)
        if type(actual) != type(expected) and not (isinstance(actual, (int, float)) and isinstance(expected, (int, float))):
            logger.debug(f"Type mismatch: Actual type {type(actual)}, Expected type {type(expected)}")
            return False

        if isinstance(actual, float) or isinstance(expected, float):
            # Ensure both can be treated as numbers for math.isclose
            if not (isinstance(actual, (int, float)) and isinstance(expected, (int, float))):
                logger.debug(f"Type mismatch for float comparison: Actual {type(actual)}, Expected {type(expected)}")
                return False 
            # Handle NaN: NaN is not close to anything, not even itself, unless both are NaN.
            if math.isnan(float(actual)) and math.isnan(float(expected)):
                return True
            if math.isnan(float(actual)) or math.isnan(float(expected)): # one is NaN, the other is not
                logger.debug(f"Float comparison: One value is NaN. Actual: {float(actual)}, Expected: {float(expected)}")
                return False
            
            are_close = math.isclose(float(actual), float(expected), rel_tol=rel_tol, abs_tol=abs_tol)
            if not are_close:
                logger.debug(f"Float comparison: math.isclose({float(actual)}, {float(expected)}, rel_tol={rel_tol}, abs_tol={abs_tol}) is False")
            return are_close
        
        if isinstance(actual, (list, tuple)):
            if not isinstance(expected, (list, tuple)) or len(actual) != len(expected):
                logger.debug(f"List/Tuple structural mismatch. Actual len: {len(actual)}, Expected len: {len(expected)}")
                return False
            # Recursively compare elements
            for i in range(len(actual)):
                if not self._compare_outputs(actual[i], expected[i], rel_tol, abs_tol):
                    # Detailed logging already happens in the recursive call
                    return False
            return True

        if isinstance(actual, dict):
            if not isinstance(expected, dict) or sorted(actual.keys()) != sorted(expected.keys()):
                # Compare sorted keys for robustness against order differences in key listings
                logger.debug(f"Dictionary key mismatch or type mismatch. Actual keys (sorted): {sorted(actual.keys())}, Expected keys (sorted): {sorted(expected.keys())}")
                return False
            # Recursively compare values
            for key in actual: # Iterate through actual's keys (which are same as expected's due to check above)
                if not self._compare_outputs(actual[key], expected[key], rel_tol, abs_tol):
                    # Detailed logging already happens in the recursive call
                    return False
            return True
        
        # For other types (int, str, bool, etc.)
        are_equal = actual == expected
        if not are_equal:
            logger.debug(f"Direct equality comparison failed. Actual: '{actual}' (type {type(actual)}), Expected: '{expected}' (type {type(expected)})")
        return are_equal


    def _assess_correctness(self, execution_results: Dict[str, Any], task_definition: TaskDefinition) -> Tuple[float, int, int, List[str]]:
        """
        Assesses the correctness of the program based on execution results from the harness.
        Uses the _compare_outputs method for robust comparisons.

        Args:
            execution_results (Dict[str, Any]): The dictionary of results from _execute_code_safely.
            task_definition (TaskDefinition): The task definition with expected outputs.

        Returns:
            Tuple[float, int, int, List[str]]: Correctness score (0.0-1.0), passed tests, 
                                               total tests defined, list of error messages from failed/errored tests.
        """
        passed_tests_count = 0
        test_detail_errors = [] # Stores messages for failed or errored tests
        
        if not task_definition.input_output_examples:
            return 1.0, 0, 0, [] # No tests to run, considered "correct" in a vacuum.
        
        total_defined_tests = len(task_definition.input_output_examples)

        if not execution_results or "test_outputs" not in execution_results or not isinstance(execution_results["test_outputs"], list):
            logger.warning("Correctness assessment: Execution results are missing 'test_outputs' or it's not a list.")
            test_detail_errors.append("Internal Error: Test output data structure was invalid for assessment.")
            return 0.0, 0, total_defined_tests, test_detail_errors

        actual_test_outputs_from_harness = execution_results["test_outputs"]

        # Verify we have one output entry for each defined test case
        if len(actual_test_outputs_from_harness) != total_defined_tests:
            mismatch_msg = (f"Correctness assessment: Mismatch in number of test outputs from harness ({len(actual_test_outputs_from_harness)}) "
                            f"and defined test cases ({total_defined_tests}). Some tests might not have run or reported.")
            logger.warning(mismatch_msg)
            test_detail_errors.append(mismatch_msg)
            # Continue to assess the results we do have.
        
        for i, expected_example_case in enumerate(task_definition.input_output_examples):
            # Find the corresponding actual output from the harness results
            actual_output_detail_case = next((res for res in actual_test_outputs_from_harness if res.get("test_case_id") == i), None)
            expected_value_for_case = expected_example_case["output"]
            input_value_for_case = expected_example_case["input"]


            if actual_output_detail_case:
                if actual_output_detail_case.get("status") == "success":
                    actual_value_from_harness = actual_output_detail_case.get("actual_output")
                    # Use the enhanced _compare_outputs method with appropriate tolerances
                    # Tolerances can be made configurable per task if needed.
                    if self._compare_outputs(actual_value_from_harness, expected_value_for_case, rel_tol=1e-5, abs_tol=1e-8):
                        passed_tests_count += 1
                    else:
                        error_detail_msg = (f"Test Case {i} FAILED: Input={input_value_for_case}, "
                                        f"Expected='{expected_value_for_case}', Got='{actual_value_from_harness}'.")
                        test_detail_errors.append(error_detail_msg)
                else: # Test case resulted in an error during its execution within the harness
                    error_info_from_harness = actual_output_detail_case.get("error_info", {})
                    error_message_from_harness_test = (f"Type: {error_info_from_harness.get('type', 'UnknownType')}, "
                                                       f"Msg: {error_info_from_harness.get('message', 'Unknown error message')}")
                    error_detail_msg = (f"Test Case {i} ERRORED: Input={input_value_for_case}. "
                                    f"Error during execution: {error_message_from_harness_test}")
                    test_detail_errors.append(error_detail_msg)
            else: # No result found for this test case ID in harness output
                error_detail_msg = f"Test Case {i}: No output detail found in execution results from harness. Input={input_value_for_case}."
                test_detail_errors.append(error_detail_msg)
        
        correctness_score = (passed_tests_count / total_defined_tests) if total_defined_tests > 0 else 1.0
        # Update total successful/failed tests in the main execution_results based on comparison
        if execution_results: # Ensure it exists
            execution_results["total_successful_tests"] = passed_tests_count
            execution_results["total_failed_tests"] = total_defined_tests - passed_tests_count
            
        return correctness_score, passed_tests_count, total_defined_tests, test_detail_errors


    async def evaluate_program(self, program: Program, task: TaskDefinition) -> Program:
        """
        Evaluates a given program for syntax, execution correctness, and performance.
        Updates the program object with evaluation results.
        """
        logger.info(f"Starting evaluation for program: {program.id} (Task: {task.id})")
        program.status = "evaluating"
        program.errors = [] # Reset errors for this evaluation run
        program.fitness_scores = { # Initialize fitness scores
            "correctness": 0.0, 
            "runtime_ms": float('inf'), 
            "passed_tests": 0.0, 
            "total_tests": float(len(task.input_output_examples) if task.input_output_examples else 0) # Initialize total_tests
        }

        # 1. Syntax Check
        syntax_errors = self._check_syntax(program.code)
        if syntax_errors:
            program.errors.extend(syntax_errors)
            program.status = "failed_evaluation" 
            logger.warning(f"Syntax errors in program {program.id}. Evaluation halted.")
            return program 

        logger.debug(f"Syntax check passed for program {program.id}.")

        # 2. Execution-based Evaluation (if I/O examples are provided)
        if task.input_output_examples:
            logger.debug(f"Executing program {program.id} against {len(task.input_output_examples)} test cases.")
            
            execution_results, execution_error_msg = await self._execute_code_safely(program.code, task_for_examples=task)
            
            if execution_error_msg: 
                logger.warning(f"Critical execution error for program {program.id}: {execution_error_msg}")
                program.errors.append(f"Execution Error: {execution_error_msg}")
                program.status = "failed_evaluation"
                # If execution_results has partial data (e.g. harness_script_error), log it
                if execution_results and execution_results.get("harness_script_error"):
                    program.errors.append(f"Harness Detail: {execution_results['harness_script_error']}")
                return program 

            if execution_results: 
                correctness, passed, total, test_detail_errors = self._assess_correctness(execution_results, task)
                program.fitness_scores["correctness"] = correctness
                program.fitness_scores["passed_tests"] = float(passed)
                # total_tests is already set, but ensure it matches if it was derived differently
                program.fitness_scores["total_tests"] = float(total) 
                program.errors.extend(test_detail_errors)

                avg_runtime = execution_results.get("average_runtime_ms", float('inf'))
                if isinstance(avg_runtime, (int, float)) and avg_runtime >= 0:
                    program.fitness_scores["runtime_ms"] = avg_runtime
                
                logger.info(f"Program {program.id} - Correctness: {correctness:.2f} ({passed}/{total} tests). Avg Runtime: {program.fitness_scores['runtime_ms']:.2f} ms.")
            else: 
                logger.error(f"Internal inconsistency: No execution error message but also no execution results for program {program.id}.")
                program.errors.append("Internal Error: Missing execution results despite no direct execution error message.")
                program.status = "failed_evaluation"
                return program
        else: 
            logger.info(f"No input/output examples for task {task.id}. Skipping execution-based correctness for program {program.id}.")
            program.fitness_scores["correctness"] = 1.0 # If only syntax check passes and no tests, consider it "correct" in that limited sense.
            program.fitness_scores["total_tests"] = 0.0
            program.errors.append("No I/O examples provided; execution correctness not evaluated.")
        
        # Finalize status based on evaluation
        # A program is considered "evaluated" if it passed syntax and execution (even if not all tests passed, it ran).
        # "failed_evaluation" means it couldn't even complete the evaluation process (syntax error, critical execution error).
        if program.status != "failed_evaluation": # If it didn't fail early due to syntax/critical exec error
            program.status = "evaluated" # It completed the evaluation process. Fitness scores reflect performance.
        
        # Add a summary error if many tests failed but no specific errors were added (e.g. if test_detail_errors was empty)
        if program.status == "evaluated" and program.fitness_scores["correctness"] < 1.0 and not program.errors:
            failed_test_count = program.fitness_scores["total_tests"] - program.fitness_scores["passed_tests"]
            if failed_test_count > 0:
                program.errors.append(f"Completed evaluation: {int(failed_test_count)}/{int(program.fitness_scores['total_tests'])} test cases failed.")

        logger.info(f"Evaluation complete for program {program.id}. Final Status: {program.status}, Fitness: {program.fitness_scores}")
        return program

    async def execute(self, program: Program, task: TaskDefinition) -> Program:
        """Main entry point for the EvaluatorAgent's execution logic."""
        return await self.evaluate_program(program, task)


# --- Example Usage (for standalone testing of EvaluatorAgent) ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG, # DEBUG for detailed output during testing
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # --- Test Case 1: Simple Sum Function ---
    test_task_sum = TaskDefinition(
        id="test_sum_001",
        description="Sum two numbers.",
        function_name_to_evolve="add_numbers",
        input_output_examples=[
            {"input": [1, 2], "output": 3},
            {"input": [-1, 1], "output": 0},
            {"input": [0, 0], "output": 0},
            {"input": [1.5, 2.5], "output": 4.0},
            {"input": [1, 0.5], "output": 1.5}
        ]
    )
    test_program_sum_correct = Program(id="sum_correct_01", code="def add_numbers(a, b):\n  return a + b")
    test_program_sum_wrong = Program(id="sum_wrong_01", code="def add_numbers(a, b):\n  return a - b") # Incorrect logic
    test_program_sum_syntax_error = Program(id="sum_syntax_err_01", code="def add_numbers(a, b):\n  return a + b +")

    # --- Test Case 2: Projectile Motion (Floats and math) ---
    test_task_projectile = TaskDefinition(
        id="test_projectile_001",
        description="Calculate projectile motion.",
        function_name_to_evolve="projectile_calc",
        input_output_examples=[
            {"input": [10, 45], "output": (10.19367991845056, 2.54841997961264)}, # (Range, Max Height)
            {"input": [20, 30], "output": (35.31010771957705, 5.09683995922528)}
        ],
        allowed_imports=["math"]
    )
    # Correct projectile motion code (example)
    projectile_code_correct = """
import math
def projectile_calc(v0, theta_degrees):
    g = 9.81
    theta_rad = math.radians(theta_degrees)
    if v0 < 0: return None # Handle invalid input
    
    R = (v0**2 * math.sin(2 * theta_rad)) / g
    H = (v0**2 * (math.sin(theta_rad))**2) / (2 * g)
    # Return with more precision, comparison will handle tolerance
    return (round(R, 8), round(H, 8)) 
"""
    test_program_projectile_correct = Program(id="proj_correct_01", code=projectile_code_correct)

    # --- Test Case 3: Lambda function as input ---
    test_task_numerical_integration = TaskDefinition(
        id="test_numerical_integration_001",
        description="Integrate f(x) from a to b using n steps (Trapezoidal).",
        function_name_to_evolve="trapezoidal_rule_test",
        input_output_examples=[
            {"input": ["lambda x: x**2", 0, 1, 10], "output": 0.335}, # Approx, exact is 1/3
            {"input": ["math.sin", 0, "math.pi", 100], "output": 1.9998355039550903} # Approx, exact is 2
        ],
        allowed_imports=["math"]
    )
    # Example code for trapezoidal rule
    trapezoidal_code = """
import math
def trapezoidal_rule_test(f_callable, a, b, n):
    if n <= 0: return None
    h = (b - a) / n
    integral_sum = f_callable(a) + f_callable(b)
    for i in range(1, n):
        integral_sum += 2 * f_callable(a + i * h)
    return (h / 2) * integral_sum
"""
    test_program_integration_correct = Program(id="integrate_correct_01", code=trapezoidal_code)


    async def run_tests():
        evaluator = EvaluatorAgent() # Task can be passed per evaluation

        logger.info("\n--- Testing Sum Function (Correct) ---")
        result_sum_correct = await evaluator.evaluate_program(test_program_sum_correct, test_task_sum)
        logger.info(f"Result for sum_correct_01: Status='{result_sum_correct.status}', Fitness={result_sum_correct.fitness_scores}, Errors={result_sum_correct.errors}")
        assert result_sum_correct.fitness_scores.get("correctness") == 1.0

        logger.info("\n--- Testing Sum Function (Wrong Logic) ---")
        result_sum_wrong = await evaluator.evaluate_program(test_program_sum_wrong, test_task_sum)
        logger.info(f"Result for sum_wrong_01: Status='{result_sum_wrong.status}', Fitness={result_sum_wrong.fitness_scores}, Errors={result_sum_wrong.errors}")
        assert result_sum_wrong.fitness_scores.get("correctness") < 1.0
        assert "FAILED" in " ".join(result_sum_wrong.errors)


        logger.info("\n--- Testing Sum Function (Syntax Error) ---")
        result_sum_syntax = await evaluator.evaluate_program(test_program_sum_syntax_error, test_task_sum)
        logger.info(f"Result for sum_syntax_err_01: Status='{result_sum_syntax.status}', Fitness={result_sum_syntax.fitness_scores}, Errors={result_sum_syntax.errors}")
        assert result_sum_syntax.status == "failed_evaluation"
        assert "SyntaxError" in " ".join(result_sum_syntax.errors)

        logger.info("\n--- Testing Projectile Motion (Correct) ---")
        # Need to pass the task_definition to the evaluator for this specific program
        evaluator_for_projectile = EvaluatorAgent(task_definition=test_task_projectile)
        result_proj_correct = await evaluator_for_projectile.evaluate_program(test_program_projectile_correct, test_task_projectile)
        logger.info(f"Result for proj_correct_01: Status='{result_proj_correct.status}', Fitness={result_proj_correct.fitness_scores}, Errors={result_proj_correct.errors}")
        # Check if correctness is close to 1.0 (allowing for small float tolerance in _compare_outputs)
        assert math.isclose(result_proj_correct.fitness_scores.get("correctness", 0.0), 1.0)


        logger.info("\n--- Testing Numerical Integration (Correct with Lambda/Math Func Input) ---")
        evaluator_for_integration = EvaluatorAgent(task_definition=test_task_numerical_integration)
        result_integration_correct = await evaluator_for_integration.evaluate_program(test_program_integration_correct, test_task_numerical_integration)
        logger.info(f"Result for integrate_correct_01: Status='{result_integration_correct.status}', Fitness={result_integration_correct.fitness_scores}, Errors={result_integration_correct.errors}")
        assert math.isclose(result_integration_correct.fitness_scores.get("correctness", 0.0), 1.0)


    asyncio.run(run_tests())
