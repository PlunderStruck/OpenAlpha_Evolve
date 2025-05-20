"""
Gradio web interface for OpenAlpha_Evolve.
"""
import gradio as gr
import asyncio
import json
import os
import sys
import time # Keep time for any potential uses, though not directly in this snippet
import logging
from datetime import datetime # Keep datetime, might be useful for logging/timestamps
from dotenv import load_dotenv

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables (e.g., for LMSTUDIO_API_BASE_URL)
load_dotenv()

from core.interfaces import TaskDefinition, Program
from task_manager.agent import TaskManagerAgent
from config import settings # This will now import settings with LM Studio config

# --- Logging Setup (remains the same, useful for Gradio app debugging) ---
class StringIOHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_capture = []
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_capture.append(msg)
        except Exception:
            self.handleError(record)
    
    def get_logs(self):
        return "\n".join(self.log_capture)
    
    def clear(self):
        self.log_capture = []

string_handler = StringIOHandler()
string_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

root_logger = logging.getLogger()
# Ensure root logger has a level set if you want string_handler to capture from all loggers
if not root_logger.hasHandlers(): # Avoid adding handlers multiple times if app reloads
    root_logger.setLevel(logging.INFO) # Or settings.LOG_LEVEL
    root_logger.addHandler(string_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)

logger = logging.getLogger(__name__) # Logger for this specific file
# Set level for app.py logger if needed, e.g., logger.setLevel(logging.DEBUG)

# Configure levels for other project loggers if desired for UI output
# for module_logger_name in ['task_manager.agent', 'code_generator.agent', 'evaluator_agent.agent', 
#                            'database_agent.agent', 'selection_controller.agent', 'prompt_designer.agent']:
#     logging.getLogger(module_logger_name).setLevel(logging.DEBUG) # Or settings.LOG_LEVEL


# --- API Key/Configuration Warning for UI ---
# Updated warning for LM Studio
LMSTUDIO_CONFIG_WARNING = ""
if settings.LMSTUDIO_API_BASE_URL == "http://localhost:1234/v1" and not os.getenv("LMSTUDIO_API_BASE_URL"):
    LMSTUDIO_CONFIG_WARNING = (
        "⚠️ LM Studio API URL is using the default ('http://localhost:1234/v1'). "
        "Ensure your LM Studio server is running there, or set LMSTUDIO_API_BASE_URL in your .env file."
    )
elif not settings.LMSTUDIO_API_BASE_URL: # Should not happen if default is set, but as a safeguard
     LMSTUDIO_CONFIG_WARNING = "⚠️ LM Studio API URL (LMSTUDIO_API_BASE_URL) is not set in .env or defaults!"


# --- Global variable for results (consider a more robust way if scaling) ---
current_results: list[Program] = []

async def run_evolution(
    task_id: str, 
    description: str, 
    function_name: str, 
    examples_json: str, 
    allowed_imports_text: str,
    population_size: int, 
    generations: int,
    progress=gr.Progress(track_tqdm=True) # Added default Progress tracker
):
    """Run the evolutionary process with the given parameters via Gradio UI."""
    string_handler.clear() # Clear previous logs for this run
    logger.info(f"Starting evolution run from Gradio UI for task: {task_id}")
    
    try:
        # Validate and parse JSON examples
        try:
            examples = json.loads(examples_json)
            if not isinstance(examples, list):
                return "Error: Input/Output Examples must be a JSON list of objects."
            for i, example in enumerate(examples):
                if not isinstance(example, dict) or "input" not in example or "output" not in example:
                    return f"Error in example {i+1}: Each example must be an object with 'input' and 'output' keys."
        except json.JSONDecodeError as e:
            return f"Error: Examples JSON is invalid. Details: {e}"
        
        # Parse allowed imports
        allowed_imports = [imp.strip() for imp in allowed_imports_text.split(",") if imp.strip()]
        
        # Override global settings with UI inputs for this run
        # Note: This changes them globally for the duration of this run.
        # If multiple users or parallel runs were possible, this would need a different approach.
        original_pop_size = settings.POPULATION_SIZE
        original_gens = settings.GENERATIONS
        settings.POPULATION_SIZE = int(population_size)
        settings.GENERATIONS = int(generations)
        
        task = TaskDefinition(
            id=task_id,
            description=description,
            function_name_to_evolve=function_name,
            input_output_examples=examples,
            allowed_imports=allowed_imports
            # evaluation_criteria and initial_code_prompt can be added as UI fields if needed
        )
        
        task_manager = TaskManagerAgent(task_definition=task)
        
        # Simple progress updates based on generations for Gradio
        # A more detailed callback from TaskManagerAgent would be better for granular progress.
        progress(0, desc="Initializing...")
        
        best_programs = await task_manager.execute() # This is the main call
        
        progress(1.0, desc="Evolution completed!") # Mark as complete
        
        # Restore original settings
        settings.POPULATION_SIZE = original_pop_size
        settings.GENERATIONS = original_gens

        global current_results # Store results for potential display
        current_results = best_programs if best_programs else []
            
        if best_programs:
            result_text = f"✅ Evolution completed successfully! Found {len(best_programs)} solution(s).\n\n"
            for i, program in enumerate(best_programs):
                result_text += f"### Solution {i+1}\n"
                result_text += f"- ID: {program.id}\n"
                result_text += f"- Fitness: {program.fitness_scores}\n" # Make sure fitness_scores is a dict
                result_text += f"- Generation: {program.generation}\n\n"
                result_text += "```python\n" + program.code + "\n```\n\n"
            return result_text, string_handler.get_logs() # Return results and logs
        else:
            return "❌ Evolution completed, but no suitable solutions were found.", string_handler.get_logs()
    
    except Exception as e:
        import traceback
        logger.error(f"Error during Gradio evolution run: {e}\n{traceback.format_exc()}")
        # Restore original settings in case of error
        if 'original_pop_size' in locals(): settings.POPULATION_SIZE = original_pop_size
        if 'original_gens' in locals(): settings.GENERATIONS = original_gens
        return f"Error during evolution: {str(e)}\n\n{traceback.format_exc()}", string_handler.get_logs()


# --- Default Task Example (Fibonacci) ---
FIB_EXAMPLES_JSON = '''[
    {"input": [0], "output": 0},
    {"input": [1], "output": 1},
    {"input": [5], "output": 5},
    {"input": [10], "output": 55}
]'''

def set_fib_example():
    """Populates UI fields with a Fibonacci example task."""
    return (
        "fibonacci_task_001",
        "Write a Python function named `fibonacci` that computes the nth Fibonacci number (0-indexed), where fib(0)=0 and fib(1)=1. The function should take a single integer as input.",
        "fibonacci",
        FIB_EXAMPLES_JSON,
        "" # No specific imports needed for basic Fibonacci
    )

# --- Gradio UI Definition ---
with gr.Blocks(title="OpenAlpha_Evolve with LM Studio") as demo:
    gr.Markdown("# 🧬 OpenAlpha_Evolve: Autonomous Algorithm Evolution (LM Studio Version)")
    gr.Markdown("""
    Define your algorithmic task, and let the system evolve solutions using a locally hosted LLM via LM Studio.
    * **Custom Tasks:** Provide a description, function name, I/O examples, and allowed imports.
    * **LM Studio Backend:** Ensure your LM Studio server is running and configured in `.env`.
    * **Evolutionary Parameters:** Adjust population size and generations. For complex tasks, larger values might be needed.
    """)
    
    if LMSTUDIO_CONFIG_WARNING:
        gr.Markdown(f"<h3 style='color:orange;'>{LMSTUDIO_CONFIG_WARNING}</h3>")
    
    with gr.Row():
        with gr.Column(scale=2): # Input column
            gr.Markdown("## ✏️ Task Definition")
            
            task_id_input = gr.Textbox(label="Task ID", placeholder="e.g., fibonacci_task_001", value="fibonacci_task_001")
            description_input = gr.Textbox(label="Task Description", placeholder="Describe the problem clearly...", value="Write a Python function named `fibonacci`...", lines=5)
            function_name_input = gr.Textbox(label="Function Name to Evolve", placeholder="e.g., fibonacci", value="fibonacci")
            examples_json_input = gr.Code(label="Input/Output Examples (JSON format)", language="json", value=FIB_EXAMPLES_JSON, lines=10)
            allowed_imports_input = gr.Textbox(label="Allowed Imports (comma-separated, e.g., math, heapq)", placeholder="e.g., math", value="")
            
            with gr.Row():
                population_size_slider = gr.Slider(label="Population Size", minimum=2, maximum=50, value=settings.POPULATION_SIZE, step=1)
                generations_slider = gr.Slider(label="Generations", minimum=1, maximum=100, value=settings.GENERATIONS, step=1)
            
            with gr.Row():
                example_btn = gr.Button("📘 Load Fibonacci Example")
                run_btn = gr.Button("🚀 Run Evolution", variant="primary")
        
        with gr.Column(scale=3): # Output column
            gr.Markdown("## 📊 Results & Logs")
            with gr.Tabs():
                with gr.TabItem("📜 Evolution Results"):
                    results_output = gr.Markdown("Evolution results will appear here...")
                with gr.TabItem("📋 Logs"):
                    logs_output = gr.Textbox(label="Live Logs", lines=20, autoscroll=True, interactive=False)
                    
    # --- Event Handlers ---
    example_btn.click(
        set_fib_example,
        outputs=[task_id_input, description_input, function_name_input, examples_json_input, allowed_imports_input]
    )
    
    run_btn.click(
        run_evolution,
        inputs=[
            task_id_input, 
            description_input, 
            function_name_input, 
            examples_json_input,
            allowed_imports_input,
            population_size_slider, 
            generations_slider
        ],
        outputs=[results_output, logs_output] # Updated to two outputs
    )

if __name__ == "__main__":
    # Launch the Gradio app
    # share=True can be used to create a public link (use with caution)
    demo.launch(server_name="0.0.0.0") # Makes it accessible on your local network
