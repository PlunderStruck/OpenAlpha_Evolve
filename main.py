"""
Main entry point for the OpenAlpha_Evolve application.
Orchestrates the different agents and manages the evolutionary loop.
"""
import asyncio
import logging
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_manager.agent import TaskManagerAgent
from core.interfaces import TaskDefinition
from config import settings # settings will be imported with LM Studio configs

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(settings.LOG_FILE, mode="a")
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting OpenAlpha_Evolve autonomous algorithmic evolution")
    logger.info(f"Configuration: Population Size={settings.POPULATION_SIZE}, Generations={settings.GENERATIONS}")
    # Updated logging for LLM configuration
    logger.info(f"LLM Configuration: Using LM Studio at endpoint '{settings.LMSTUDIO_API_BASE_URL}' with model placeholder '{settings.LMSTUDIO_MODEL_NAME}'.")
    # Comment out or remove Gemini-specific logging:
    # logger.info(f"LLM Models: Pro={settings.GEMINI_PRO_MODEL_NAME}, Flash={settings.GEMINI_FLASH_MODEL_NAME}, Eval={settings.GEMINI_EVALUATION_MODEL}")


    # --- Task Definition (Example: Shortest Path Problem) ---
    # This task definition remains the same as it's problem-specific.
    task = TaskDefinition(
        id="generic_shortest_path_problem",
        description=(
            "Given a weighted, directed graph and a starting node, find the shortest distance "
            "from the starting node to all other nodes in the graph. "
            "The graph is represented as a dictionary where keys are node identifiers (e.g., strings or integers), "
            "and values are dictionaries representing outgoing edges. In these inner dictionaries, "
            "keys are neighbor node identifiers and values are the weights (costs) of the edges to those neighbors. "
            "If a node is unreachable from the start node, its distance should be considered infinity. "
            "The function should return a dictionary where keys are node identifiers and values are the "
            "calculated shortest distances from the start node. The start node's distance to itself is 0."
        ),
        function_name_to_evolve="solve_shortest_paths",
        input_output_examples=[
            {
                "input": [{"A": {"B": 1, "C": 4}, "B": {"C": 2, "D": 5}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": 3, "D": 4}
            },
            {
                "input": [{"A": {"B": 1}, "B": {"A": 2, "C": 5}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": 6, "D": 7}
            },
            {
                "input": [{"A": {"B": 1}, "B": {}, "C": {"D": 1}, "D": {}}, "A"],
                "output": {"A": 0, "B": 1, "C": float('inf'), "D": float('inf')}
            },
            {
                "input": [{"A": {}, "B": {"C":1}}, "A"],
                "output": {"A": 0, "B": float('inf'), "C": float('inf')}
            },
            {
                "input": [{}, "A"], # Graph with only the start node, or start node not in graph
                "output": {"A": 0} # Distance to itself is 0, others effectively infinity if not mentioned
            },
             {
                "input": [{"X": {"Y":1}}, "Z"], # Start node Z not in graph keys
                "output": {"Z": 0, "X": float('inf'), "Y": float('inf')} # If Z is considered, others are inf.
            },
            { # A more complex graph
                "input": [
                    {"s": {"u": 10, "x": 5}, "u": {"v": 1, "x": 2}, "v": {"y": 4}, "x": {"u": 3, "v": 9, "y": 2}, "y": {"s": 7, "v": 6}},
                    "s"
                ],
                "output": {"s": 0, "u": 7, "v": 8, "x": 5, "y": 7}
            }
        ],
        allowed_imports=["heapq"], # e.g., for Dijkstra's algorithm
    )

    # Initialize the TaskManagerAgent with the defined task
    task_manager = TaskManagerAgent(
        task_definition=task
    )

    # Execute the evolutionary process
    best_programs = await task_manager.execute()

    # Log the results
    if best_programs:
        logger.info(f"Evolutionary process completed. Best program(s) found: {len(best_programs)}")
        for i, program in enumerate(best_programs):
            logger.info(f"Final Best Program {i+1} ID: {program.id}")
            logger.info(f"Final Best Program {i+1} Fitness: {program.fitness_scores}")
            logger.info(f"Final Best Program {i+1} Code:\n{program.code}")
    else:
        logger.info("Evolutionary process completed, but no suitable programs were found.")

    logger.info("OpenAlpha_Evolve run finished.")

if __name__ == "__main__":
    asyncio.run(main())
