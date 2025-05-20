# task_manager/agent.py
import logging
import asyncio
import sys
import uuid # Keep uuid import, it might be useful for other unique ID needs or future changes.
from typing import List, Dict, Any, Optional

from core.interfaces import (
    TaskManagerInterface, TaskDefinition, Program, BaseAgent,
    PromptDesignerInterface, CodeGeneratorInterface, EvaluatorAgentInterface,
    DatabaseAgentInterface, SelectionControllerInterface
)
from config import settings

# Import specific agent implementations
# These should be the classes adapted for LM Studio if you've made those changes.
from prompt_designer.agent import PromptDesignerAgent
from code_generator.agent import CodeGeneratorAgent # This should be your LM Studio version
from evaluator_agent.agent import EvaluatorAgent
from database_agent.agent import InMemoryDatabaseAgent # Or your SQLite agent if configured
from selection_controller.agent import SelectionControllerAgent

logger = logging.getLogger(__name__)

class TaskManagerAgent(TaskManagerInterface):
    """
    Manages the overall evolutionary process for a given task.
    Orchestrates other agents like PromptDesigner, CodeGenerator, Evaluator, etc.
    """
    def __init__(self, task_definition: TaskDefinition, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the TaskManagerAgent.

        Args:
            task_definition (TaskDefinition): The definition of the task to be solved.
            config (Optional[Dict[str, Any]]): Optional configuration dictionary.
        """
        super().__init__(config)
        self.task_definition = task_definition
        
        # Initialize specialized agents
        self.prompt_designer: PromptDesignerInterface = PromptDesignerAgent(task_definition=self.task_definition)
        self.code_generator: CodeGeneratorInterface = CodeGeneratorAgent() # Uses LM Studio settings
        self.evaluator: EvaluatorAgentInterface = EvaluatorAgent(task_definition=self.task_definition)
        
        # Initialize database and selection controller
        # Ensure DATABASE_TYPE in settings correctly points to the desired database agent
        if settings.DATABASE_TYPE == "in_memory":
            self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
        # Add other database types here if implemented (e.g., SQLiteDatabaseAgent)
        # elif settings.DATABASE_TYPE == "sqlite":
        #     from database_agent.sqlite_agent import SQLiteDatabaseAgent # Assuming this path
        #     self.database: DatabaseAgentInterface = SQLiteDatabaseAgent()
        else:
            logger.warning(f"Unsupported DATABASE_TYPE '{settings.DATABASE_TYPE}'. Defaulting to InMemoryDatabaseAgent.")
            self.database: DatabaseAgentInterface = InMemoryDatabaseAgent()
            
        self.selection_controller: SelectionControllerInterface = SelectionControllerAgent()

        # Evolutionary parameters from settings
        self.population_size = settings.POPULATION_SIZE
        self.num_generations = settings.GENERATIONS
        # Number of parents to select for creating the next generation's offspring
        # This can be a fixed number or a percentage of the population size.
        self.num_parents_to_select = max(1, self.population_size // 2) # Ensure at least one parent if pop_size is small

        # Callback for Gradio progress (optional)
        self.progress_callback = None


    async def initialize_population(self) -> List[Program]:
        """
        Initializes the first generation of programs.

        Generates `self.population_size` programs using the initial prompt,
        evaluates them, and saves them to the database.

        Returns:
            List[Program]: The list of initialized and evaluated programs.
        """
        logger.info(f"Initializing population for task: {self.task_definition.id} with size {self.population_size}")
        initial_population = []
        generation_tasks = []

        # Create tasks for generating initial programs
        for i in range(self.population_size):
            program_id = f"{self.task_definition.id}_gen0_prog{i}" # Initial programs have unique IDs based on index
            logger.debug(f"Preparing initial program {i+1}/{self.population_size} with id {program_id}")
            
            # Define an async task for each program generation
            async def create_initial_program(p_id: str):
                initial_prompt = self.prompt_designer.design_initial_prompt() # Pass task_definition if method expects it
                generated_code = await self.code_generator.generate_code(initial_prompt, temperature=0.8) # Higher temp for diversity
                
                program = Program(
                    id=p_id,
                    code=generated_code,
                    generation=0,
                    status="unevaluated" # Will be evaluated next
                )
                return program
            generation_tasks.append(create_initial_program(program_id))

        # Execute all generation tasks concurrently
        generated_programs_results = await asyncio.gather(*generation_tasks, return_exceptions=True)

        for result in generated_programs_results:
            if isinstance(result, Exception):
                logger.error(f"Error during initial program generation: {result}", exc_info=result)
            elif result and result.code.strip(): # Ensure code was actually generated
                initial_population.append(result)
                await self.database.save_program(result) # Save unevaluated program
            elif result:
                logger.warning(f"Initial program generation for {result.id} resulted in empty code. Skipping.")
            else: # Should not happen if exceptions are caught
                logger.error("An unknown error occurred, initial program result was None.")


        logger.info(f"Initialized {len(initial_population)} raw programs. Now evaluating...")
        # Evaluate the newly created initial population
        evaluated_initial_population = await self.evaluate_population(initial_population)
        
        logger.info(f"Finished initializing and evaluating population with {len(evaluated_initial_population)} programs.")
        return evaluated_initial_population

    async def evaluate_population(self, population: List[Program]) -> List[Program]:
        """
        Evaluates a list of programs.

        For each program not yet evaluated, it calls the EvaluatorAgent.
        Updates program status, fitness scores, and errors. Saves updated programs.

        Args:
            population (List[Program]): The list of programs to evaluate.

        Returns:
            List[Program]: The list of programs after evaluation.
        """
        if not population:
            logger.info("evaluate_population called with an empty list. Nothing to do.")
            return []
            
        logger.info(f"Evaluating population of {len(population)} programs.")
        evaluated_programs_accumulator = [] # To store results as they come
        
        # Create evaluation tasks only for unevaluated programs
        evaluation_tasks = []
        programs_to_evaluate = [prog for prog in population if prog.status != "evaluated"]
        
        if not programs_to_evaluate:
            logger.info("No programs in the provided list need evaluation.")
            # Return the original population if all were already evaluated or failed before.
            # However, it's safer to rebuild the list based on what was passed.
            return [prog for prog in population]


        for prog_to_eval in programs_to_evaluate:
            evaluation_tasks.append(self.evaluator.evaluate_program(prog_to_eval, self.task_definition))
        
        # Run evaluation tasks concurrently
        evaluation_results = await asyncio.gather(*evaluation_tasks, return_exceptions=True)
        
        # Process results
        # Rebuild the list of programs based on evaluation outcomes
        # Start with programs that were not sent for evaluation (e.g., already evaluated)
        processed_programs_map = {prog.id: prog for prog in population if prog.status == "evaluated" or prog.status == "failed_evaluation"}

        for i, result_or_exc in enumerate(evaluation_results):
            original_program_that_was_evaluated = programs_to_evaluate[i] # Get the corresponding input program

            if isinstance(result_or_exc, Exception):
                logger.error(f"Error evaluating program {original_program_that_was_evaluated.id}: {result_or_exc}", exc_info=result_or_exc)
                original_program_that_was_evaluated.status = "failed_evaluation" # Mark as failed
                original_program_that_was_evaluated.errors.append(f"Evaluation Exception: {str(result_or_exc)}")
                processed_programs_map[original_program_that_was_evaluated.id] = original_program_that_was_evaluated
            elif result_or_exc: # This is the evaluated Program object
                evaluated_program = result_or_exc
                processed_programs_map[evaluated_program.id] = evaluated_program
            else: # Should not happen if exceptions are caught
                 logger.error(f"Unknown error: Evaluation result was None for program {original_program_that_was_evaluated.id}")
                 original_program_that_was_evaluated.status = "failed_evaluation"
                 original_program_that_was_evaluated.errors.append("Unknown evaluation error: result was None")
                 processed_programs_map[original_program_that_was_evaluated.id] = original_program_that_was_evaluated

            # Save each program immediately after its evaluation result is processed
            if original_program_that_was_evaluated.id in processed_programs_map: # Ensure we save the processed one
                await self.database.save_program(processed_programs_map[original_program_that_was_evaluated.id])
            
        # Construct the final list of programs in the original order as much as possible, now updated
        final_evaluated_population = [processed_programs_map.get(p.id, p) for p in population]

        successful_evals = sum(1 for p in final_evaluated_population if p.status == "evaluated")
        logger.info(f"Finished evaluating population. {successful_evals}/{len(population)} programs successfully evaluated (others may have failed or were already processed).")
        return final_evaluated_population


    async def manage_evolutionary_cycle(self):
        """
        Manages the main evolutionary loop: selection, offspring generation, evaluation.
        This is the core orchestrator of the genetic algorithm.

        Returns:
            List[Program]: The best program(s) found after all generations.
        """
        logger.info(f"Starting evolutionary cycle for task: {self.task_definition.description[:50]}...")
        
        # Initialize and evaluate the first population
        current_population = await self.initialize_population()
        if not current_population:
            logger.error("Initialization failed to produce any programs. Aborting evolution.")
            return []

        for gen in range(1, self.num_generations + 1):
            logger.info(f"--- Generation {gen}/{self.num_generations} ---")
            if self.progress_callback: # For Gradio UI
                 await self.progress_callback(gen, self.num_generations, 0, f"Starting Generation {gen}")


            # Select parents from the current evaluated population
            # Ensure population has programs with fitness scores for selection
            eligible_for_parenting = [p for p in current_population if p.status == "evaluated" and p.fitness_scores]
            if not eligible_for_parenting:
                logger.warning(f"Generation {gen}: No eligible parents (evaluated with fitness) in current population. Ending evolution early.")
                break
            
            parents = self.selection_controller.select_parents(eligible_for_parenting, self.num_parents_to_select)
            if not parents:
                logger.warning(f"Generation {gen}: No parents selected by the selection controller. Ending evolution early.")
                break
            logger.info(f"Generation {gen}: Selected {len(parents)} parents.")
            if self.progress_callback:
                 await self.progress_callback(gen, self.num_generations, 1, "Parents Selected")

            # Generate offspring
            offspring_population_tasks = []
            offspring_counter_this_generation = 0 # Counter for unique offspring IDs this generation

            # Determine how many offspring each parent (or pair) should produce
            # This aims to create roughly self.population_size new offspring
            # Simple division, might lead to slightly more/less if not perfectly divisible.
            num_offspring_to_generate = self.population_size 
            
            # Create tasks to generate offspring
            # This loop structure assumes we want `num_offspring_to_generate` total offspring.
            # We cycle through parents to produce them.
            for i in range(num_offspring_to_generate):
                parent_for_this_offspring = parents[i % len(parents)] # Cycle through selected parents
                child_id = f"{self.task_definition.id}_gen{gen}_child{offspring_counter_this_generation}"
                
                offspring_population_tasks.append(
                    self.generate_offspring(parent_for_this_offspring, gen, child_id)
                )
                offspring_counter_this_generation += 1
            
            # Execute offspring generation tasks concurrently
            generated_offspring_results = await asyncio.gather(*offspring_population_tasks, return_exceptions=True)

            newly_generated_offspring = []
            for result_or_exc in generated_offspring_results:
                if isinstance(result_or_exc, Exception):
                    logger.error(f"Error generating an offspring: {result_or_exc}", exc_info=result_or_exc)
                elif result_or_exc and result_or_exc.code.strip(): # Ensure Program object and has code
                    newly_generated_offspring.append(result_or_exc)
                    await self.database.save_program(result_or_exc) # Save unevaluated offspring
                elif result_or_exc:
                    logger.warning(f"Offspring generation for {result_or_exc.id} resulted in empty code. Skipping.")

            logger.info(f"Generation {gen}: Generated {len(newly_generated_offspring)} raw offspring.")
            if self.progress_callback:
                 await self.progress_callback(gen, self.num_generations, 2, f"Generated {len(newly_generated_offspring)} Offspring")

            if not newly_generated_offspring:
                logger.warning(f"Generation {gen}: No offspring successfully generated. Population might stagnate.")
                # Decide if to continue or break, for now, we continue with current pop if no offspring
            
            # Evaluate the new offspring
            evaluated_offspring = await self.evaluate_population(newly_generated_offspring)
            logger.info(f"Generation {gen}: Evaluated {len(evaluated_offspring)} offspring.")
            if self.progress_callback:
                 await self.progress_callback(gen, self.num_generations, 3, "Offspring Evaluated")

            # Combine current population (parents/survivors from last gen) and new evaluated offspring
            combined_for_survival = current_population + evaluated_offspring
            
            # Select survivors for the next generation
            # Ensure only valid programs are passed to selection controller
            eligible_for_survival = [p for p in combined_for_survival if p.status == "evaluated" and p.fitness_scores]
            if not eligible_for_survival:
                logger.warning(f"Generation {gen}: No programs eligible for survival selection. Ending evolution.")
                break

            current_population = self.selection_controller.select_survivors(
                current_population, # Pass current pop (which might be just parents or all from previous gen)
                evaluated_offspring,  # Pass newly evaluated offspring
                self.population_size # Target size for the next generation
            )
            logger.info(f"Generation {gen}: New population size after survival selection: {len(current_population)}.")
            if self.progress_callback:
                 await self.progress_callback(gen, self.num_generations, 4, "Survivors Selected")


            # Log best program of this generation
            if current_population:
                # Sort by correctness (desc), then by runtime (asc - lower is better)
                best_program_this_gen_list = sorted(
                    [p for p in current_population if p.status == "evaluated" and p.fitness_scores], 
                    key=lambda p: (
                        p.fitness_scores.get("correctness", -1.0), 
                        -p.fitness_scores.get("runtime_ms", float('inf')) # Negate for sorting: higher (less negative) runtime is worse
                    ), 
                    reverse=True # True for correctness (higher is better), negation handles runtime
                )
                if best_program_this_gen_list:
                     logger.info(f"Generation {gen}: Best program: ID={best_program_this_gen_list[0].id}, Fitness={best_program_this_gen_list[0].fitness_scores}")
                else:
                    logger.warning(f"Generation {gen}: No evaluated programs in current population to determine best.")
            else:
                logger.warning(f"Generation {gen}: Current population is empty after survival selection. Ending evolution.")
                break
            
            if not current_population: # Double check if population became empty
                break


        logger.info("Evolutionary cycle completed.")
        # Retrieve and log the overall best program(s) from the database after all generations
        # The definition of "best" depends on the task and database implementation.
        # For InMemoryDatabaseAgent, it might sort all stored programs.
        final_best_programs = await self.database.get_best_programs(
            task_id=self.task_definition.id, 
            limit=3, 
            objective="correctness" # Assuming 'correctness' is a key in fitness_scores
        )
        if final_best_programs:
            logger.info(f"Overall Best Program Found: ID={final_best_programs[0].id}, Code:\n{final_best_programs[0].code}\nFitness: {final_best_programs[0].fitness_scores}")
            return final_best_programs
        else:
            logger.info("No best program found at the end of evolution from the database.")
            # Fallback: if database search fails, return the best from the very last population if available
            if 'current_population' in locals() and current_population:
                best_last_gen = sorted(
                    [p for p in current_population if p.status == "evaluated" and p.fitness_scores],
                    key=lambda p: (p.fitness_scores.get("correctness", -1.0), -p.fitness_scores.get("runtime_ms", float('inf'))),
                    reverse=True
                )
                if best_last_gen:
                    logger.info(f"Best from final population: ID={best_last_gen[0].id}, Fitness={best_last_gen[0].fitness_scores}")
                    return [best_last_gen[0]]
            return []
    
    async def generate_offspring(self, parent: Program, generation_num: int, child_id:str) -> Optional[Program]:
        """
        Generates a single offspring program from a parent program.

        This involves designing a prompt (for mutation or bug-fixing) and using
        the CodeGeneratorAgent to get new code (often as a diff).

        Args:
            parent (Program): The parent program.
            generation_num (int): The current generation number.
            child_id (str): The unique ID for the child program.

        Returns:
            Optional[Program]: The generated offspring program, or None if generation fails.
        """
        logger.debug(f"Generating offspring {child_id} from parent {parent.id} for generation {generation_num}")
        
        prompt_type = "mutation" # Default prompt type
        mutation_prompt_str = ""

        # Determine if we should try a bug-fix prompt or a mutation prompt
        # Condition: if parent has errors and correctness is very low
        # The threshold (e.g., 0.1) for correctness can be tuned.
        if parent.errors and parent.fitness_scores.get("correctness", 1.0) < 0.1: # Low correctness
            primary_error = parent.errors[0] # Use the first error message
            # Try to find more execution details if available (e.g., from evaluator)
            execution_details = None
            if len(parent.errors) > 1 and isinstance(parent.errors[1], str) and \
               ("stdout" in parent.errors[1].lower() or "stderr" in parent.errors[1].lower()):
                execution_details = parent.errors[1]
            
            mutation_prompt_str = self.prompt_designer.design_bug_fix_prompt(
                program=parent, 
                error_message=primary_error, 
                execution_output=execution_details
                # task=self.task_definition # If method expects it
            )
            logger.info(f"Attempting bug fix for parent {parent.id} using diff. Error: {primary_error}")
            prompt_type = "bug_fix"
        else:
            # Prepare feedback for general mutation
            feedback_for_mutation = {
                "errors": parent.errors, # Include errors even if not triggering bug-fix mode
                "correctness_score": parent.fitness_scores.get("correctness"),
                "runtime_ms": parent.fitness_scores.get("runtime_ms"),
                "passed_tests": parent.fitness_scores.get("passed_tests"),
                "total_tests": parent.fitness_scores.get("total_tests"),
            }
            # Remove None values from feedback to keep prompt clean
            feedback_for_mutation = {k: v for k, v in feedback_for_mutation.items() if v is not None}

            mutation_prompt_str = self.prompt_designer.design_mutation_prompt(
                program=parent, 
                evaluation_feedback=feedback_for_mutation
                # task=self.task_definition # If method expects it
            )
            logger.info(f"Attempting mutation for parent {parent.id} using diff.")
        
        # Generate code (expecting a diff) using the CodeGeneratorAgent
        # Temperature can be adjusted; slightly lower for bug-fixing, higher for mutation might be an idea.
        temp_for_generation = 0.7 if prompt_type == "bug_fix" else 0.75

        # The `execute` method of CodeGeneratorAgent handles diff application if `output_format="diff"`
        generated_code_or_diff_result = await self.code_generator.execute(
            prompt=mutation_prompt_str, 
            temperature=temp_for_generation, 
            output_format="diff", # Explicitly request a diff
            parent_code_for_diff=parent.code # Provide parent code for diff application
        )

        # Validate the result from code_generator.execute
        # If output_format was "diff", this should be the *applied* code.
        if not generated_code_or_diff_result.strip():
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) resulted in empty code/diff after application. Skipping.")
            return None
        
        if generated_code_or_diff_result == parent.code:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) using diff resulted in no change to the code. Skipping.")
            return None
        
        # Check for common LLM failure indicators in the *returned code/diff string itself*
        # (The CodeGeneratorAgent's _apply_diff might also log warnings about diff format)
        if "<<<<<<< SEARCH" in generated_code_or_diff_result and \
           "=======" in generated_code_or_diff_result and \
           ">>>>>>> REPLACE" in generated_code_or_diff_result:
            logger.warning(f"Offspring generation for parent {parent.id} ({prompt_type}) seems to have returned raw diff instead of applied code. LLM or diff application may have failed. Skipping. Content:\n{generated_code_or_diff_result[:500]}")                      
            return None # Don't create a program with raw diff markers
        
        if "# Error:" in generated_code_or_diff_result[:100] or "Error:" in generated_code_or_diff_result[:100]:
            logger.warning(f"LLM output for offspring of {parent.id} ({prompt_type}) indicates an error in generation: {generated_code_or_diff_result[:200]}. Skipping.")
            return None

        # Create the new Program object for the offspring
        offspring = Program(
            id=child_id,
            code=generated_code_or_diff_result, # This is the new, potentially modified code
            generation=generation_num,
            parent_id=parent.id,
            status="unevaluated" # Will be evaluated later
        )
        logger.info(f"Successfully generated offspring {offspring.id} from parent {parent.id} ({prompt_type}).")
        return offspring

    async def execute(self) -> Any:
        """
        Main execution method for the TaskManagerAgent.
        Starts and manages the evolutionary cycle.
        """
        return await self.manage_evolutionary_cycle()


# --- Example Usage (for standalone testing of TaskManagerAgent) ---
if __name__ == '__main__':
    # Configure logging for the test run
    logging.basicConfig(
        level=logging.INFO, # Set to DEBUG for more verbose output
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)] # Ensure logs go to console for testing
    )
    
    # Define a sample task for testing
    # This task should be solvable by the LLM you are using with LM Studio.
    sample_task_def = TaskDefinition(
        id="sum_list_task_002",
        description="Write a Python function called `solve_sum(numbers)` that takes a list of integers `numbers` and returns their sum. The function should handle empty lists correctly by returning 0.",
        function_name_to_evolve="solve_sum", # Ensure this matches the function name in the description
        input_output_examples=[
            {"input": [[1, 2, 3]], "output": 6}, # Input for *args is a list of lists/tuples
            {"input": [[]], "output": 0},
            {"input": [[-1, 0, 1]], "output": 0},
            {"input": [[10, 20, 30, 40, 50]], "output": 150}
        ],
        evaluation_criteria={"target_metric": "correctness", "goal": "maximize"}, # Example criteria
        # The initial prompt is usually handled by PromptDesignerAgent based on the description.
        # initial_code_prompt = "Please provide a Python function `solve_sum(numbers)` that sums a list of integers. Handle empty lists by returning 0."
        allowed_imports = [] # No specific imports needed for simple sum
    )
    
    # Create an instance of the TaskManagerAgent
    task_manager_instance = TaskManagerAgent(task_definition=sample_task_def)                        

    # Override parameters for a quick test run if needed
    task_manager_instance.num_generations = settings.GENERATIONS # Use from settings or override
    task_manager_instance.population_size = settings.POPULATION_SIZE 
    task_manager_instance.num_parents_to_select = max(1, settings.POPULATION_SIZE // 2)

    # Define an async function to run the task manager
    async def run_task_manager_test():
        logger.info("Starting TaskManagerAgent test run...")
        try:
            best_programs_found = await task_manager_instance.manage_evolutionary_cycle()
            if best_programs_found:
                print(f"\n*** Evolution Test Complete! Best program(s) found: ***")
                for i, prog in enumerate(best_programs_found):
                    print(f"--- Program {i+1} ---")
                    print(f"ID: {prog.id}")
                    print(f"Generation: {prog.generation}")
                    print(f"Fitness: {prog.fitness_scores}")
                    print(f"Code:\n{prog.code}")
            else:
                print("\n*** Evolution Test Complete! No suitable program was found. ***")
        except Exception as e:
            logger.error("An error occurred during the TaskManagerAgent test run.", exc_info=True)

    # Run the asyncio event loop for the test
    asyncio.run(run_task_manager_test())
