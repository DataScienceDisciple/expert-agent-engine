import logging
import asyncio
from typing import Optional
from pathlib import Path

# Third-party imports
from agents import Agent, Runner

# Local application imports
from src.config_loader import AppConfig
from src.context import ConversationContext
from src.conversation import ConversationHistory
from src.agent_factory import create_user_agent, create_expert_agent
from src.output_manager import OutputManager

logger = logging.getLogger(__name__)


class ConversationEngine:
    """Orchestrates the conversation flow between User and Expert agents."""

    def __init__(self, config: AppConfig):
        """Initializes the engine with configuration and agents."""
        logger.info("Initializing ConversationEngine...")
        self.config = config
        self.history = ConversationHistory(
            max_iterations=config.max_iterations,
            history_file_path=config.history_file_path
        )
        self.context = ConversationContext(user_goal=config.user_agent_goal)
        self.output_manager = OutputManager(output_dir=str(config.output_dir))

        try:
            logger.info("Creating User Agent...")
            self.user_agent: Agent[ConversationContext] = create_user_agent(
                config)
            logger.info("Creating Expert Agent...")
            self.expert_agent: Agent[ConversationContext] = create_expert_agent(
                config)
            logger.info("ConversationEngine initialized successfully.")
        except Exception as e:
            logger.error(
                f"Failed to initialize agents in ConversationEngine: {e}", exc_info=True)
            # Propagate the error to prevent engine use with failed setup
            raise RuntimeError(
                "Failed to initialize agents for ConversationEngine") from e

    # --- Placeholder for run_conversation (Task 5.2+) ---
    async def run_conversation(self) -> ConversationHistory:
        """Runs the main conversation loop."""
        logger.info("Starting conversation run...")
        logger.info(f"Max iterations: {self.history.max_iterations}")

        while not self.history.is_complete():
            current_turn = self.history.current_turn + 1  # Turns are 1-based for logging
            logger.info(
                f"--- Starting Turn {current_turn}/{self.history.max_iterations} ---")

            # --- User Agent Turn Logic (Task 5.3) --- #
            logger.info(
                f"Turn {current_turn}: User Agent generating question...")
            question = None
            try:
                # Get current history or create initial message if empty
                current_history_for_user = self.history.get_history()

                # If this is the first turn and history is empty, provide an initial system message
                if not current_history_for_user:
                    # Create initial input with system message containing the user goal
                    initial_input = [
                        {"role": "system", "content": f"The conversation goal is: {self.context.user_goal}"}]
                    user_agent_input = initial_input
                else:
                    user_agent_input = current_history_for_user

                # The Runner expects the history list as the main input
                user_agent_result = await Runner.run(
                    self.user_agent,
                    input=user_agent_input,
                    context=self.context
                )
                if user_agent_result and isinstance(user_agent_result.final_output, str):
                    question = user_agent_result.final_output.strip()
                    logger.info(
                        f"Turn {current_turn}: User Agent generated question: '{question[:100]}...'")
                else:
                    logger.warning(
                        f"Turn {current_turn}: User Agent did not produce a valid string output. Result: {user_agent_result}")
            except Exception as e:
                logger.error(
                    f"Turn {current_turn}: Error during User Agent run: {e}", exc_info=True)
                # Decide how to handle - break loop? Skip turn? For now, log and break.
                logger.error("Aborting conversation due to User Agent error.")
                break  # Exit the loop on error

            if not question:
                logger.warning(
                    f"Turn {current_turn}: User Agent failed to generate a question. Skipping turn.")
                # Increment turn even if agent failed, to avoid infinite loop if it always fails
                self.history.increment_turn()  # Moved from 5.5 for this case
                continue  # Skip expert turn if no question

            # Add generated question to history
            self.history.add_message(role="user", content=question)

            # --- Expert Agent Turn Logic (Task 5.4) --- #
            logger.info(
                f"Turn {current_turn}: Expert Agent generating answer...")
            answer = None
            try:
                current_history_for_expert = self.history.get_history()
                # Pass the history including the latest user question
                expert_agent_result = await Runner.run(
                    self.expert_agent,
                    input=current_history_for_expert,
                    context=self.context
                )
                if expert_agent_result and isinstance(expert_agent_result.final_output, str):
                    answer = expert_agent_result.final_output.strip()
                    # Log length instead of potentially long/sensitive answer
                    logger.info(
                        f"Turn {current_turn}: Expert Agent generated answer. Length: {len(answer)}")
                    # Debug log snippet
                    logger.debug(f"Expert answer snippet: {answer[:100]}...")
                else:
                    logger.warning(
                        f"Turn {current_turn}: Expert Agent did not produce a valid string output. Result: {expert_agent_result}")

            except Exception as e:
                logger.error(
                    f"Turn {current_turn}: Error during Expert Agent run: {e}", exc_info=True)
                # Decide how to handle - break loop? Log and continue? Log and break.
                logger.error(
                    "Aborting conversation due to Expert Agent error.")
                break  # Exit the loop on error

            if not answer:
                logger.warning(
                    f"Turn {current_turn}: Expert Agent failed to generate an answer. Ending conversation.")
                # If the expert fails, we probably can't continue meaningfully
                break  # Exit loop

            # Add generated answer to history
            self.history.add_message(role="assistant", content=answer)

            # --- Turn Increment (Task 5.5) --- #
            self.history.increment_turn()
            logger.info(f"--- Completed Turn {current_turn} ---")

        logger.info("Conversation run finished.")

        # --- Save the conversation transcript (Task 8) --- #
        output_file_path: Optional[Path] = None
        try:
            # Generate a filename prefix using the user goal (truncated if too long)
            goal_prefix = self.context.user_goal[:30].replace(' ', '_').lower()
            if len(goal_prefix) < 5:  # If goal is too short, use default
                goal_prefix = "conversation"

            # Save the conversation transcript
            transcript_filename = self.output_manager.generate_filename(
                prefix=goal_prefix)
            output_file_path = self.output_manager.save_transcript(
                self.history,
                filename=transcript_filename
            )
            logger.info(
                f"Saved conversation transcript to: {output_file_path}")
        except Exception as e:
            logger.error(
                f"Failed to save conversation transcript: {e}", exc_info=True)

        # --- Generate and Save Takeaways (Task 9) --- #
        if output_file_path:
            try:
                takeaways = await self.output_manager.generate_takeaways(
                    history=self.history,
                    context=self.context,
                    expert_agent=self.expert_agent,
                    config=self.config
                )
                if not takeaways.startswith("Error:"):
                    takeaway_file_path = self.output_manager.save_takeaways(
                        takeaways,
                        transcript_filename=output_file_path.name
                    )
                    logger.info(
                        f"Saved takeaways to: {takeaway_file_path}")
                else:
                    logger.error(
                        f"Skipping takeaway saving due to generation error: {takeaways}")
            except Exception as e:
                logger.error(
                    f"Failed to generate or save takeaways: {e}", exc_info=True)
        else:
            logger.warning(
                "Skipping takeaway generation because transcript saving failed.")

        return self.history

    # --- Placeholder for get_final_history (Task 5.6) ---
    def get_final_history(self) -> ConversationHistory:
        """Returns the final conversation history object."""
        return self.history
