import logging
import os
from typing import Optional, TYPE_CHECKING

# Third-party imports
from agents import Agent
# Needed for dynamic instructions type hint
# from agents.runners import RunContextWrapper

# Local application imports
from src.config_loader import AppConfig
from src.context import ConversationContext

# --- Type Hinting for Forward Reference ---
if TYPE_CHECKING:
    from agents.run_context import RunContextWrapper

# Configure basic logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _set_openai_key(api_key: Optional[str]):
    """Sets the OpenAI API key environment variable if provided.

    The openai-agents SDK primarily relies on the environment variable.
    Also sets openai.api_key for compatibility if using the client directly.
    """
    if api_key:
        os.environ['OPENAI_API_KEY'] = api_key
        logger.info("Set OPENAI_API_KEY environment variable from config.")
        # Set for direct client usage as well, though SDK might not need this directly
        try:
            import openai
            openai.api_key = api_key
        except ImportError:
            logger.warning(
                "'openai' library not directly importable, skipping direct openai.api_key set.")

    elif not os.getenv('OPENAI_API_KEY'):
        logger.error(
            "OpenAI API Key not found in config and OPENAI_API_KEY environment variable is not set.")
        raise ValueError(
            "OpenAI API Key must be provided either in config file or as OPENAI_API_KEY environment variable.")
    else:
        logger.info("Using existing OPENAI_API_KEY environment variable.")

# --- Placeholder for dynamic instructions function (Task 3.4) ---


def _get_user_agent_instructions(context: 'RunContextWrapper[ConversationContext]', agent: Agent[ConversationContext]) -> str:
    """Generates dynamic instructions for the User Agent.

    These instructions guide the LLM to act as the User Agent,
    focusing on generating the next question based on the goal and history.
    """
    goal = context.context.user_goal
    # The conversation history is implicitly available to the LLM via the Runner
    logger.debug(
        f"Generating dynamic instructions for User Agent with goal: {goal}")
    return (
        f"You are a User Agent simulating a human researcher. Your overall goal is: \"{goal}\". "
        f"You are in a conversation with an expert. Your *only* task right now is to generate the single, specific, open-ended question you should ask the expert next to progress towards your goal, based on the conversation history provided so far. "
        f"Analyze the most recent turns of the conversation, especially the last response from the expert. "
        f"Formulate a concise follow-up question that probes deeper into relevant information, asks for clarification, or explores a new angle directly related to achieving your goal: \"{goal}\". "
        f"IMPORTANT: Your output MUST be *only the question text itself*. Do not include any preamble, self-correction, explanation, or surrounding text like 'Here is my question:' or 'Okay, I will ask:'. Just output the question."
    )

# --- Placeholder for User Agent creation function (Task 3.5) ---


def create_user_agent(config: AppConfig) -> Agent[ConversationContext]:
    """Creates the User Agent that generates follow-up questions."""
    _set_openai_key(config.openai_api_key)  # Ensure key is available
    logger.info(f"Creating User Agent with model: {config.openai_model}")

    # Determine if we're using extended config or legacy
    instructions = None
    if config.user_agent_config:
        # Use the new extended configuration
        instructions = config.get_user_agent_instructions()
        logger.info(
            f"Using extended User Agent configuration: {config.user_agent_config.name}")
    else:
        # Use dynamic instructions function
        instructions = _get_user_agent_instructions
        logger.info("Using dynamic User Agent instructions")

    try:
        user_agent = Agent[ConversationContext](
            name="UserAgent" if not config.user_agent_config else config.user_agent_config.name,
            instructions=instructions,
            model=config.openai_model
            # Potentially adjust model_settings like temperature if needed
        )
        logger.info("User Agent instance created successfully.")
        return user_agent
    except Exception as e:
        logger.error(
            f"Failed to create User Agent instance: {e}", exc_info=True)
        raise

# --- Placeholder for Expert Agent creation function (Task 3.6) ---


def create_expert_agent(config: AppConfig) -> Agent[ConversationContext]:
    """Creates the Expert Agent that answers questions based on persona."""
    _set_openai_key(config.openai_api_key)  # Ensure key is available

    # Determine if we're using extended config or legacy
    expert_instructions = config.get_expert_agent_instructions()
    expert_name = "ExpertAgent"

    if config.expert_agent_config:
        expert_name = config.expert_agent_config.name
        logger.info(
            f"Using extended Expert Agent configuration: {expert_name}")

    logger.info(
        f"Creating Expert Agent with persona starting: '{expert_instructions[:50]}...' and model: {config.openai_model}")

    try:
        expert_agent = Agent[ConversationContext](
            name=expert_name,
            instructions=expert_instructions,  # Either from extended config or legacy persona
            model=config.openai_model
            # Add tools or handoffs here if needed in the future
        )
        logger.info("Expert Agent instance created successfully.")
        return expert_agent
    except Exception as e:
        logger.error(
            f"Failed to create Expert Agent instance: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # This block provides basic manual testing for the agent factory.
    # Requires a valid OPENAI_API_KEY environment variable to be set.

    logger.info("Starting agent_factory.py manual tests...")

    # --- Test _set_openai_key --- (Keep this basic check)
    api_key_in_env = os.getenv('OPENAI_API_KEY')
    if not api_key_in_env:
        logger.warning(
            "OPENAI_API_KEY environment variable not set. Some tests will be skipped.")
        # Test that it raises error if key is missing
        try:
            _set_openai_key(None)
            logger.error(
                "_set_openai_key did not raise error when key was missing!")
        except ValueError:
            logger.info(
                "Correctly caught ValueError when API key was missing.")
    else:
        logger.info(
            "Testing _set_openai_key with existing environment variable.")
        _set_openai_key(None)  # Should use existing env var
        logger.info("Testing _set_openai_key with explicit value.")
        _set_openai_key("sk-dummykeyfromconfig")  # Test setting it
        # Reset env var to original value for subsequent tests
        os.environ['OPENAI_API_KEY'] = api_key_in_env

    # --- Test Agent Creation (Requires API Key) ---
    if not api_key_in_env:
        logger.warning(
            "Skipping Agent Creation tests as OPENAI_API_KEY is not set.")
    else:
        from src.config_loader import load_config, ConfigLoaderError
        from pathlib import Path

        # Create dummy config and context for testing
        dummy_config_path = Path("config/temp_factory_test_config.yaml")
        dummy_output_dir = Path("temp_factory_output")
        cleanup_needed = False
        try:
            dummy_config_path.parent.mkdir(exist_ok=True)
            dummy_output_dir.mkdir(exist_ok=True)
            dummy_config_path.write_text(
                "userAgentGoal: Test the agent factory functions.\n"
                "expertAgentPersona: You are a test expert agent persona.\n"
                "maxIterations: 1\n"
                f"openaiModel: gpt-4.1-mini\n"
                f"outputDir: {str(dummy_output_dir)}\n"
                "historyFilePath: null\n"
                # API key loaded from environment
            )
            cleanup_needed = True
            logger.info(f"Loading test config from: {dummy_config_path}")
            test_config = load_config(str(dummy_config_path))

            logger.info("\n--- Testing User Agent Creation ---")
            test_context = ConversationContext(
                user_goal=test_config.user_agent_goal)
            user_agent = create_user_agent(test_config)
            print(
                f"Successfully created User Agent: Name='{user_agent.name}', Model='{user_agent.model}'")
            assert isinstance(user_agent, Agent)

            logger.info("\n--- Testing Expert Agent Creation ---")
            expert_agent = create_expert_agent(test_config)
            print(
                f"Successfully created Expert Agent: Name='{expert_agent.name}', Model='{expert_agent.model}'")
            print(
                f"Expert Agent Instructions (start): '{expert_agent.instructions[:60]}...'")
            assert isinstance(expert_agent, Agent)

            logger.info("\nAgent creation tests passed.")

        except (ConfigLoaderError, ValueError, ImportError, Exception) as e:
            logger.error(
                f"Error during agent creation testing: {e}", exc_info=True)
        finally:
            # Clean up dummy files
            if cleanup_needed:
                dummy_config_path.unlink(missing_ok=True)
                if dummy_output_dir.exists():
                    dummy_output_dir.rmdir()  # Only remove if empty
                logger.info("Cleaned up temporary test files.")
