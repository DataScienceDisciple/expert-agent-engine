import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Configure logging early
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    stream=sys.stdout)  # Log to stdout for CLI
logger = logging.getLogger(__name__)

# Add project root to sys.path if needed (adjust based on your structure)
# project_root = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(project_root))

try:
    from src.config_loader import load_config, ConfigLoaderError
    from src.engine import ConversationEngine
except ImportError as e:
    logger.error(
        f"Failed to import necessary modules: {e}. Ensure script is run from project root or PYTHONPATH is set.")
    sys.exit(1)


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the Expert Agent conversation flow.")
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the configuration file (JSON or YAML)."
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Override the maximum number of conversation turns specified in the config file."
    )
    return parser.parse_args()


async def main():
    """Main asynchronous function to run the conversation."""
    args = parse_arguments()
    config_path = Path(args.config_file)

    if not config_path.is_file():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from: {config_path}")
    try:
        config = load_config(str(config_path))
    except (ConfigLoaderError, FileNotFoundError) as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Override max_iterations if provided via CLI
    if args.max_iterations is not None:
        if args.max_iterations > 0:
            logger.info(
                f"Overriding maxIterations from config. Using CLI value: {args.max_iterations}")
            config.max_iterations = args.max_iterations
        else:
            logger.warning(
                "Invalid --max-iterations value provided. Must be positive. Using value from config.")

    logger.info("Initializing Conversation Engine...")
    try:
        engine = ConversationEngine(config=config)
    except RuntimeError as e:
        logger.error(f"Failed to initialize conversation engine: {e}")
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected init errors
        logger.error(
            f"Unexpected error during engine initialization: {e}", exc_info=True)
        sys.exit(1)

    logger.info("Starting conversation run...")
    try:
        final_history = await engine.run_conversation()
        logger.info("Conversation run completed successfully.")
        # OutputManager already logs file paths upon successful save
        # Optionally, print final history summary here if desired
        # final_messages = final_history.get_history()
        # logger.info(f"Conversation finished after {final_history.current_turn} turns.")
    except Exception as e:
        logger.error(
            f"An error occurred during the conversation run: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    # Ensure the script is run with an async event loop
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nConversation interrupted by user.")
        sys.exit(0)
