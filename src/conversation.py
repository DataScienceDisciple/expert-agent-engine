import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# Define type hints for clarity
Message = Dict[str, str]  # e.g., {"role": "user", "content": "..."}
History = List[Message]


class ConversationHistory:
    """Manages the conversation history between agents."""

    def __init__(self, max_iterations: int, history_file_path: Optional[Path] = None):
        """Initializes the history, optionally loading from a file.

        Args:
            max_iterations: The maximum number of User Agent -> Expert Agent turns.
            history_file_path: Optional path to a file containing initial history.
        """
        self.max_iterations: int = max_iterations
        self.messages: History = []
        self.current_turn: int = 0

        if history_file_path:
            self._load_from_file(history_file_path)

    def add_message(self, role: str, content: str):
        """Adds a message to the history.

        Args:
            role: The role of the message sender ('user', 'assistant', 'system').
            content: The message content.
        """
        if role not in ["user", "assistant", "system"]:
            logger.warning(
                f"Attempted to add message with invalid role: {role}")
            # Decide if we should raise error or just skip
            # For now, let's be strict
            raise ValueError(
                f"Invalid role '{role}'. Must be 'user', 'assistant', or 'system'.")

        if not isinstance(content, str):
            logger.warning("Attempted to add non-string content to history.")
            raise ValueError("Message content must be a string.")

        message: Message = {"role": role, "content": content}
        self.messages.append(message)
        logger.debug(
            f"Added message: Role={role}, Content Length={len(content)}")

    def get_history(self) -> History:
        """Returns the current conversation history list."""
        return self.messages.copy()  # Return a copy to prevent external modification

    def _load_from_file(self, file_path: Path):
        """Loads initial history from a text file.

        For MVP, treats the entire file content as a single initial 'system' message
        to provide context.
        Future enhancement: Parse structured history if needed.
        """
        logger.info(f"Attempting to load initial history from: {file_path}")
        try:
            if not file_path.exists() or not file_path.is_file():
                logger.error(
                    f"History file path does not exist or is not a file: {file_path}")
                # Depending on requirements, we might raise error or just warn
                raise FileNotFoundError(f"History file not found: {file_path}")

            content = file_path.read_text(encoding='utf-8').strip()
            if content:
                # Prepend as a system message to provide context
                # Note: Roles 'user' or 'assistant' might also make sense depending on use case.
                self.add_message(
                    "system", f"Initial context from history file ({file_path.name}):\n---\n{content}\n---")
                logger.info(
                    f"Successfully loaded and prepended history from {file_path.name} as system message.")
            else:
                logger.warning(f"History file {file_path.name} was empty.")

        except FileNotFoundError as e:
            raise e  # Re-raise specific error
        except Exception as e:
            logger.error(
                f"Failed to load or process history file {file_path}: {e}", exc_info=True)
            # Decide how to handle this - fail hard or continue without history?
            # For now, let's raise an error.
            raise IOError(f"Error processing history file {file_path}: {e}")

    def format_for_display(self) -> str:
        """Formats the conversation history for simple display or saving."""
        lines = []
        for msg in self.messages:
            role = msg.get("role", "unknown").capitalize()
            content = msg.get("content", "")
            lines.append(f"--- {role} ---")
            lines.append(content)
            lines.append("\n")  # Add space between messages
        return "\n".join(lines)

    def increment_turn(self):
        """Increments the conversation turn counter."""
        self.current_turn += 1
        logger.info(
            f"Advanced to turn {self.current_turn}/{self.max_iterations}")

    def is_complete(self) -> bool:
        """Checks if the conversation has reached the maximum number of turns."""
        complete = self.current_turn >= self.max_iterations
        if complete:
            logger.info(
                f"Conversation reached max iterations ({self.max_iterations}).")
        return complete


# Basic test block
if __name__ == '__main__':
    logger.info("Running ConversationHistory tests...")
    test_history_path = Path("config/temp_history_test.txt")
    test_history_path.parent.mkdir(exist_ok=True)
    test_history_path.write_text("This is the initial context from a file.")

    try:
        print("\n--- Testing Initialization with File ---")
        history = ConversationHistory(
            max_iterations=5, history_file_path=test_history_path)
        print(f"Initial messages count: {len(history.messages)}")
        print(
            f"First message content (start): {history.messages[0]['content'][:50]}...")
        assert len(history.messages) == 1
        assert history.messages[0]['role'] == 'system'

        print("\n--- Testing Adding Messages ---")
        history.add_message("user", "Hello expert agent!")
        history.add_message("assistant", "Hello user! How can I help?")
        print(f"Messages count after adding: {len(history.messages)}")
        assert len(history.messages) == 3

        print("\n--- Testing Turn Counting & Completion ---")
        print(
            f"Is complete? {history.is_complete()} (Turn: {history.current_turn})")
        assert not history.is_complete()
        for i in range(5):
            history.increment_turn()
            print(
                f"Is complete? {history.is_complete()} (Turn: {history.current_turn})")
        assert history.is_complete()

        print("\n--- Testing History Formatting ---")
        display_text = history.format_for_display()
        print("Formatted History (first 100 chars):")
        print(display_text[:100] + "...")
        assert "--- System ---" in display_text
        assert "--- User ---" in display_text
        assert "--- Assistant ---" in display_text

        print("\n--- Testing Initialization without File ---")
        history_no_file = ConversationHistory(max_iterations=3)
        print(
            f"Initial messages count (no file): {len(history_no_file.messages)}")
        assert len(history_no_file.messages) == 0

        print("\n--- Testing Invalid Role ---")
        try:
            history.add_message("invalid_role", "test")
        except ValueError as e:
            print(f"Caught expected ValueError: {e}")

    except Exception as e:
        logger.error(
            f"Error during ConversationHistory tests: {e}", exc_info=True)
    finally:
        # Clean up dummy file
        test_history_path.unlink(missing_ok=True)
        logger.info("ConversationHistory tests finished.")
