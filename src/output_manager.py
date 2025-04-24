import os
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
from agents import Agent, Runner

# Local application imports
from src.conversation import ConversationHistory
from src.context import ConversationContext
from src.config_loader import AppConfig

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages output file operations for conversation transcripts and takeaways."""

    def __init__(self, output_dir: str):
        """Initialize the OutputManager with the target output directory.

        Args:
            output_dir: The directory path where output files will be saved
        """
        self.output_dir = Path(output_dir)
        self._ensure_output_dir_exists()

    def _ensure_output_dir_exists(self) -> None:
        """Create the output directory if it doesn't exist."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured output directory exists: {self.output_dir}")
        except Exception as e:
            logger.error(
                f"Failed to create output directory {self.output_dir}: {e}")
            raise RuntimeError(f"Cannot create output directory: {e}") from e

    def generate_filename(self, prefix: str = "conversation", extension: str = "txt") -> str:
        """Generate a unique filename with timestamp.

        Args:
            prefix: Prefix for the filename
            extension: File extension without the dot

        Returns:
            A unique filename with timestamp
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{extension}"

    def format_conversation(self, history: ConversationHistory) -> str:
        """Format conversation history as plain text.

        Args:
            history: The conversation history object

        Returns:
            Formatted text of the conversation
        """
        messages = history.get_history()
        if not messages:
            return "No conversation history available."

        lines = []
        lines.append(
            f"# Conversation Transcript (Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append(f"# Total Turns: {history.current_turn}")
        lines.append("")

        for i, message in enumerate(messages, 1):
            role = message.get('role', 'unknown')
            content = message.get('content', '')

            # Format the role title with proper capitalization
            if role == 'user':
                role_title = "User"
            elif role == 'assistant':
                role_title = "Expert"
            elif role == 'system':
                role_title = "System"
            else:
                role_title = role.capitalize()

            lines.append(
                f"## {role_title} (Turn {(i+1)//2 if role != 'system' else 0})")
            lines.append(content)
            lines.append("")  # Add blank line between messages

        return "\n".join(lines)

    def save_transcript(self, history: ConversationHistory, filename: Optional[str] = None) -> Path:
        """Save conversation transcript to a file.

        Args:
            history: The conversation history to save
            filename: Optional specific filename to use (otherwise auto-generated)

        Returns:
            Path to the saved file
        """
        if not filename:
            filename = self.generate_filename()

        file_path = self.output_dir / filename
        formatted_text = self.format_conversation(history)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            logger.info(
                f"Successfully saved conversation transcript to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save transcript to {file_path}: {e}")
            raise RuntimeError(f"Failed to save transcript: {e}") from e

    async def generate_takeaways(self,
                                 history: ConversationHistory,
                                 context: ConversationContext,
                                 expert_agent: Agent[ConversationContext],
                                 config: AppConfig) -> str:
        """Generate takeaways from the conversation using the Expert Agent.

        Args:
            history: The conversation history object
            context: The conversation context (for the goal)
            expert_agent: The expert agent instance to use for generation
            config: The application configuration

        Returns:
            A string containing the generated takeaways, or an error message.
        """
        logger.info("Generating takeaways from conversation...")
        full_history = history.get_history()
        if not full_history:
            logger.warning("Cannot generate takeaways from empty history.")
            return "Error: Cannot generate takeaways from empty history."

        takeaway_prompt = (
            f"Based on the preceding conversation transcript where the initial goal was \"{context.user_goal}\", "
            f"please distill the key takeaways, insights, and conclusions. "
            f"Focus on the most important information relevant to the original goal. "
            f"Present the takeaways clearly and concisely, perhaps as bullet points."
            f"Do not refer to yourself, just provide the takeaways."
        )

        # Create the input for the summarization call
        # We add the summarization prompt as a final user message
        summarization_input = full_history + [
            {"role": "user", "content": takeaway_prompt}
        ]

        try:
            # Use the expert agent to generate takeaways
            # Note: We are reusing the expert agent, which might not be ideal
            # if its persona strongly influences the summarization style.
            # A dedicated summarizer agent could be better in complex scenarios.
            takeaway_result = await Runner.run(
                expert_agent,
                input=summarization_input,
                context=context  # Context might not be strictly needed here but pass for consistency
            )

            if takeaway_result and isinstance(takeaway_result.final_output, str):
                takeaways = takeaway_result.final_output.strip()
                logger.info(
                    f"Successfully generated takeaways. Length: {len(takeaways)}")
                return takeaways
            else:
                logger.warning(
                    f"Failed to generate valid takeaways. Result: {takeaway_result}")
                return "Error: Failed to generate valid takeaways from the LLM."
        except Exception as e:
            logger.error(
                f"Error during takeaway generation: {e}", exc_info=True)
            return f"Error generating takeaways: {e}"

    def save_takeaways(self, takeaways: str, transcript_filename: str) -> Path:
        """Save takeaways to a separate file, linked to the transcript.

        Args:
            takeaways: The generated takeaways text.
            transcript_filename: The filename of the original transcript.

        Returns:
            Path to the saved takeaways file.
        """
        base_name = Path(transcript_filename).stem
        takeaways_filename = f"{base_name}_takeaways.txt"
        file_path = self.output_dir / takeaways_filename

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Takeaways from: {transcript_filename}\n\n")
                f.write(takeaways)
            logger.info(f"Successfully saved takeaways to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to save takeaways to {file_path}: {e}")
            raise RuntimeError(f"Failed to save takeaways: {e}") from e
