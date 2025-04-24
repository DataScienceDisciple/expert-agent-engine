from dataclasses import dataclass


@dataclass
class ConversationContext:
    """Holds shared context/state for an agent conversation run."""
    user_goal: str
