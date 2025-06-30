from typing import List, Dict, Callable, Optional, TypedDict, Awaitable, ClassVar
from dataclasses import dataclass, field
from pydantic import BaseModel
import asyncio
import time

from ..agent.output_types import Actions, DisplayText

# Type definitions
WebSocketSend = Callable[[str], Awaitable[None]]
BroadcastFunc = Callable[[List[str], dict, Optional[str]], Awaitable[None]]


class AudioPayload(TypedDict):
    """Type definition for audio payload"""

    type: str
    audio: Optional[str]
    volumes: Optional[List[float]]
    slice_length: Optional[int]
    display_text: Optional[DisplayText]
    actions: Optional[Actions]
    forwarded: Optional[bool]


@dataclass
class BroadcastContext:
    """Context for broadcasting messages in group chat"""

    broadcast_func: Optional[BroadcastFunc] = None
    group_members: Optional[List[str]] = None
    current_client_uid: Optional[str] = None


class ConversationConfig(BaseModel):
    """Configuration for conversation chain"""

    conf_uid: str = ""
    history_uid: str = ""
    client_uid: str = ""
    character_name: str = "AI"


@dataclass
class BatchMessage:
    """Single message in batch collection"""
    user_name: str
    content: str
    timestamp: float
    client_uid: str


@dataclass
class GroupConversationState:
    """State for group conversation"""

    # Class variable to track current states
    _states: ClassVar[Dict[str, "GroupConversationState"]] = {}

    group_id: str
    conversation_history: List[str] = field(default_factory=list)
    memory_index: Dict[str, int] = field(default_factory=dict)
    group_queue: List[str] = field(default_factory=list)
    session_emoji: str = ""
    current_speaker_uid: Optional[str] = None
    
    # Batch processing fields
    batch_messages: List[BatchMessage] = field(default_factory=list)
    batch_timer_task: Optional[asyncio.Task] = None
    is_processing_batch: bool = False
    last_activity_time: float = field(default_factory=time.time)

    def __post_init__(self):
        """Register state instance after initialization"""
        GroupConversationState._states[self.group_id] = self

    @classmethod
    def get_state(cls, group_id: str) -> Optional["GroupConversationState"]:
        """Get conversation state by group_id"""
        return cls._states.get(group_id)

    @classmethod
    def remove_state(cls, group_id: str) -> None:
        """Remove conversation state when done"""
        state = cls._states.get(group_id)
        if state and state.batch_timer_task:
            state.batch_timer_task.cancel()
        cls._states.pop(group_id, None)

    def add_batch_message(self, user_name: str, content: str, client_uid: str) -> None:
        """Add a message to the batch queue"""
        self.batch_messages.append(BatchMessage(
            user_name=user_name,
            content=content,
            timestamp=time.time(),
            client_uid=client_uid
        ))
        self.last_activity_time = time.time()

    def format_batch_messages(self) -> str:
        """Format collected messages for LLM input"""
        if not self.batch_messages:
            return ""
        
        formatted_messages = []
        for i, msg in enumerate(self.batch_messages, 1):
            # Format: number. [UserName: message content]
            formatted_messages.append(f"{i}. [{msg.user_name}: {msg.content}]")
        
        return "\n".join(formatted_messages)

    def clear_batch_messages(self) -> None:
        """Clear the batch message queue"""
        self.batch_messages.clear()
