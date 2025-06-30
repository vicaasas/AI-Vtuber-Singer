import asyncio
import json
from typing import Dict, Optional, Callable
import time

import numpy as np
from fastapi import WebSocket
from loguru import logger

from ..chat_group import ChatGroupManager
from ..chat_history_manager import store_message
from ..service_context import ServiceContext
from .group_conversation import process_group_conversation, process_batch_group_conversation
from .single_conversation import process_single_conversation
from .conversation_utils import EMOJI_LIST
from .types import GroupConversationState


async def handle_conversation_trigger(
    msg_type: str,
    data: dict,
    client_uid: str,
    context: ServiceContext,
    websocket: WebSocket,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    chat_group_manager: ChatGroupManager,
    received_data_buffers: Dict[str, np.ndarray],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    broadcast_to_group: Callable,
    client_users: Dict[str, object] = None,
) -> None:
    """Handle triggers that start a conversation"""
    if msg_type == "ai-speak-signal":
        user_input = ""
        await websocket.send_text(
            json.dumps(
                {
                    "type": "full-text",
                    "text": "AI wants to speak something...",
                }
            )
        )
    elif msg_type == "text-input":
        user_input = data.get("text", "")
    else:  # mic-audio-end
        user_input = received_data_buffers[client_uid]
        received_data_buffers[client_uid] = np.array([])

    images = data.get("images")
    session_emoji = np.random.choice(EMOJI_LIST)

    group = chat_group_manager.get_client_group(client_uid)
    if group and len(group.members) > 1:
        # Handle batch group conversation
        await handle_batch_group_conversation(
            group_id=group.group_id,
            client_uid=client_uid,
            user_input=user_input,
            context=context,
            client_contexts=client_contexts,
            client_connections=client_connections,
            broadcast_to_group=broadcast_to_group,
            group_members=list(group.members),
            images=images,
            session_emoji=session_emoji,
            current_conversation_tasks=current_conversation_tasks,
            client_users=client_users,
        )
    else:
        # Use client_uid as task key for individual conversations
        current_conversation_tasks[client_uid] = asyncio.create_task(
            process_single_conversation(
                context=context,
                websocket_send=websocket.send_text,
                client_uid=client_uid,
                user_input=user_input,
                images=images,
                session_emoji=session_emoji,
            )
        )


async def handle_batch_group_conversation(
    group_id: str,
    client_uid: str,
    user_input: str,
    context: ServiceContext,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_to_group: Callable,
    group_members: list,
    images: Optional[dict],
    session_emoji: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    client_users: Dict[str, object] = None,
) -> None:
    """Handle batch group conversation with 7-second collection window"""
    
    # Get or create conversation state
    state = GroupConversationState.get_state(group_id)
    if not state:
        # Convert set to list if needed
        members_list = list(group_members) if isinstance(group_members, set) else group_members
        state = GroupConversationState(
            group_id=group_id,
            session_emoji=session_emoji,
            group_queue=members_list,
            memory_index={uid: 0 for uid in members_list},
        )
    
    # Skip AI speak signals for batch processing
    if isinstance(user_input, str) and user_input == "":
        return
    
    # Process user input to text
    if isinstance(user_input, np.ndarray):
        # Convert audio to text using ASR
        from .conversation_utils import process_user_input
        text_input = await process_user_input(
            user_input, context.asr_engine, client_connections[client_uid].send_text
        )
    else:
        text_input = user_input
    
    # Get user name from WSUser or fallback to context
    user_name = "Human"  # Default fallback
    if client_users and client_uid in client_users:
        user_name = client_users[client_uid].client_name
    elif hasattr(context, 'character_config') and context.character_config.human_name:
        user_name = context.character_config.human_name
    
    # Add message to batch
    state.add_batch_message(user_name, text_input, client_uid)
    logger.info(f"Added message to batch for group {group_id}: {user_name}: {text_input}")
    
    # Convert set to list if needed for broadcasting
    members_list = list(group_members) if isinstance(group_members, set) else group_members
    
    # Broadcast the message to other group members immediately
    await broadcast_to_group(
        members_list,
        {
            "type": "user-input-transcription", 
            "text": text_input,
            "user_name": user_name,
        },
        client_uid,
    )
    
    # Cancel existing timer if it exists
    if state.batch_timer_task and not state.batch_timer_task.done():
        state.batch_timer_task.cancel()
    
    # Start new 7-second timer if not already processing
    if not state.is_processing_batch:
        state.batch_timer_task = asyncio.create_task(
            batch_timer_handler(
                group_id=group_id,
                client_contexts=client_contexts,
                client_connections=client_connections,
                broadcast_to_group=broadcast_to_group,
                group_members=members_list,
                images=images,
                current_conversation_tasks=current_conversation_tasks,
                client_users=client_users,
            )
        )


async def batch_timer_handler(
    group_id: str,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_to_group: Callable,
    group_members: list,
    images: Optional[dict],
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    client_users: Dict[str, object] = None,
) -> None:
    """Handle the 7-second batch timer"""
    try:
        await asyncio.sleep(7.0)  # Wait 7 seconds
        
        state = GroupConversationState.get_state(group_id)
        if not state or len(state.batch_messages) == 0:
            logger.info(f"No messages to process for group {group_id}")
            return
        
        # Mark as processing to prevent new timers
        state.is_processing_batch = True
        
        logger.info(f"Processing batch for group {group_id} with {len(state.batch_messages)} messages")
        
        # Start batch conversation task
        task_key = group_id
        try:
            current_conversation_tasks[task_key] = asyncio.create_task(
                process_batch_group_conversation(
                    group_id=group_id,
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_to_group,
                    group_members=group_members,
                    images=images,
                    session_emoji=state.session_emoji,
                    client_users=client_users,
                )
            )
            logger.info(f"Batch conversation task created for group {group_id}")
            
            # Wait for the task to complete and handle any errors
            await current_conversation_tasks[task_key]
            logger.info(f"Batch conversation task completed for group {group_id}")
            
        except Exception as e:
            logger.error(f"Error in batch conversation task for group {group_id}: {e}", exc_info=True)
            # Reset processing state on error
            if state:
                state.is_processing_batch = False
            raise
        finally:
            # Clean up task reference
            current_conversation_tasks.pop(task_key, None)
        
    except asyncio.CancelledError:
        logger.info(f"Batch timer cancelled for group {group_id}")
        # Reset processing state on cancellation
        state = GroupConversationState.get_state(group_id)
        if state:
            state.is_processing_batch = False
    except Exception as e:
        logger.error(f"Error in batch timer handler for group {group_id}: {e}", exc_info=True)
        # Reset processing state on error
        state = GroupConversationState.get_state(group_id)
        if state:
            state.is_processing_batch = False


async def handle_individual_interrupt(
    client_uid: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    context: ServiceContext,
    heard_response: str,
):
    if client_uid in current_conversation_tasks:
        task = current_conversation_tasks[client_uid]
        if task and not task.done():
            task.cancel()
            logger.info("🛑 Conversation task was successfully interrupted")

        try:
            context.agent_engine.handle_interrupt(heard_response)
        except Exception as e:
            logger.error(f"Error handling interrupt: {e}")

        if context.history_uid:
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="ai",
                content=heard_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )
            store_message(
                conf_uid=context.character_config.conf_uid,
                history_uid=context.history_uid,
                role="system",
                content="[Interrupted by user]",
            )


async def handle_group_interrupt(
    group_id: str,
    heard_response: str,
    current_conversation_tasks: Dict[str, Optional[asyncio.Task]],
    chat_group_manager: ChatGroupManager,
    client_contexts: Dict[str, ServiceContext],
    broadcast_to_group: Callable,
) -> None:
    """Handles interruption for a group conversation"""
    task = current_conversation_tasks.get(group_id)
    if not task or task.done():
        return

    # Get state and speaker info before cancellation
    state = GroupConversationState.get_state(group_id)
    current_speaker_uid = state.current_speaker_uid if state else None

    # Get context from current speaker
    context = None
    group = chat_group_manager.get_group_by_id(group_id)
    if current_speaker_uid:
        context = client_contexts.get(current_speaker_uid)
        logger.info(f"Found current speaker context for {current_speaker_uid}")
    if not context and group and group.members:
        logger.warning(f"No context found for group {group_id}, using first member")
        context = client_contexts.get(next(iter(group.members)))

    # Now cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.info(f"🛑 Group conversation {group_id} cancelled successfully.")

    current_conversation_tasks.pop(group_id, None)
    GroupConversationState.remove_state(group_id)  # Clean up state after we've used it

    # Store messages with speaker info
    if context and group:
        for member_uid in group.members:
            if member_uid in client_contexts:
                try:
                    member_ctx = client_contexts[member_uid]
                    member_ctx.agent_engine.handle_interrupt(heard_response)
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="ai",
                        content=heard_response,
                        name=context.character_config.character_name,
                        avatar=context.character_config.avatar,
                    )
                    store_message(
                        conf_uid=member_ctx.character_config.conf_uid,
                        history_uid=member_ctx.history_uid,
                        role="system",
                        content="[Interrupted by user]",
                    )
                except Exception as e:
                    logger.error(f"Error handling interrupt for {member_uid}: {e}")

    await broadcast_to_group(
        list(group.members),
        {
            "type": "interrupt-signal",
            "text": "conversation-interrupted",
        },
    )
