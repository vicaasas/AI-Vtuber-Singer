from typing import Any, Dict, List, Optional, Union
import asyncio
import json
from loguru import logger
from fastapi import WebSocket
import numpy as np
import re

from .conversation_utils import (
    create_batch_input,
    process_agent_output,
    process_user_input,
    finalize_conversation_turn,
    cleanup_conversation,
    EMOJI_LIST,
)
from .types import (
    BroadcastFunc,
    GroupConversationState,
    BroadcastContext,
    WebSocketSend,
)
from ..service_context import ServiceContext
from ..chat_history_manager import store_message
from .tts_manager import TTSTaskManager, GroupTTSTaskManager


async def process_batch_group_conversation(
    group_id: str,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
) -> None:
    """Process batch group conversation with collected messages
    
    Args:
        group_id: ID of the group
        client_contexts: Dictionary of client contexts
        client_connections: Dictionary of client WebSocket connections
        broadcast_func: Function to broadcast messages to group
        group_members: List of group member UIDs
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
    """
    
    state = GroupConversationState.get_state(group_id)
    if not state or len(state.batch_messages) == 0:
        logger.warning(f"No batch messages found for group {group_id}")
        return
    
    # Convert to list if it's a set and ensure we have members
    members_list = list(group_members) if isinstance(group_members, set) else group_members
    if not members_list:
        logger.error(f"No group members found for group {group_id}")
        return
    
    # Select the first available AI to respond (you can modify this logic)
    responding_ai_uid = members_list[0]
    responding_context = client_contexts.get(responding_ai_uid)
    
    if not responding_context:
        logger.error(f"No context found for responding AI {responding_ai_uid}")
        await broadcast_func(
            members_list,
            {
                "type": "error",
                "message": f"No AI context found for {responding_ai_uid}",
            },
        )
        return
    
    logger.info(f"Selected AI {responding_ai_uid} to respond for group {group_id}")
    
    # Use GroupTTSTaskManager for broadcasting audio to all members
    tts_manager = GroupTTSTaskManager(
        broadcast_func=broadcast_func,
        group_members=members_list,
        responding_client_uid=responding_ai_uid,
    )
    
    try:
        logger.info(f"Batch Group Conversation {session_emoji} started for group {group_id}!")
        
        # Format batch messages for LLM
        formatted_input = state.format_batch_messages()
        logger.info(f"Formatted batch input: {formatted_input}")
        
        # Broadcast thinking state
        await broadcast_func(
            members_list,
            {"type": "control", "text": "conversation-chain-start"},
        )
        await broadcast_func(
            members_list,
            {"type": "full-text", "text": "Processing messages..."},
        )
        
        # Initialize group conversation context
        try:
            init_group_conversation_contexts(client_contexts)
            logger.info("Group conversation contexts initialized")
        except Exception as e:
            logger.error(f"Error initializing group conversation contexts: {e}")
        
        # Store messages in history for all group members
        try:
            for msg in state.batch_messages:
                for member_uid in members_list:
                    if member_uid in client_contexts:
                        member_context = client_contexts[member_uid]
                        store_message(
                            conf_uid=member_context.character_config.conf_uid,
                            history_uid=member_context.history_uid,
                            role="human",
                            content=msg.content,
                            name=msg.user_name,
                        )
            logger.info(f"Stored {len(state.batch_messages)} messages in history for all group members")
        except Exception as e:
            logger.error(f"Error storing messages in history: {e}")
        
        # Create batch input for AI
        try:
            batch_input = create_batch_input(
                input_text=formatted_input,
                images=images,
                from_name="Human"
            )
            logger.info("Batch input created successfully")
        except Exception as e:
            logger.error(f"Error creating batch input: {e}")
            raise
        
        # Process AI response with group TTS manager
        responding_ws_send = client_connections[responding_ai_uid].send_text
        logger.info(f"Starting AI response processing for {responding_ai_uid}")
        
        full_response = await process_ai_response(
            context=responding_context,
            batch_input=batch_input,
            websocket_send=responding_ws_send,
            tts_manager=tts_manager,
        )
        
        logger.info(f"AI response received: {full_response[:100]}..." if len(full_response) > 100 else f"AI response received: {full_response}")
        
        # Handle TTS completion
        if tts_manager.task_list:
            logger.info(f"Processing {len(tts_manager.task_list)} TTS tasks")
            await asyncio.gather(*tts_manager.task_list)
            # Broadcast backend-synth-complete to all group members
            await broadcast_func(
                members_list,
                {"type": "backend-synth-complete"},
            )

            broadcast_ctx = BroadcastContext(
                broadcast_func=broadcast_func,
                group_members=members_list,
                current_client_uid=responding_ai_uid,
            )

            await finalize_conversation_turn(
                tts_manager=tts_manager,
                websocket_send=responding_ws_send,
                client_uid=responding_ai_uid,
                broadcast_ctx=broadcast_ctx,
            )
        else:
            logger.info("No TTS tasks to process")
            # Send conversation end signal even if no TTS
            await broadcast_func(
                members_list,
                {"type": "control", "text": "conversation-chain-end"},
            )
        
        # Store AI response in history for all group members
        if full_response:
            try:
                for member_uid in members_list:
                    if member_uid in client_contexts:
                        member_context = client_contexts[member_uid]
                        store_message(
                            conf_uid=member_context.character_config.conf_uid,
                            history_uid=member_context.history_uid,
                            role="ai",
                            content=full_response,
                            name=responding_context.character_config.character_name,
                            avatar=responding_context.character_config.avatar,
                        )
                logger.info("AI response stored in history for all group members")
            except Exception as e:
                logger.error(f"Error storing AI response in history: {e}")
        else:
            logger.warning("No AI response to store")
        
        # Clear processed messages and reset state
        state.clear_batch_messages()
        state.is_processing_batch = False
        
        logger.info(f"Batch Group Conversation {session_emoji} completed successfully!")
        
    except asyncio.CancelledError:
        logger.info(f"🤡👍 Batch Group Conversation {session_emoji} cancelled because interrupted.")
        # Send end signal on cancellation
        try:
            await broadcast_func(
                members_list,
                {"type": "control", "text": "conversation-chain-end"},
            )
        except Exception as e:
            logger.error(f"Error sending end signal on cancellation: {e}")
        raise
    except Exception as e:
        logger.error(f"Error in batch group conversation: {e}", exc_info=True)
        try:
            await broadcast_func(
                members_list,
                {
                    "type": "error",
                    "message": f"Error in batch conversation: {str(e)}",
                },
            )
            # Send end signal on error
            await broadcast_func(
                members_list,
                {"type": "control", "text": "conversation-chain-end"},
            )
        except Exception as broadcast_error:
            logger.error(f"Error broadcasting error message: {broadcast_error}")
        raise
    finally:
        # Cleanup
        try:
            cleanup_conversation(tts_manager, session_emoji)
            logger.info("TTS manager cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up TTS manager: {e}")
        
        # Reset processing state
        if state:
            state.is_processing_batch = False
            logger.info("Processing state reset")


async def process_group_agent_output(
    output: Any,
    context: ServiceContext,
    tts_manager: GroupTTSTaskManager,
    responding_ws_send: WebSocketSend,
) -> str:
    """Process agent output for group conversations with broadcasting support"""
    from ..agent.output_types import SentenceOutput, AudioOutput
    
    # Set character info
    output.display_text.name = context.character_config.character_name
    output.display_text.avatar = context.character_config.avatar

    full_response = ""
    try:
        if isinstance(output, SentenceOutput):
            # Handle SentenceOutput type
            async for display_text, tts_text, actions in output:
                logger.debug(f"🏃 Processing group sentence output: '''{tts_text}'''...")

                # Apply translation if needed
                if context.translate_engine:
                    if len(re.sub(r'[\s.,!?，。！？\'"』」）】\s]+', "", tts_text)):
                        tts_text = context.translate_engine.translate(tts_text)
                    logger.info(f"🏃 Text after translation: '''{tts_text}'''...")

                full_response += display_text.text
                
                # Use the group TTS manager which will broadcast to all members
                await tts_manager.speak(
                    tts_text=tts_text,
                    display_text=display_text,
                    actions=actions,
                    live2d_model=context.live2d_model,
                    tts_engine=context.tts_engine,
                    websocket_send=responding_ws_send,  # This is used for fallback/logging
                )
                
        elif isinstance(output, AudioOutput):
            # Handle AudioOutput type
            async for audio_path, display_text, transcript, actions in output:
                logger.debug(f"🏃 Processing group audio output: '''{transcript}'''...")
                full_response += transcript
                
                # For AudioOutput, we need to broadcast the prepared payload directly
                from ..utils.stream_audio import prepare_audio_payload
                audio_payload = prepare_audio_payload(
                    audio_path=audio_path,
                    display_text=display_text,
                    actions=actions.to_dict() if actions else None,
                )
                # Broadcast to all group members using the TTS manager's broadcast method
                await tts_manager._broadcast_audio_payload(audio_payload)
        else:
            logger.warning(f"Unknown output type in group conversation: {type(output)}")

    except Exception as e:
        logger.error(f"Error processing group agent output: {e}", exc_info=True)
        # Send error to responding AI only (not broadcast)
        try:
            await responding_ws_send(json.dumps({
                "type": "error",
                "message": f"Error processing response: {str(e)}"
            }))
        except Exception as ws_error:
            logger.error(f"Error sending websocket error message: {ws_error}")
        raise

    return full_response


async def process_ai_response(
    context: ServiceContext,
    batch_input: Any,
    websocket_send: WebSocketSend,
    tts_manager: TTSTaskManager,
) -> str:
    """Process AI response for batch input"""
    full_response = ""

    try:
        logger.info("Starting agent engine chat")
        agent_output = context.agent_engine.chat(batch_input)
        logger.info("Agent engine chat initiated successfully")

        # Check if we're using GroupTTSTaskManager for group conversations
        if isinstance(tts_manager, GroupTTSTaskManager):
            async for output in agent_output:
                logger.debug(f"Processing group agent output: {output}")
                response_part = await process_group_agent_output(
                    output=output,
                    context=context,
                    tts_manager=tts_manager,
                    responding_ws_send=websocket_send,
                )
                full_response += response_part
                logger.debug(f"Group response part processed: {response_part}")
        else:
            # Use regular processing for individual conversations
            async for output in agent_output:
                logger.debug(f"Processing agent output: {output}")
                response_part = await process_agent_output(
                    output=output,
                    character_config=context.character_config,
                    live2d_model=context.live2d_model,
                    tts_engine=context.tts_engine,
                    websocket_send=websocket_send,
                    tts_manager=tts_manager,
                    translate_engine=context.translate_engine,
                )
                full_response += response_part
                logger.debug(f"Response part processed: {response_part}")

        logger.info(f"AI response processing completed. Total response length: {len(full_response)}")

    except Exception as e:
        logger.error(f"Error processing AI response: {e}", exc_info=True)
        # Send error message to websocket
        try:
            await websocket_send(json.dumps({
                "type": "error",
                "message": f"Error processing AI response: {str(e)}"
            }))
        except Exception as ws_error:
            logger.error(f"Error sending websocket error message: {ws_error}")
        raise

    return full_response


async def process_group_conversation(
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    initiator_client_uid: str,
    user_input: Union[str, np.ndarray],
    images: Optional[List[Dict[str, Any]]] = None,
    session_emoji: str = np.random.choice(EMOJI_LIST),
) -> None:
    """Process group conversation

    Args:
        client_contexts: Dictionary of client contexts
        client_connections: Dictionary of client WebSocket connections
        broadcast_func: Function to broadcast messages to group
        group_members: List of group member UIDs
        initiator_client_uid: UID of conversation initiator
        user_input: Text or audio input from user
        images: Optional list of image data
        session_emoji: Emoji identifier for the conversation
    """
    # Create TTSTaskManager for each member
    tts_managers = {uid: TTSTaskManager() for uid in group_members}

    try:
        logger.info(f"Group Conversation Chain {session_emoji} started!")

        # Initialize state with group_id
        state = GroupConversationState(
            group_id=f"group_{initiator_client_uid}",  # Use same format as chat_group
            session_emoji=session_emoji,
            group_queue=list(group_members),
            memory_index={
                uid: 0 for uid in group_members
            },  # Initialize memory index for each member
        )

        # Initialize group conversation context for each AI
        init_group_conversation_contexts(client_contexts)

        # Get human name from initiator context
        initiator_context = client_contexts.get(initiator_client_uid)
        human_name = (
            initiator_context.character_config.human_name
            if initiator_context
            else "Human"
        )

        # Process initial input
        input_text = await process_group_input(
            user_input=user_input,
            initiator_context=initiator_context,
            initiator_ws_send=client_connections[initiator_client_uid].send_text,
            broadcast_func=broadcast_func,
            group_members=group_members,
            initiator_client_uid=initiator_client_uid,
        )

        for member_uid in group_members:
            member_context = client_contexts[member_uid]
            store_message(
                conf_uid=member_context.character_config.conf_uid,
                history_uid=member_context.history_uid,
                role="human",
                content=input_text,
                name=human_name,
            )

        state.conversation_history = [f"{human_name}: {input_text}"]

        # Main conversation loop
        while state.group_queue:
            try:
                current_member_uid = state.group_queue.pop(0)
                await handle_group_member_turn(
                    current_member_uid=current_member_uid,
                    state=state,
                    client_contexts=client_contexts,
                    client_connections=client_connections,
                    broadcast_func=broadcast_func,
                    group_members=group_members,
                    images=images,
                    tts_manager=tts_managers[current_member_uid],
                )
            except Exception as e:
                logger.error(f"Error in group member turn: {e}")
                await handle_member_error(
                    broadcast_func, group_members, f"Error in conversation: {str(e)}"
                )

    except asyncio.CancelledError:
        logger.info(
            f"🤡👍 Group Conversation {session_emoji} cancelled because interrupted."
        )
        raise
    except Exception as e:
        logger.error(f"Error in group conversation chain: {e}")
        await handle_member_error(
            broadcast_func, group_members, f"Fatal error in conversation: {str(e)}"
        )
        raise
    finally:
        # Cleanup all TTS managers
        for uid, tts_manager in tts_managers.items():
            cleanup_conversation(tts_manager, session_emoji)
        # Clean up
        GroupConversationState.remove_state(state.group_id)


def init_group_conversation_state(
    group_members: List[str], session_emoji: str
) -> GroupConversationState:
    """Initialize group conversation state"""
    return GroupConversationState(
        conversation_history=[],
        memory_index={uid: 0 for uid in group_members},
        group_queue=list(group_members),
        session_emoji=session_emoji,
    )


def init_group_conversation_contexts(
    client_contexts: Dict[str, ServiceContext],
) -> None:
    """Initialize group conversation context for each AI participant"""
    ai_names = [ctx.character_config.character_name for ctx in client_contexts.values()]

    for context in client_contexts.values():
        agent = context.agent_engine
        if hasattr(agent, "start_group_conversation"):
            agent.start_group_conversation(
                human_name="Human",
                ai_participants=[
                    name
                    for name in ai_names
                    if name != context.character_config.character_name
                ],
            )
            logger.debug(
                f"Initialized group conversation context for "
                f"{context.character_config.character_name}"
            )


async def process_group_input(
    user_input: Union[str, np.ndarray],
    initiator_context: ServiceContext,
    initiator_ws_send: WebSocketSend,
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    initiator_client_uid: str,
) -> str:
    """Process and broadcast user input to group"""
    input_text = await process_user_input(
        user_input, initiator_context.asr_engine, initiator_ws_send
    )
    await broadcast_transcription(
        broadcast_func, group_members, input_text, initiator_client_uid
    )
    return input_text


async def broadcast_transcription(
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    text: str,
    exclude_uid: str,
) -> None:
    """Broadcast transcription to group members"""
    await broadcast_func(
        group_members,
        {
            "type": "user-input-transcription",
            "text": text,
        },
        exclude_uid,
    )


async def handle_group_member_turn(
    current_member_uid: str,
    state: GroupConversationState,
    client_contexts: Dict[str, ServiceContext],
    client_connections: Dict[str, WebSocket],
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    images: Optional[List[Dict[str, Any]]],
    tts_manager: TTSTaskManager,
) -> None:
    """Handle a single group member's conversation turn"""
    # Update current speaker before processing
    state.current_speaker_uid = current_member_uid

    await broadcast_thinking_state(broadcast_func, group_members)

    context = client_contexts[current_member_uid]
    current_ws_send = client_connections[current_member_uid].send_text

    new_messages = state.conversation_history[state.memory_index[current_member_uid] :]
    new_context = "\n".join(new_messages) if new_messages else ""

    batch_input = create_batch_input(
        input_text=new_context, images=images, from_name="Human"
    )

    logger.info(
        f"AI {context.character_config.character_name} "
        f"(client {current_member_uid}) receiving context:\n{new_context}"
    )

    full_response = await process_member_response(
        context=context,
        batch_input=batch_input,
        current_ws_send=current_ws_send,
        tts_manager=tts_manager,
    )

    if tts_manager.task_list:
        await asyncio.gather(*tts_manager.task_list)
        await current_ws_send(json.dumps({"type": "backend-synth-complete"}))

        broadcast_ctx = BroadcastContext(
            broadcast_func=broadcast_func,
            group_members=group_members,
            current_client_uid=current_member_uid,
        )

        await finalize_conversation_turn(
            tts_manager=tts_manager,
            websocket_send=current_ws_send,
            client_uid=current_member_uid,
            broadcast_ctx=broadcast_ctx,
        )

    if full_response:
        ai_message = f"{context.character_config.character_name}: {full_response}"
        state.conversation_history.append(ai_message)
        logger.info(f"Appended complete response: {ai_message}")

        for member_uid in group_members:
            member_context = client_contexts[member_uid]
            store_message(
                conf_uid=member_context.character_config.conf_uid,
                history_uid=member_context.history_uid,
                role="ai",
                content=full_response,
                name=context.character_config.character_name,
                avatar=context.character_config.avatar,
            )

    state.memory_index[current_member_uid] = len(state.conversation_history)
    state.group_queue.append(current_member_uid)

    # Clear speaker after turn completes
    state.current_speaker_uid = None


async def broadcast_thinking_state(
    broadcast_func: BroadcastFunc, group_members: List[str]
) -> None:
    """Broadcast thinking state to group"""
    await broadcast_func(
        group_members,
        {"type": "control", "text": "conversation-chain-start"},
    )
    await broadcast_func(
        group_members,
        {"type": "full-text", "text": "Thinking..."},
    )


async def handle_member_error(
    broadcast_func: BroadcastFunc,
    group_members: List[str],
    error_message: str,
) -> None:
    """Handle and broadcast member error"""
    await broadcast_func(
        group_members,
        {
            "type": "error",
            "message": error_message,
        },
    )


async def process_member_response(
    context: ServiceContext,
    batch_input: Any,
    current_ws_send: WebSocketSend,
    tts_manager: TTSTaskManager,
) -> str:
    """Process group member's response"""
    full_response = ""

    try:
        agent_output = context.agent_engine.chat(batch_input)

        async for output in agent_output:
            response_part = await process_agent_output(
                output=output,
                character_config=context.character_config,
                live2d_model=context.live2d_model,
                tts_engine=context.tts_engine,
                websocket_send=current_ws_send,
                tts_manager=tts_manager,
                translate_engine=context.translate_engine,
            )
            full_response += response_part

    except Exception as e:
        logger.error(f"Error processing member response: {e}")
        raise

    return full_response
