"""Description: This file contains the implementation of the `AsyncLLM` class.
This class is responsible for handling asynchronous interaction with OpenAI API compatible
endpoints for language generation.
"""

from typing import AsyncIterator, List, Dict, Any
from openai import (
    AsyncStream,
    AsyncOpenAI,
    APIError,
    APIConnectionError,
    RateLimitError,
)
from openai.types.chat import ChatCompletionChunk
from loguru import logger
import json
import ast
from .stateless_llm_interface import StatelessLLMInterface
import requests

class AsyncLLM(StatelessLLMInterface):
    def __init__(
        self,
        model: str,
        base_url: str,
        llm_api_key: str = "z",
        organization_id: str = "z",
        project_id: str = "z",
        temperature: float = 1.0,
    ):
        """
        Initializes an instance of the `AsyncLLM` class.

        Parameters:
        - model (str): The model to be used for language generation.
        - base_url (str): The base URL for the OpenAI API.
        - organization_id (str, optional): The organization ID for the OpenAI API. Defaults to "z".
        - project_id (str, optional): The project ID for the OpenAI API. Defaults to "z".
        - llm_api_key (str, optional): The API key for the OpenAI API. Defaults to "z".
        - temperature (float, optional): What sampling temperature to use, between 0 and 2. Defaults to 1.0.
        """
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.client = AsyncOpenAI(
            base_url=base_url,
            organization=organization_id,
            project=project_id,
            api_key=llm_api_key,
        )

        logger.info(
            f"Initialized AsyncLLM with the parameters: {self.base_url}, {self.model}"
        )
    async def generate_song(self,args):
        url = "https://apibox.erweima.ai/api/v1/generate"
        # url = "https://apibox.erweima.ai/api/v1/lyrics"

        payload = json.dumps({
        # "prompt": "A calm and relaxing piano track with soft melodies",
        "prompt": args['lyrics'],
        "style": args['mood'],
        "title": args['topic'],
        "customMode": True,
        "instrumental": False,
        "model": "V4",
        # "negativeTags": "Heavy Metal, Upbeat Drums",
        "callBackUrl": "https://2871-219-70-65-54.ngrok-free.app/callback"
        },ensure_ascii=False)
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Authorization': 'Bearer 073a4ad877509f5a847ff7dcb1d8b8a8'
        }
        response = requests.request("POST", url, headers=headers, data=payload)
        print(response)
        return {
            "lyrics": args['lyrics'],
            "topic": args['topic'],
            "mood": args['mood'],
        }

    async def chat_completion(
        self, messages: List[Dict[str, Any]], system: str = None
    ) -> AsyncIterator[str]:
        """
        Generates a chat completion using the OpenAI API asynchronously.

        Parameters:
        - messages (List[Dict[str, Any]]): The list of messages to send to the API.
        - system (str, optional): System prompt to use for this completion.

        Yields:
        - str: The content of each chunk from the API response.

        Raises:
        - APIConnectionError: When the server cannot be reached
        - RateLimitError: When a 429 status code is received
        - APIError: For other API-related errors
        """
        logger.debug(f"Messages: {messages}")
        stream = None
        try:
            # If system prompt is provided, add it to the messages
            messages_with_system = messages
            if system:
                messages_with_system = [
                    # {"role": "system", "content": system},
                    *messages,
                ]
            messages_with_system = [
                d for i, d in enumerate(messages)
                if d["role"] != "system" or i == 0
            ]
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "generate_song",
                        "description": "生成有趣的歌曲",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "topic": {
                                    "type": "string",
                                    "description": '歌曲主題',
                                },
                                "mood": {
                                    "type": "string",
                                    "description": "歌曲風格",
                                },
                                "lyrics": {
                                    "type": "string",
                                    "description": "歌詞，而歌詞長度要超過100字",
                                },
                            },
                            "required": ["topic","mood","lyrics"],
                        },
                    },
                },
            ]
            stream: AsyncStream[
                ChatCompletionChunk
            ] = await self.client.chat.completions.create(
                messages=messages_with_system,
                model=self.model,
                stream=True,
                temperature=self.temperature,
                tools=tools,
                # tool_choice="none"
                tool_choice="auto"
                # tool_choice={
                #     "type": "function",
                #     "function": {
                #         "name": "generate_song"
                #     }
                # }
            )
            tool_call_args = ""
            collecting_tool_args = False
            tool_call_id = None
            function_name = None

            async for chunk in stream:
                delta = chunk.choices[0].delta

                # 處理 function call 的 arguments
                if delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function.name == "generate_song":
                            collecting_tool_args = True
                            tool_call_id = tool_call.id
                            function_name = tool_call.function.name
                            if tool_call.function.arguments:
                                tool_call_args += tool_call.function.arguments

                elif collecting_tool_args and delta.tool_calls:
                    for tool_call in delta.tool_calls:
                        if tool_call.function.arguments:
                            tool_call_args += tool_call.function.arguments

                elif delta.content is not None:
                    yield delta.content

            # 嘗試解析完整的 arguments JSON
            if collecting_tool_args:
                try:
                    args = ast.literal_eval(tool_call_args)
                    logger.info(f"🔧 Tool Call Args: {args}")
                    song = await self.generate_song(args)
                    yield "\n🎵 歌曲生成完成:\n"
                    yield f"主題: {song['topic']}\n風格: {song['mood']}\n歌詞:\n{song['lyrics']}"
                except json.JSONDecodeError as e:
                    logger.error(f"JSON 解析錯誤: {e}")
                    yield f"\n[錯誤] 無法解析 function call 的參數: {e}"

        except APIConnectionError as e:
            logger.error(
                f"Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. \nCheck the configurations and the reachability of the LLM backend. \nSee the logs for details. \nTroubleshooting with documentation: https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E \n{e.__cause__}"
            )
            yield "Error calling the chat endpoint: Connection error. Failed to connect to the LLM API. Check the configurations and the reachability of the LLM backend. See the logs for details. Troubleshooting with documentation: [https://open-llm-vtuber.github.io/docs/faq#%E9%81%87%E5%88%B0-error-calling-the-chat-endpoint-%E9%94%99%E8%AF%AF%E6%80%8E%E4%B9%88%E5%8A%9E]"

        except RateLimitError as e:
            logger.error(
                f"Error calling the chat endpoint: Rate limit exceeded: {e.response}"
            )
            yield "Error calling the chat endpoint: Rate limit exceeded. Please try again later. See the logs for details."

        except APIError as e:
            logger.error(f"LLM API: Error occurred: {e}")
            logger.info(f"Base URL: {self.base_url}")
            logger.info(f"Model: {self.model}")
            logger.info(f"Messages: {messages}")
            logger.info(f"temperature: {self.temperature}")
            yield "Error calling the chat endpoint: Error occurred while generating response. See the logs for details."

        finally:
            # make sure the stream is properly closed
            # so when interrupted, no more tokens will being generated.
            if stream:
                logger.debug("Chat completion finished.")
                await stream.close()
                logger.debug("Stream closed.")
