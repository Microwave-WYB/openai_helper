"""
This module provides classes and functions to manage a chat session with the OpenAI API.
"""
import json
from typing import Dict, List, Literal

import openai
from openai.types.chat.chat_completion_message import (
    ChatCompletionMessage,
    FunctionCall,
)
import tiktoken

from .function_call import FunctionCallManager

_ASSISTANT_PROMPT = "Assistant:\n    {content}"
_USER_PROMPT = "User:\n    "


def count_token(text: str) -> int:
    """
    Count the number of tokens in a string.

    Args:
        input (str): Input string.

    Returns:
        int: Number of tokens.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


CompactingMethod = Literal["fifo", "summarize"]


class HistoryManager:
    """
    Class to manage the message history.
    """

    def __init__(
        self,
        token_threshold: int = 2000,
        max_tokens: int = 4000,
        compacting_method: CompactingMethod = "fifo",
        keep_top: int = 1,
        keep_bottom: int = 6,
        messages: List[Dict[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initialize a HistoryManager object.

        Args:
            token_threshold (int, optional): Maximum token before compacting. Defaults to 2000.
            max_tokens (int, optional): Maximum token in the history. Defaults to 4000.
            compacting_method (CompactingMethod, optional): How to compact the history. Defaults to "fifo".
            keep_top (int, optional): Top messages to keep. Defaults to 1.
            keep_bottom (int, optional): Bottom message to keep. Set to 0 to use token_threshold. Defaults to 6.
            messages (List[Dict[str, str]], optional): Pre-existing messages if there are any. Defaults to [].
            verbose (bool, optional): Defaults to False.
        """
        assert compacting_method in [
            "fifo",
            "summarize",
        ], "Compacting method must be either 'fifo' or 'summarize'"
        assert keep_top >= 0, "keep_top must be greater than or equal to 0"
        assert keep_bottom >= 0, "keep_bottom must be greater than or equal to 0"
        assert (
            token_threshold >= 0
        ), "token_threshold must be greater than or equal to 0"
        assert (
            token_threshold <= max_tokens
        ), "token_threshold must be less than max_tokens"
        self.token_threshold = token_threshold
        self.max_tokens = max_tokens
        self.compacting_method = compacting_method
        self.keep_top = keep_top
        self.keep_bottom = keep_bottom
        self.verbose = verbose
        self.messages = messages
        self.all_messages = messages

    def get_total_tokens(self) -> int:
        """
        Get the total number of tokens in the message history.

        Returns:
            int: Total number of tokens.
        """
        return sum(count_token(msg["content"]) for msg in self.messages)

    def compact(self) -> None:
        """
        Compact the message history using the specified compacting method.
        """
        if self.verbose:
            print(f"Compacted messages. Current token count: {self.get_total_tokens()}")

        if self.compacting_method == "fifo":
            total_tokens = self.get_total_tokens()
            if total_tokens <= self.token_threshold:
                # Not enough tokens to require compacting
                if len(self.messages) > self.keep_top + self.keep_bottom:
                    top_msgs = self.messages[: self.keep_top]
                    bottom_msgs = self.messages[-self.keep_bottom :]
                    self.messages = top_msgs + [
                        msg for msg in bottom_msgs if msg not in top_msgs
                    ]
            else:
                while (
                    self.get_total_tokens() > self.token_threshold
                    and len(self.messages) > self.keep_top
                ):
                    # Try to keep keep_top messages
                    self.messages.pop(self.keep_top)

                # If keep_top messages themselves exceed threshold, retain oldest messages
                if self.get_total_tokens() > self.token_threshold:
                    self.messages = self.messages[: self.keep_top]
        elif self.compacting_method == "summarize":
            raise NotImplementedError(
                "Summarization compacting method not implemented yet."
            )
        else:
            raise ValueError(f"Unknown compacting method: {self.compacting_method}")

    def add(self, message: Dict[str, str]) -> None:
        """
        Add a message to the message history.

        Args:
            message (Dict[str, str]): Message to add.
        """
        assert "role" in message, "Message must have a role"
        assert "content" in message, "Message must have content"
        assert message["role"] in [
            "system",
            "user",
            "assistant",
            "function",
        ], "Message role must be one of 'system', 'user', 'assistant', or 'function'"
        assert (
            count_token(message["content"]) <= self.max_tokens
        ), "Message exceeds maximum token count"
        if self.messages is None:
            self.messages = []
        if self.all_messages is None:
            self.all_messages = []
        self.all_messages.append(message)
        self.messages.append(message)
        self.compact()


class ChatSession:
    """
    Class to manage a chat session.
    """

    def __init__(
        self,
        functions: FunctionCallManager = None,
        model: str = "gpt-3.5-turbo",
        verbose: bool = False,
    ) -> None:
        """
        Initialize a ChatSession object.

        Args:
            functions (OpenAIFunctionCall, optional): Allowed functions. Defaults to None.
            model (str, optional): Model name. Defaults to "gpt-3.5-turbo".
            verbose (bool, optional): Whether to print debug info. Defaults to False.
        """
        self.functions = functions
        self.model = model
        self.client = openai.OpenAI()
        self.verbose = verbose

    def send_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> ChatCompletionMessage:
        """
        Send messages using OpenAI API. Allow all OpenAI API keyword args.

        Args:
            messages (List[Dict[str, str]]): List of messages to send

        Returns:
            ChatCompletionMessage: Response from OpenAI API.
        """
        args = {
            "model": self.model,
            "messages": messages,
        }

        if self.functions and self.functions.functions:
            args["functions"] = [f["info"] for f in self.functions.functions.values()]

        response = self.client.chat.completions.create(
            **args,
            **kwargs,
        )

        return response

    def handle_function(
        self,
        function_call: FunctionCall,
        verbose: bool = False,
    ) -> Dict:
        """
        Handle a function call from the OpenAI API.

        Args:
            function_call (FunctionCall): Function call object from OpenAI API.
            verbose (bool, optional): Whether to print debug info. Defaults to False.

        Returns:
            Dict: A dictionary containing the function's role, name, and content, which includes both the input arguments and the output.
        """
        function_name = function_call.name
        function_args = json.loads(function_call.arguments)
        function_output = self.functions.call(function_name, **function_args)

        if verbose:
            print(f"Function name: {function_name}")
            print(f"Function arguments:\n{function_args}")
            print(f"Function output: {function_output}")

        # Prepare the content string with input arguments and output
        content = (
            f"Function input:\n{function_args}\nFunction output:\n{function_output}"
        )

        return {
            "role": "function",
            "name": function_name,
            "content": content,
        }

    def start(
        self, history_manager: HistoryManager, no_confirm: bool = False, **kwargs
    ) -> None:
        """
        Start a chat session in the terminal.

        Args:
            history_manager (HistoryManager): History manager to use.
            no_confirm (bool, optional): Whether to skip confirmation for function calls. Defaults to False.
        """
        print("Starting chat session. Type 'exit' to exit.")

        while True:
            try:
                # Get input from the user
                user_message = input(_USER_PROMPT.format(content=""))
                if user_message.lower() == "exit":
                    print("Ending chat session.")
                    break

                # Send the message to the API
                history_manager.add({"role": "user", "content": user_message})
                response = self.send_messages(history_manager.messages, **kwargs)
                function_call = response.choices[0].message.function_call

                # Print out the response content
                assistant_message = response.choices[0].message.content
                if assistant_message is not None:
                    print(_ASSISTANT_PROMPT.format(content=assistant_message))

                # Continuously handle function calls until there are none
                while response.choices[0].finish_reason == "function_call":
                    function_name = function_call.name
                    print(f"Calling function: {function_name}")
                    print(f"Arguments: {function_call.arguments}}}")

                    confirmation = None
                    if no_confirm:
                        confirmation = "y"
                    while confirmation not in ["y", "n", ""] and not no_confirm:
                        confirmation = input("Confirm function call? [Y/n]: ").lower()

                    if confirmation in ["y", ""]:
                        function_response = self.handle_function(
                            function_call, self.verbose
                        )
                        history_manager.add(function_response)

                        # Send the updated messages back to the model
                        response = self.send_messages(
                            history_manager.messages, **kwargs
                        )

                        # Print out the response content
                        follow_up_message = response.choices[0].message.content
                        if follow_up_message is not None:
                            print(_ASSISTANT_PROMPT.format(content=follow_up_message))

                    else:
                        # If user skips the function call, break from the function handling loop
                        print("Function call skipped.")
                        break

            except EOFError:
                print("Ending chat session.")
                break
            except KeyboardInterrupt:
                print("Ending chat session.")
                break
