import time
import openai
import json
import tiktoken
from typing import Dict, Union, List, Tuple, Literal
from .function_call import OpenAIFunctionCall

_ASSISTANT_PROMPT = "Assistant:\n    {content}"
_USER_PROMPT = "User:\n    "


def count_token(input: str) -> int:
    """
    Count the number of tokens in a string.

    Args:
        input (str): Input string.

    Returns:
        int: Number of tokens.
    """
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(input))


CompactingMethod = Literal["fifo", "summarize"]


class HistoryManager:
    def __init__(
        self,
        token_threshold: int = 3000,
        max_tokens: int = 8000,
        compacting_method: CompactingMethod = "fifo",
        keep_top: int = 1,
        keep_bottom: int = 6,
        messages: List[Dict[str, str]] = [],
        verbose: bool = False,
    ) -> None:
        """
        Initialize a HistoryManager object.

        Args:
            token_threshold (int, optional): Maximum token before compacting. Defaults to 3000.
            max_tokens (int, optional): Maximum token in the history. Defaults to 8000.
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
        assert token_threshold >= 0, "token_threshold must be greater than or equal to 0"
        assert token_threshold <= max_tokens, "token_threshold must be less than max_tokens"
        self.token_threadhold = token_threshold
        self.max_tokens = max_tokens
        self.compacting_method = compacting_method
        self.keep_top = keep_top
        self.keep_bottom = keep_bottom
        self.verbose = verbose
        self.messages = messages

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

        if self.verbose:
            print(f"Compacted messages. Current token count: {self.get_total_tokens()}")

    def add(self, message: Dict[str, str]) -> None:
        """
        Add a message to the message history.

        Args:
            message (Dict[str, str]): Message to add.
        """
        assert message.keys() == {"role", "content"}, "Message must have role and content keys"
        assert message["role"] in ["system", "user", "assistant", "function"], "Message role must be one of 'system', 'user', 'assistant', or 'function'"
        assert count_token(message["content"]) <= self.max_tokens, "Message exceeds maximum token count"
        self.messages.append(message)
        self.compact()


class ChatSession:
    def __init__(
        self,
        functions: OpenAIFunctionCall = None,
        model: str = "gpt-3.5-turbo",
        verbose: bool = False,
    ) -> None:
        self.functions = functions
        self.model = model
        self.verbose = verbose

    def send_messages(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[Dict, Union[None, Dict]]:
        """
        Send messages using OpenAI API. Allow all OpenAI API keyword args.

        Args:
            messages (List[Dict[str, str]]): List of messages to send

        Returns:
            Tuple[Dict, Union[None, Dict]]: response, function_call_info
        """
        args = {
            "model": self.model,
            "messages": messages,
        }

        if self.functions and self.functions.functions:
            args["functions"] = [f["info"] for f in self.functions.functions.values()]

        try:
            response = openai.ChatCompletion.create(
                **args,
                **kwargs,
            )
        except openai.error.RateLimitError:
            print("Rate limit exceeded, waiting 3 seconds...")
            time.sleep(3)
            return self.send_messages(messages, **kwargs)

        # Extract the function call if present
        function_call_info = None
        assistant_response = response["choices"][0]["message"]
        if "function_call" in assistant_response:
            function_call_info = {
                "name": assistant_response["function_call"]["name"],
                "arguments": json.loads(
                    assistant_response["function_call"]["arguments"]
                ),
            }

        return response, function_call_info

    def handle_function(
        self,
        function_call: Dict,
        verbose: bool = False,
    ) -> Dict:
        """
        Handle a function call from the OpenAI API.

        Args:
            function_call (Dict): Function call info from the OpenAI API
            verbose (bool, optional): Whether to print debug info. Defaults to False.

        Returns:
            Dict: A dictionary containing the function's role, name, and content, which includes both the input arguments and the output.
        """
        function_name = function_call["name"]
        function_args = function_call["arguments"]
        function_output = self.functions.call(function_name, **function_args)

        if verbose:
            print(f"Function call: {function_call}")
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
        self, messages: List[Dict[str, str]] = [], no_confirm: bool = False
    ) -> None:
        """
        Start a chat session in the terminal.

        Args:
            messages (List[Dict[str, str]], optional): Pre-existing messages if there are any. Defaults to [].
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
                messages.append({"role": "user", "content": user_message})
                response, function_call_info = self.send_messages(messages)

                # Print out the response content
                assistant_message = response["choices"][0]["message"]["content"]
                if assistant_message is not None:
                    print(_ASSISTANT_PROMPT.format(content=assistant_message))

                # Continuously handle function calls until there are none
                while function_call_info:
                    print(f"Calling function: {function_call_info['name']}")
                    print("Arguments:")
                    for arg, val in function_call_info["arguments"].items():
                        print(f"    {arg}: {val}")

                    confirmation = None
                    if no_confirm:
                        confirmation = "y"
                    while confirmation not in ["y", "n", ""] and not no_confirm:
                        confirmation = input("Confirm function call? [Y/n]: ").lower()

                    if confirmation in ["y", ""]:
                        function_response = self.handle_function(
                            function_call_info, self.verbose
                        )
                        messages.append(function_response)

                        # Send the updated messages back to the model
                        response, function_call_info = self.send_messages(messages)

                        # Print out the response content
                        follow_up_message = response["choices"][0]["message"]["content"]
                        if follow_up_message is not None:
                            print(_ASSISTANT_PROMPT.format(content=follow_up_message))

                    else:
                        print("Function call skipped.")
                        break  # If user skips the function call, break from the function handling loop

            except EOFError:
                print("Ending chat session.")
                break
            except KeyboardInterrupt:
                print("Ending chat session.")
                break
