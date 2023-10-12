import time
import openai
import json
import tiktoken
from typing import Dict, Union, List, Tuple
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
            Dict: _description_
        """
        function_name = function_call["name"]
        function_args = function_call["arguments"]
        function_output = self.functions.call(function_name, **function_args)

        if verbose:
            print(f"Function call: {function_call}")
            print(f"Function output: {function_output}")

        return {
            "role": "function",
            "name": function_name,
            "content": f"function_output:\n{function_output}",
        }

    def start(
        self, messages: List[Dict[str, str]] = [], no_confirm: bool = False
    ) -> None:
        """
        Start a chat session.

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
