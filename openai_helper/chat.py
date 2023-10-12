import time
import openai, json
from typing import Dict, Union, List, Tuple
from .function_call import OpenAIFunctionCall


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
            args["functions"] = [
                f["info"] for f in self.functions.functions.values()
            ]

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
            "content": function_output,
        }
