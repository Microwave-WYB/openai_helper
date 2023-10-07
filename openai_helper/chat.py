import time
import openai, json
from typing import Dict, Union, List
from .function_call import OpenAIFunctionCall


class ChatSession:
    def __init__(
        self,
        functions: OpenAIFunctionCall = None,
        model: str = "gpt-3.5-turbo",
        verbose: bool = False,
    ) -> None:
        self.function_call = functions
        self.model = model
        self.verbose = verbose

    def send_messages(
        self,
        messages: List[Dict[str, str]],
        handle_function_call: bool = True,
        **kwargs,
    ) -> Dict:
        # create a dictionary of arguments
        args = {
            "model": self.model,
            "messages": messages,
        }

        # add functions parameter only if self.function_call.functions contains any functions
        if self.function_call.functions:
            args["functions"] = [
                f["info"] for f in self.function_call.functions.values()
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

        if handle_function_call and self.function_call:
            response = self.handle_function(response, messages, **kwargs)

        return response

    def handle_function(
        self, response: Dict, messages: List[Dict[str, str]], **kwargs
    ) -> Dict:
        assistant_response = response["choices"][0]["message"]

        if assistant_response.get("function_call"):
            function_name = assistant_response["function_call"]["name"]
            function_args = json.loads(assistant_response["function_call"]["arguments"])
            function_output = self.function_call.call(function_name, **function_args)

            # If verbose is True, print the function call response and output
            if self.verbose:
                print(f"Function call response: {assistant_response}")
                print(f"Function output: {function_output}")

            # Add the previous message and function response to the messages list
            messages.append(assistant_response)
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_output,
                }
            )

            # Recall the API with the function output
            try:
                response = openai.ChatCompletion.create(
                    model=self.model, messages=messages, **kwargs
                )
            except openai.error.RateLimitError:
                print("Rate limit exceeded, waiting 3 seconds...")
                time.sleep(3)
                return self.send_messages(messages, **kwargs)

        return response
