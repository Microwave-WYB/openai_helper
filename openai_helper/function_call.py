import os
import openai
import json
import asyncio


SYSTEM_PROMPT = """
Given the function information , generate a JSON function object for an OpenAI API call.
Remember to escape quotes in generated JSON strings if necessary.

Example:
Function Name: get_current_weather Docstring: Get the current weather in a given location

Args:
    location (str): The city and state, e.g. San Francisco, CA
    unit (str): The unit of temperature to return, either "celsius" or "fahrenheit"

Returns:
    str: A JSON string containing the current weather in the given location

Example Output:
{
    "name": "get_current_weather",
    "description": "Get the current weather in a given location",
    "parameters": {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    },
}
"""

PROMPT_TEMPLATE = """
Function Name: {function_name}
Docstring: {docstring}
Output:
"""


def generate_function_object(function_name: str, docstring: str, cache: dict) -> dict:
    """
    Generate a JSON function object for an OpenAI API call.

    Args:
        function_name (str): The name of the function
        docstring (str): The docstring of the function
        cache (dict): A cache of previously generated function objects

    Returns:
        dict: The generated function object
    """
    # If the function object is in the cache, return it
    if function_name in cache and docstring in cache[function_name]:
        return cache[function_name][docstring]

    # Otherwise, generate the function object
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }
    user_message = {
        "role": "user",
        "content": PROMPT_TEMPLATE.format(
            function_name=function_name, docstring=docstring
        ),
    }
    response = (
        openai.ChatCompletion.create(
            model="gpt-4",
            messages=[system_message, user_message],
            temperature=0,
        )
        .choices[0]
        .message.content
    ).strip()

    # Store the generated function object in the cache
    function_object = json.loads(response)

    if function_name not in cache:
        cache[function_name] = {}
    cache[function_name][docstring] = function_object

    return function_object


class OpenAIFunctionCall:
    def __init__(self, cache_file: str = "./function_cache.json"):
        self.functions = {}
        self.cache_file = cache_file
        self.cache = {}
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                self.cache = json.load(f)

    def register(self, func: callable) -> callable:
        """
        Register a function to be called by the OpenAI API.

        Args:
            func (callable): The function to register

        Returns:
            callable: The function that was registered
        """
        function_info = generate_function_object(
            func.__name__, func.__doc__ or "", self.cache
        )
        if function_info:
            self.functions[func.__name__] = {"info": function_info, "callable": func}
            self.save_cache()  # Save the cache to a file whenever it's updated
        print(f"Registered function: {func.__name__}")
        return func

    def save_cache(self):
        """
        Save the cache to a file.
        """
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def call(self, function_name, *args, **kwargs) -> str:
        """
        Call the registered function by its name. Returns the result of calling the function as string.

        Args:
            function_name (str): The name of the registered function.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            The result of calling the registered function.
        """
        if function_name not in self.functions:
            raise ValueError(f"Function {function_name} not registered.")
        func = self.functions[function_name]["callable"]
        # return func(*args, **kwargs)
        try:
            return str(func(*args, **kwargs))
        except TypeError as e:
            # function output must be able to be converted to string
            raise TypeError(
                f"Function {function_name} output must be able to be converted to string."
            ) from e
