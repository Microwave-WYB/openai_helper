"""
Demonstrates how to use the openai_helper package to create a chatbot that can generate
random numbers.
"""
from random import randint
from openai_helper import OpenAIFunctionCall, ChatSession

functions = OpenAIFunctionCall()


@functions.register
def random_number(minimum: int, maximum: int) -> int:
    """
    Generate a random number from minimum to maximum.

    Args:
        minimum (int): The minimum bound for the random number.
        maximum (int): The maximum bound for the random number.

    Returns:
        int: A random number between minimum and maximum.
    """
    return randint(minimum, maximum)


if __name__ == "__main__":
    message = {
        "role": "user",
        "content": "Generate a random number between 1 and 10000",
    }
    chat = ChatSession(functions, model="gpt-4", verbose=True)

    response, function_call_info = chat.send_messages(
        [message], temperature=0, max_tokens=500
    )

    print(response.choices[0].message.content)

    if function_call_info:
        function_output = chat.handle_function(function_call_info, verbose=True)

    response, _ = chat.send_messages(
        [message, function_output], temperature=0, max_tokens=500
    )
    print(response.choices[0].message.content)
