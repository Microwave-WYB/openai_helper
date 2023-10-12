from openai_helper import OpenAIFunctionCall, ChatSession
from random import randint

functions = OpenAIFunctionCall()


@functions.register
def random_number(min: int, max: int) -> int:
    """
    Generate a random number from min to max.

    Args:
        min (int): The minimum bound for the random number.
        max (int): The maximum bound for the random number.

    Returns:
        int: A random number between min and max.
    """
    return randint(min, max)


message = {"role": "user", "content": "Generate a random number between 1 and 10000"}
chat = ChatSession(functions, model="gpt-4", verbose=True)

response, function_call_info = chat.send_messages([message], temperature=0, max_tokens=500)

print(response.choices[0].message.content)

if function_call_info:
    function_output = chat.handle_function(function_call_info, verbose=True)

response, _ = chat.send_messages([message, function_output], temperature=0, max_tokens=500)
print(response.choices[0].message.content)