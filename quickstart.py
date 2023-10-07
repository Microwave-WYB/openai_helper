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
print(
    chat.send_messages([message], temperature=0, max_tokens=500)
    .choices[0]
    .message.content
)
