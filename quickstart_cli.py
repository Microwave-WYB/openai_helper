from openai_helper import OpenAIFunctionCall, ChatSession, HistoryManager
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


system_prompt = """
You are a random number generator assistant. You will help your user to generate random numbers.
If the user did not provide a range, you need to ask the user for a range.
"""
message = {"role": "system", "content": system_prompt}
history_manager = HistoryManager(3000, 8000, "fifo", 1, 4, [message], True)
chat = ChatSession(functions, model="gpt-4", verbose=True)
chat.start(no_confirm=True, history_manager=history_manager)
