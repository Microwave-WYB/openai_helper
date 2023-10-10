# OpenAI Helper 

This is a simple wrapper to the OpenAI API. You can easily define custom functions and connect it to OpenAI's models.

## Usage

To install the library, run the following command in the terminal:

```sh
pip install git+https://github.com/Microwave-WYB/openai_helper.git
```

Let's have a look at the example in [quickstart.py](./quickstart.py):

```python
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
```

This script does the following:
1. Initialize an OpenAIFunctionCall object to store all the functions.
2. Define and register a custom function that generates random numbers.
3. Send the prompt message to OpenAI.

By default, ChatSession will automatically handle function calls. If you don't want function calls to be handled automatically, you can send messages with `handle_function_call=False` keyword argument.
