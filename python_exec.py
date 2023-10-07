from openai_helper import OpenAIFunctionCall, ChatSession
import subprocess
import tempfile
import os

functions = OpenAIFunctionCall()

@functions.register
def python_exec(code: str) -> str:
    """
    Execute Python code and return the output.

    Args:
        code (str): The Python code to execute

    Returns:
        str: The output of the Python code
    """
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as temp:
        
        # Write the code to the file
        temp.write(code.encode())
        temp.close()

        # Execute the file and capture the output
        try:
            output = subprocess.check_output(["python", temp.name])
        except subprocess.CalledProcessError as e:
            output = e.output

        # Delete the temporary file
        os.remove(temp.name)

    return output.decode()

message = {"role": "user", "content": "Generate a random number between 1 and 10000"}
chat = ChatSession(functions, model="gpt-4", verbose=True)
print(
    chat.send_messages([message], temperature=0, max_tokens=500)
    .choices[0]
    .message.content
)
