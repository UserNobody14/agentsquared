import typer
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from phoenix.otel import register

tracer_provider = register(
    project_name="coreapp",  # Default is 'default'
    endpoint="http://localhost:6006/v1/traces",
)

# from langchain.chains import LLMChain
from langchain.schema import StrOutputParser


marimo_info = """
Marimo is a modern notebook python framework, a "Cell" is a requirement and the sample code for a variable named "Purpose" would look like this: "@app.cell
def _(mo):
    purpose = mo.ui.text(
        label="Purpose",
        placeholder="e.g., generate a story, analyze data"
    )
    return (purpose,)

@app.cell
def _(mo, purpose):
    mo.hstack([purpose, mo.md(purpose.value)]
    )
    return"
First, select which variables the user is going to input based on their stated prompt, and then create code similar to the above for the provided variables.
Modify the variables so they match the following instructions:
"""

general_marimo_info_instructions = """


You are an AI Agent that generates AI Agents, using the framework "Marimo" which is a replacement of  Jupyter notebooks, that enables you to easily output the Marimo notebook code directly from an LLM.

Here are Three Important Variables You Need to Hold on To:

[app-prompt]
[app-apikey]
[app-name]

You will be generating several marimo cells which will represent the agent, or part of the agent.
@app.cell
def _():
    import marimo as mo
    return (mo,)

Which is just import marimo as mo wrapped in Marimo Markdown, so it displays in a Cell by itself.

I can press the "Play" button and only that portion of the Python code will be run.

I need you to first just write a Python program, that is working, that makes a query to OpenAI and uses the [app-apikey]
Here's an example of a notebook that knows how to call a deepseek instance:

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "openai==1.60.2",
# ]
# ///

import marimo

__generated_with = "0.10.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"\"\"\"
        # Using DeepSeek

        This example shows how to use [mo.ui.chat](https://docs.marimo.io/api/inputs/chat/?h=mo.ui.chat) to make a chatbot backed by [Deepseek](https://deepseek.com/).
        \"\"\"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r\"\"\"
        <a href="https://api-docs.deepseek.com/" target="_blank" rel="noopener noreferrer">
          <img
            src="https://chat.deepseek.com/deepseek-chat.jpeg"
            alt="Powered by deepseek"
            width="450"
          />
        </a>
        \"\"\"
    ).center()
    return


@app.cell
def _(mo):
    import os

    os_key = os.environ.get("DEEPSEEK_API_KEY")
    input_key = mo.ui.text(label="Deepseek API key", kind="password")
    input_key if not os_key else None
    return input_key, os, os_key


@app.cell
def _(input_key, mo, os_key):
    key = os_key or input_key.value

    mo.stop(
        not key,
        mo.md("Please provide your Deepseek AI API key in the input field."),
    )
    return (key,)


@app.cell
def _(key, mo):
    chatbot = mo.ui.chat(
       mo.ai.llm.openai(
           model="deepseek-reasoner",
           system_message="You are a helpful assistant.",
           api_key=key,
           base_url="https://api.deepseek.com",
       ),
        prompts=[
            "Hello",
            "How are you?",
            "I'm doing great, how about you?",
        ],
    )
    chatbot
    return (chatbot,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("Access the chatbot's historical messages with [chatbot.value](https://docs.marimo.io/api/inputs/chat.html#accessing-chat-history).")
    return


@app.cell
def _(chatbot):
    # chatbot.value is the list of chat messages
    chatbot.value
    return


if __name__ == "__main__":
    app.run()

Here are some condensed instructions for how Marimo Works:

Okay, I'm going to give you lot of the markdown documentation for an application called "Marimo" which is like Jupyter notebooks for Python, but this enables in code markdown to be included in the python file, and marimo is able to parse the markdown, and display it in a web browser inside of Cells, that be run individually, just like Jupyter, but Better!

Here is as much of the documentation as I could find.

I want you to create a markdown instruction list that I can use with the API to get back pure python code, with the markdown code properly added.

Also, please include some very minimal instructions for the AI LLM so it can understand that markdown formatting, thank you:

The final Python script you are going to write is going to build a "OpenAI asking agent" with variable functionality, the only thing I can give you now are the three variables used to create me:

[app-prompt]
[app-apikey]
[app-name]

I need you to include in the python script, to understand what the end user app could be called, and what it's about, then design three questions that the user will put in themselves, and then those three questions will be sent, alongside the prompt for OpenAI again, and OpenAI will respond with it's best response to that prompt with the users input.
Here is an example, of if someone inputs the three variables to you:
[app-prompt] - "Make an agent that sets up my exercise routine"
[app-apikey] - "sk-897t779g977giuh98y79giuy"
[app-name] - "exerbot"
You will be envisioning the prompt based on the above app-name and the app-prompt, and the three inputs which are the questions that will be sent along with the OpenAI query.
Here's an example of a prompt that might come from you in this case:
"you are a weight loss coach, you are going to take in three variables from the user (Age, Weight, and Goal) please return a friendly encouragement, and also a list of steps a person of those parameters and that goal can do to reach their weightloss goal. "
And the three questions might be:
Age: Weight: Goal:
I want you to make a Marimo only python file, using the above example as a starting point, and I want you to ask users a few questions,
Here is the prompt you need to pre-write into the final marimo program:
you are a weight loss coach, you are going to take in three variables from the user (Age, Weight, and Goal) please return a friendly encouragement, and also a list of steps a person of those parameters and that goal can do to reach their weightloss goal.
It needs to take in those three inputs, and then send the variables with the prompt in one shot to the AI, and spit out the response back to the Marimo page.
Please just respond with only the Marimo python code, nothing else.


"""

# Base templates for the three sections
INPUT_TEMPLATE = """{marimo_info}. Create a marimo cell that requests user input for {purpose}. 
The input should be stored in variables that will be used later.
Only return the Python code, no explanations."""

ANALYSIS_TEMPLATE = """{marimo_info}. Create a marimo cell that uses the following user input variables: {input_vars}
to analyze and process data using an LLM with this goal: {goal}
Only return the Python code, no explanations."""

OUTPUT_TEMPLATE = """{marimo_info}. Create a marimo cell that takes the analysis results and formats them 
in a clear, visually appealing way using marimo's display capabilities.
The variables to display are: {output_vars}
Only return the Python code, no explanations."""


def create_marimo_section(llm, template, **kwargs):
    """Helper function to generate each section of the marimo notebook"""
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke(kwargs)

    # Strip out markdown code block markers if present
    cleaned_code = response.replace("```python", "").replace("```", "").strip()
    return cleaned_code


def main(prompt: str, apikey: str, projectname: str):
    """
    This is a CLI that generates a marimo app from a prompt.
    """
    # Initialize the LLM
    llm = ChatOpenAI(api_key=apikey, temperature=0.7)

    # First, analyze the user's prompt to determine the required components
    planning_template = """Given this prompt for an AI agent: {prompt}
    Return a JSON object with these fields:
    - input_vars: list of required input variables
    - analysis_goal: specific goal for the analysis section
    - output_vars: list of variables to display in the results"""

    planning_chain = (
        ChatPromptTemplate.from_template(planning_template) | llm | StrOutputParser()
    )
    plan_response = planning_chain.invoke({"prompt": prompt})

    # Parse the JSON response
    import json

    plan = json.loads(plan_response)

    # Generate each section
    input_section = create_marimo_section(
        llm, INPUT_TEMPLATE, purpose=prompt, marimo_info=marimo_info
    )

    analysis_section = create_marimo_section(
        llm,
        ANALYSIS_TEMPLATE,
        input_vars=plan["input_vars"],
        goal=plan["analysis_goal"],
        marimo_info=general_marimo_info_instructions,
    )

    output_section = create_marimo_section(
        llm,
        OUTPUT_TEMPLATE,
        output_vars=plan["output_vars"],
        marimo_info=general_marimo_info_instructions,
    )

    # Combine sections into a marimo file
    marimo_content = f"""
import marimo

__generated_with = "0.11.8"
app = marimo.App(width="medium")


@app.cell
def _():
    from phoenix.otel import register

    tracer_provider = register(
      project_name="appname", # Default is 'default'
      endpoint="http://localhost:6006/v1/traces",
    )
    return register, tracer_provider


@app.cell
def _(tracer_provider):
    from openinference.instrumentation.langchain import LangChainInstrumentor

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    return (LangChainInstrumentor,)

{input_section}

{analysis_section}

{output_section}

if __name__ == "__main__":
    app.run()

"""
    # Create the projectname folder if it doesn't exist
    os.makedirs(projectname, exist_ok=True)

    # Save to file inside projectname folder
    with open(f"{projectname}/main.py", "w") as f:
        f.write(marimo_content)

    # Open up the projectname folder & run "marimo edit main.py"
    import subprocess

    # Navigate to the projectname directory
    os.chdir(projectname)
    subprocess.run(["marimo", "edit", "main.py"])


def mainx():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
