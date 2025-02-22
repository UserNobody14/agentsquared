import typer
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from phoenix.otel import register

tracer_provider = register(
    project_name="coreapp",  # Default is 'default'
    endpoint="http://localhost:6006/v1/traces",
)

# from langchain.chains import LLMChain
from langchain.schema import StrOutputParser

# Base templates for the three sections
INPUT_TEMPLATE = """Create a marimo cell that requests user input for {purpose}. 
The input should be stored in variables that will be used later.
Only return the Python code, no explanations."""

ANALYSIS_TEMPLATE = """Create a marimo cell that uses the following user input variables: {input_vars}
to analyze and process data using an LLM with this goal: {goal}
Only return the Python code, no explanations."""

OUTPUT_TEMPLATE = """Create a marimo cell that takes the analysis results and formats them 
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
    input_section = create_marimo_section(llm, INPUT_TEMPLATE, purpose=prompt)

    analysis_section = create_marimo_section(
        llm,
        ANALYSIS_TEMPLATE,
        input_vars=plan["input_vars"],
        goal=plan["analysis_goal"],
    )

    output_section = create_marimo_section(
        llm, OUTPUT_TEMPLATE, output_vars=plan["output_vars"]
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

    # Save to file
    with open(f"{projectname}.py", "w") as f:
        f.write(marimo_content)


def mainx():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
