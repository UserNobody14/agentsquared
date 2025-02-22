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


if __name__ == "__main__":
    app.run()
