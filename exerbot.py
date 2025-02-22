
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

app_name = input("Enter the name of the exercise app: ")
target_audience = input("Enter the target audience of the exercise app: ")
features = input("Enter the features of the exercise app: ")

class ExerciseApp:
    def __init__(self, app_name, target_audience, features):
        self.app_name = app_name
        self.target_audience = target_audience
        self.features = features

    def generate_app(self):
        print(f"Creating a new exercise app called {self.app_name} for {self.target_audience} with features like {self.features}")

exercise_app = ExerciseApp(app_name, target_audience, features)
exercise_app.generate_app()

class MarimoCell:
    def __init__(self, age, gender, fitness_level, exercise_preferences):
        self.age = age
        self.gender = gender
        self.fitness_level = fitness_level
        self.exercise_preferences = exercise_preferences
        
    def recommend_exercise_routine(self):
        # Use LLM to recommend personalized exercise routines based on user input
        # Add code here to process data and make recommendations
        
        return recommended_exercise_routine

# Example of how to use the MarimoCell class
user_input = {
    'age': 30,
    'gender': 'male',
    'fitness_level': 'beginner',
    'exercise_preferences': ['running', 'weightlifting']
}

marimo_cell = MarimoCell(**user_input)
recommended_exercise_routine = marimo_cell.recommend_exercise_routine()
print(recommended_exercise_routine)

# Create a marimo cell to display analysis results
recommended_exercises = ['Running', 'Swimming', 'Cycling']
duration = '30 minutes'
intensity_level = 'Moderate'
calories_burned = 250

marimo.cell({
    "type": "table",
    "data": {
        "columns": ['Recommended Exercises', 'Duration', 'Intensity Level', 'Calories Burned'],
        "rows": [
            [recommended_exercises[0], duration, intensity_level, calories_burned],
            [recommended_exercises[1], duration, intensity_level, calories_burned],
            [recommended_exercises[2], duration, intensity_level, calories_burned]
        ]
    }
})

if __name__ == "__main__":
    app.run()

