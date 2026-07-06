import json


# 1. This mimics our Small_LLM_Model
class MockLLM:
    def encode(self, text: str) -> list[int]:
        print(f"👉 [ENCODE & ADD TO MEMORY]: '{text}'")
        return [1]  # Mock token IDs

    def choose_value(self, field_name: str) -> str:
        # This simulates the model filling in the blank based on what we ask
        answers = {
            "function_name": "get_weather",
            "location": "Paris",
            "days": "5",
        }
        val = answers[field_name]
        print(f"🤖 [MODEL GENERATED VALUE]: '{val}'")
        return val


# 2. This mimics a single Function Parameter Definition
class Parameter:
    def __init__(self, type_str: str):
        self.type = type_str


# 3. This mimics the FunctionDefinition
class MockFunction:
    def __init__(self, name: str, parameters: dict):
        self.name = name
        self.parameters = parameters


# 4. Here is our stripped-down Decoder class
class MiniDecoder:
    def __init__(self):
        self.llm = MockLLM()
        # We define one function: get_weather(location: string, days: integer)
        self.functions = {
            "get_weather": MockFunction(
                "get_weather",
                {
                    "location": Parameter("string"),
                    "days": Parameter("integer"),
                },
            )
        }
        self.ids = []

    def add(self, text: str):
        # This simulates adding fixed JSON text to the model's memory
        self.ids += self.llm.encode(text)

    def run(self, prompt: str) -> dict:
        print(f"--- STARTING RUN FOR PROMPT: '{prompt}' ---\n")

        # STEP 1: Force start the JSON string
        self.add('{"name": "')

        # STEP 2: Let the model pick the function name
        name = self.llm.choose_value("function_name")
        self.add(name)

        # STEP 3: Setup the parameter block syntax manually
        self.add('", "parameters": {')

        parameters = {}
        items = list(self.functions[name].parameters.items())

        # STEP 4: Loop through each parameter required by the function
        for index, (key, schema) in enumerate(items):
            # Write the parameter key syntax: e.g., "location":
            self.add(f'"{key}": ')

            if schema.type == "string":
                self.add('"')  # Add opening quote
                val = self.llm.choose_value(key)  # Model fills string value
                self.add('"')  # Add closing quote
                parameters[key] = val
            else:
                val = self.llm.choose_value(key)  # Model fills number value
                self.add(val)
                parameters[key] = int(val)

            # If there are more parameters left, add a comma separator
            if index < len(items) - 1:
                self.add(", ")

        # STEP 5: Slam the final brackets shut
        self.add("}}")

        print("\n--- RUN FINISHED ---")
        return {"prompt": prompt, "name": name, "parameters": parameters}


# --- EXECUTE THE REAL EXERCISE ---
decoder = MiniDecoder()
final_output = decoder.run("What is the weather like in Paris for 5 days?")

print("\nFinal Python Dictionary Returned:")
print(json.dumps(final_output, indent=2))
