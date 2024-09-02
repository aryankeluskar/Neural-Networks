from transformers import pipeline

# Initialize the pipeline for text generation using the Sentdex/WSB-GPT-13B model
pipe = pipeline("text-generation", model="Sentdex/WSB-GPT-13B")

# Define your prompt
prompt = """### Comment:
Should I go for Robinhood or Fidelity, as a beginner investor who is a college student and just wants to invest in invest funds?

### REPLY:
"""

# Generate text based on the prompt
generated_text = pipe(prompt, max_length=128, num_return_sequences=1)

# Extract and print the generated text
print(generated_text[0]['generated_text'].split("### END.")[0])
