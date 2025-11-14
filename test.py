from factcheck import FactCheck

factcheck_instance = FactCheck(
    retriever="google",
    api_config={
        "OPENAI_API_KEY": ""
    }
)

# Example text
text = "Your text here"

# Run the fact-check pipeline
results = factcheck_instance.check_response(text)
print(results)