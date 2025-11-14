from factcheck import FactCheck
import json

factcheck_instance = FactCheck(
    retriever="serper",
    api_config={
        "OPENAI_API_KEY": "",
        "SERPER_API_KEY": ""
    }
)

text = "Paris is the capital of France, and London is the capital of Germany"
results = factcheck_instance.check_text(text)
print(json.dumps(results, indent=4))