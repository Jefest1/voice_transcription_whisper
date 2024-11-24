import ollama
response = ollama.chat(
    model='llama3.2',
    messages=[
        {'role': 'system', 'content': 'Test connection'},
        {'role': 'user', 'content': 'Hello'}
    ]
)
print(response)
