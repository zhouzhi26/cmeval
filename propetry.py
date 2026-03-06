import json
import time
from openai import OpenAI
from datetime import datetime
import pandas as pd
import csv

client = OpenAI(
    api_key="",
    base_url="",
)


input_file = "property.json"
output_file = ""
results = []
with open(input_file, 'r', encoding='utf-8') as f:
    question=pd.read_json(f)

try:
    with open(output_file, 'r', encoding='utf-8') as f:
        results = pd.read_json(f)
        start_index = len(results)
        print(f"Found {start_index} existing records, resuming from index {start_index}...")
except FileNotFoundError:
    print("No existing output file found, starting from the beginning...")
except json.JSONDecodeError:
    print("Output file format error, starting from the beginning...")

print(len(question))
start=len(results)
print(question)
print(question.iloc[0])

for i in range(start,len(question)):
   
    print(f"Processing question {i}...")
    completion = client.chat.completions.create(
            model="",
            messages=[
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content':f"Please answer the following question only with True or False.{question.iloc[i,2]}.Please output only True or False !"}
            ],
            #extra_body={"enable_thinking": True,
            #"thinking_budget": 1024},
            #stream=True
        )
    reason=""
    ans=""
    for chunk in completion:
        if not chunk.choices[0]:
            print(chunk.usage)
        delta=chunk.choices[0].delta
        if delta.reasoning_content is not None:
            reason+=delta.reasoning_content
        if delta.content is not None:
            ans+=delta.content
    print(ans,"------------------\n")
    print(reason)

    r={
        "chemical_name": question.iloc[i,0],
        "cas_number": question.iloc[i,1],
        "question": question.iloc[i,2],
        "answer": question.iloc[i,3],
        "model_response": ans,
        #"reasoning": reason
    }
    print(r)
    if isinstance(results, pd.DataFrame):
        results = results.to_dict(orient='records')
    results.append(r)
    print(f"Processed {i} questions, saving intermediate results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('over')


