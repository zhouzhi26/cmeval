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


input_file = ""
output_file = ""

with open(input_file, 'r', encoding='utf-8') as f:
    question=pd.read_csv(f)

print(len(question))
results = []
for i in range(0,9000):
   
    print(f"Processing question {i}...")
    completion = client.chat.completions.create(
            model="",
            messages=[
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content':f"Please answer the following question only with True or False.{question.iloc[i,4]}"}
            ],
            extra_body={"enable_thinking": False}
        )
    r={
        "common_name": question.iloc[i,2],
        "question": question.iloc[i,4],
        "source_tag": "True" if question.iloc[i,-1]==1 else "False",
        "answer": completion.choices[0].message.content
    }
    print(r)
    results.append(r)
    print(f"Processed {i} questions, saving intermediate results...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('over')
       

