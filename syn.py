import json
import time
from openai import OpenAI
from datetime import datetime


client = OpenAI(
    api_key="",
    base_url="",
)


input_file = ""
output_file = ""

print("Reading question data...")
with open(input_file, 'r', encoding='utf-8') as f:
    questions = json.load(f)

print(f"Found {len(questions)} questions in total")

results = []
total = len(questions)


for idx, item in enumerate(questions, 1):
    try:
        print(f"Processing question {idx}/{total}...")
        
     
        completion = client.chat.completions.create(
            model="",
            messages=[
                {'role': 'system', 'content': ''},
                {'role': 'user', 'content': item["smile_name_question"]}
            ],
            extra_body={"enable_thinking": False}
        )

       
        model_answer = completion.choices[0].message.content
        print(model_answer)
        
        
     
        result = {
            "chemistry_name": item.get("chemistry_name", ""),
            "smile_name_question": item.get("smile_name_question", ""),#"chemistry_name_question"
            "source_tag": item.get("source_tag", ""),
            "model_response": model_answer
        }
        
        results.append(result)       
        print(f"Processed {idx} questions, saving intermediate results...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
    
        
        
    except Exception as e:
        print(f"Error while processing question {idx}: {str(e)}")
        # Even if an error occurs, still record the question information
        result = {
            "chemistry_name": item.get("chemistry_name", ""),
            "smile_name_question": item.get("smile_name_question", ""),
            #"chemistry_name_question": item.get("chemistry_name_question", ""),
            "source_tag": item.get("source_tag", ""),
            "model_response": f"Error: {str(e)}"
        }
        results.append(result)

# Save all results at the end
print("Saving final results...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Done! Results saved to {output_file}")
print(f"Total processed questions: {len(results)}")
