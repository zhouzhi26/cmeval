import json
from pathlib import Path

folder = Path(r"c:\Users\Lenovo\Desktop\result\cmeval\property")
json_files = list(folder.glob("*.json"))
print(json_files)
for i in json_files:

    with open(i, "r",encoding="utf-8") as f:
        data = json.load(f)
        TP=0
        TN=0
        FP=0
        FN=0
        l=len(data)
        for j in data:
          
            
            if "False" == j["answer"] and "No" in j["model_response"]:
                TN+=1
            elif "False" == j["answer"] and "Yes" in j["model_response"]:
                FP+=1
            elif "True" == j["answer"] and "No" in j["model_response"]:
                FN+=1
            elif "True" == j["answer"] and "Yes" in j["model_response"]:
                TP+=1
        acc=(TP+TN)/(TP+TN+FP+FN)
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=2*precision*recall/(precision+recall)
        print(f"{i}: {acc}, {precision}, {recall}, {f1}")

       