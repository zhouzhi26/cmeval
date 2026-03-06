import json
from pathlib import Path
folder = Path(r"c:\Users\Lenovo\Desktop\result\chemeval")
json_files = list(folder.glob("*.json"))
for i in json_files:

    with open(i, "r",encoding="utf-8") as f:
        data = json.load(f)
        TP=0
        TN=0
        FP=0
        FN=0
        a=0
        l=len(data)
        for j in data:
            if "False" == j["source_tag"] and "False" in j["answer"]:
                TN+=1
                a+=1
            elif "False" == j["source_tag"] and "True" in j["answer"]:
                FP+=1
                a+=1
            elif "True" == j["source_tag"] and "False" in j["answer"]:
                FN+=1
                a+=1
            elif "True" == j["source_tag"] and "True" in j["answer"]:
                TP+=1
                a+=1
        print(a,len(data))
        acc=(TP+TN)/(TP+TN+FP+FN)
        precision=TP/(TP+FP)
        recall=TP/(TP+FN)
        f1=2*precision*recall/(precision+recall)
        print(f"{i}: {acc}, {precision}, {recall}, {f1}")

       