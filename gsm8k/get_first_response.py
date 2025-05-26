import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import fire
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file
from tqdm import tqdm
from gsm8k.extract_answer import extract_answer

def main_fuc(model):
    temperature=0
    data=read_file("gsm8k/data/gsm8k_test.jsonl")
    save_file_path=f"gsm8k/logs/{model}/{model}_first_response.jsonl"
    new_data=[]
    for dp in tqdm(data):
        id=dp["id"]
        query=dp["question"]
        messages=[{"role":"user","content":query}]
        response=call_llm(model=model,messages=messages,temperature=temperature)
        pred=extract_answer(dp["question"],response)
        new_dp={
            "id":id,
            "query":query,
            "response":response,
            "pred":pred
        }
        new_data.append(new_dp)
        if len(new_data)%10==0:
            save_file(save_file_path,new_data)
    save_file(save_file_path,new_data)

if __name__=="__main__":
    fire.Fire(main_fuc)
