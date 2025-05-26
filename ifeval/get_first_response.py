"""
Get the initial response from the model
"""
import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import fire
from tqdm import tqdm
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file

def main_fuc(model):
    data=read_file("instruction_following_eval/data/input_data.jsonl")
    save_file_path=f"ifeval/logs/{model}/{model}_first_response.jsonl"
    new_data=[]
    for dp in tqdm(data):
        response=call_llm(model=model,
                temperature=0.0,
                messages=[{"role":"user","content":dp["prompt"]}])
        new_dp={
            "id": dp["key"],
            "prompt": dp["prompt"],
            "response": response
        }
        new_data.append(new_dp)
        if len(new_data)%10==0:
            save_file(save_file_path,new_data)
    save_file(save_file_path,new_data)


if __name__ == "__main__":
    fire.Fire(main_fuc)