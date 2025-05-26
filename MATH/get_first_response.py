import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import fire
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file
from tqdm import tqdm
from MATH.extract_answer import extract_answer

def get_first_response(model,query,temperature):
    """Get the initial response for the command"""
    messages=[{"role":"user","content":query}]
    response=call_llm(model=model,messages=messages,temperature=temperature)
    return response


def main_fuc(model):
    data=read_file("MATH/data/all_random_500.jsonl")
    save_file_path=f"MATH/logs/{model}/all_random_500_first_response.jsonl"
    new_data=[]
    for dp in tqdm(data):
        query=dp["query"]
        response=get_first_response(model=model,query=query,temperature=0.0)
        pred=extract_answer(query,response)
        new_dp={
            "id":dp["id"],
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