"""
We refactor the original ifeval evaluation code to adapt to the evaluation of smulti-round self-correction.
"""
import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import fire
import collections
from utils.process_file import read_file,save_file
from instruction_following_eval import instructions_registry

def print_report(outputs,turn=-1):
    """Prints a report on accuracy scores."""

    result_dict={
        "strict-prompt":None,
        "strict-instruction":None,
        "loose-prompt":None,
        "loose-instruction":None,
    }
    for eval_type in ["strict","loose"]:
        prompt_total = 0
        prompt_correct = 0
        instruction_total = 0
        instruction_correct = 0

        tier0_total = collections.defaultdict(int)
        tier0_correct = collections.defaultdict(int)

        tier1_total = collections.defaultdict(int)
        tier1_correct = collections.defaultdict(int)

        for example in outputs["logs"]:
            if turn >= len(example["turns"]):
                cur_turn=-1
            else:
                cur_turn=turn
            instruction_id_list = [item["instruction_id"] for item in example["turns"][cur_turn]["is_following_list"]]
            if eval_type=="strict":
                follow_instruction_list = [item["is_strict_following"] for item in example["turns"][cur_turn]["is_following_list"]]
            else:
                follow_instruction_list = [item["is_loose_following"] for item in example["turns"][cur_turn]["is_following_list"]]
            
            prompt_total += 1
            if all(follow_instruction_list):
                prompt_correct += 1

            instruction_total += len(instruction_id_list)
            instruction_correct += sum(follow_instruction_list)

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list
            ):
                instruction_id = instruction_id.split(":")[0]
                tier0_total[instruction_id] += 1
                if followed_or_not:
                    tier0_correct[instruction_id] += 1

            for instruction_id, followed_or_not in zip(
                instruction_id_list, follow_instruction_list
            ):
                tier1_total[instruction_id] += 1
                if followed_or_not:
                    tier1_correct[instruction_id] += 1
        if eval_type=="strict":
            result_dict["strict-prompt"]=round(prompt_correct*100 / prompt_total,2)
            result_dict["strict-instruction"]=round(instruction_correct*100 / instruction_total,2)
        else:
            result_dict["loose-prompt"]=round(prompt_correct*100 / prompt_total,2)
            result_dict["loose-instruction"]=round(instruction_correct*100 / instruction_total,2)
    return result_dict

def get_loose_responses(response):
        r = response.split("\n")
        response_remove_first = "\n".join(r[1:]).strip()
        response_remove_last = "\n".join(r[:-1]).strip()
        response_remove_both = "\n".join(r[1:-1]).strip()
        revised_response = response.replace("*", "")
        revised_response_remove_first = response_remove_first.replace("*", "")
        revised_response_remove_last = response_remove_last.replace("*", "")
        revised_response_remove_both = response_remove_both.replace("*", "")
        all_responses = [
            response,
            revised_response,
            response_remove_first,
            response_remove_last,
            response_remove_both,
            revised_response_remove_first,
            revised_response_remove_last,
            revised_response_remove_both,
        ]
        return all_responses

class IfEvalEvaulator:
    def __init__(self,response_file_path):
        self.input_data=read_file("instruction_following_eval/data/input_data.jsonl")
        self.response_data=read_file(response_file_path)
        self.response_data["logs"]=self.response_data["logs"]

    def eval_all(self):
        for dp in self.response_data["logs"]:
            input_dp=[input_dp for input_dp in self.input_data if input_dp['key']==dp['id']][0]
            self.eval_dp(dp,input_dp)
    
    def eval_dp(self,dp,input_dp):
        instruction_list=input_dp['instruction_id_list']
        # For the evaluation of the response in each turn
        for turn in dp["turns"]:
            turn["is_following_list"]=[]
            response=turn['response']
            # assert len(response) > 0
            loose_responses=get_loose_responses(response)
            for index, instruction_id in enumerate(instruction_list):
                instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
                instruction = instruction_cls(instruction_id)
                instruction.build_description(**input_dp['kwargs'][index])
                args = instruction.get_instruction_args()
                if args and "prompt" in args:
                    instruction.build_description(prompt=input_dp['prompt'])
                is_strict_following=self.is_strict_follow_instruction(response,instruction)
                is_loose_following=self.is_loose_follow_instruction(loose_responses,instruction)
                turn["is_following_list"].append({"instruction_id":instruction_id,
                                                "args":args,
                                                "is_strict_following":is_strict_following,
                                                "is_loose_following":is_loose_following})
                turn["strict_success"]=all([item['is_strict_following'] for item in turn["is_following_list"]])
                turn["loose_success"]=all([item['is_loose_following'] for item in turn["is_following_list"]])
        dp["final_strict_success"]=all([item['is_strict_following'] for item in dp["turns"][-1]["is_following_list"]])
        dp["final_loose_success"]=all([item['is_loose_following'] for item in dp["turns"][-1]["is_following_list"]])
                

    def is_strict_follow_instruction(self,response,instruction):
        is_following=False
        if response.strip() and instruction.check_following(response):
            is_following=True
        return is_following
        
    def is_loose_follow_instruction(self,loose_responses,instruction):
        is_following=False
        for r in loose_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break
        return is_following
    
    def __call__(self):
        self.eval_all()
        max_turn=max([len(dp["turns"]) for dp in self.response_data["logs"]])
        eval_result={}
        for i in range(max_turn):
            eval_result[F"cur_turn_{i}"]=print_report(self.response_data,i)
        print(eval_result)
        new_data={
            "start_time":self.response_data["start_time"],
            "end_time":self.response_data["end_time"],
            "args":self.response_data["args"],
            "result":eval_result,
            "logs":self.response_data["logs"],
        }
        return new_data

def main_fuc(infered_file_path):
    evalutor=IfEvalEvaulator(infered_file_path)
    evaled_data=evalutor()
    save_file_path=infered_file_path.replace("infered","evaled")
    print("save_file_path:",save_file_path)
    save_file(save_file_path,evaled_data)


if __name__=="__main__":
    fire.Fire(main_fuc)



