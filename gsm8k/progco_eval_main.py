import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import fire
from tqdm import tqdm
from utils.call_llm import call_llm
from utils.process_file import read_file, save_file
from gsm8k.extract_answer import extract_answer
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,confusion_matrix


judge_prompt="Determine if the following two answers represent the same answer:\n\nAnswer 1: {target}\n\nAnswer 2: {pred}\n\nOutput only one word: \"True\" or \"False\""

def find_cache(triplets, a, b):
    return [t[2] for t in triplets if t[0] == a and t[1] == b]

judge_cache=[]

def judge(target,pred):
    """
    First perform rule-based matching, if the rules cannot match, then perform LLM judging.
    """
    target=target.strip()
    pred=pred.strip()
    fined_cache=find_cache(judge_cache,target,pred)
    # reuse
    if len(fined_cache)==1:
        return fined_cache[0]
    else:
        new_cache=True
        judge_result=False
        if target==pred:  # If the strings are exactly equal
            judge_result=True
        else:
            try:
                # If the string can be converted to a float
                if abs(float(target) - float(pred)) < 1e-3 or abs(float(target) - float(pred*100)) < 1e-3:
                    judge_result=True
                else:
                    judge_result=False
            except:
                # LLM judging
                input_content=judge_prompt.format(target=target,pred=pred)
                output=call_llm(model="gpt-4o-mini-0718",messages=[{"role":"user","content":input_content}])
                if "true" in output.lower():
                    judge_result=True
                else:
                    judge_result=False
        if new_cache:
            judge_cache.append((target,pred,judge_result))
        return judge_result



class Evaluator:
    def __init__(self,infered_data_path):
        assert "infered" in infered_data_path
        self.init_data=read_file("gsm8k/data/gsm8k_test.jsonl")
        self.infered_data=read_file(infered_data_path)
        self.evaled_data_path=infered_data_path.replace("infered","evaled")

    def eval(self):
        # For each data point
        for dp_id,dp in enumerate(tqdm(self.infered_data["logs"])):
            target=[init_dp["label"] for init_dp in self.init_data if init_dp["id"]==dp["id"]][0]
            target=str(target)
            # For each turn
            for turn_id,turn in enumerate(dp["turns"]):
                if "result" in turn:
                    pred=turn["result"]
                else:
                    pred = extract_answer(query=dp["query"],response=turn["response"])
                success=judge(target,pred)
                self.infered_data["logs"][dp_id]["turns"][turn_id]["target"]=target
                self.infered_data["logs"][dp_id]["turns"][turn_id]["pred"]=pred
                self.infered_data["logs"][dp_id]["turns"][turn_id]["success"]=success
            # For last response
            final_response=dp['final_response']
            if "final_result" in dp:
                pred=dp["final_result"]
            else:
                pred = extract_answer(query=dp["query"],response=final_response)
            success=judge(target,pred)
            final_response_dict={}
            final_response_dict["response"]=final_response
            final_response_dict["target"]=target
            final_response_dict["pred"]=pred
            final_response_dict["success"]=success
            self.infered_data["logs"][dp_id]["final_response"]=final_response_dict
            



    def report_recall(self):
        """Statistics the recall (0 for correct answer, 1 for incorrect answer)"""
        recall_dict={}
        preds=[0 if len(dp["turns"])==1 else 1 for dp in self.infered_data["logs"]]
        targets=[0 if dp["turns"][0]["success"] else 1 for dp in self.infered_data["logs"]]
        recall_dict["acc"]=round(accuracy_score(targets,preds)*100,2)
        recall_dict["recall"]=round(recall_score(targets,preds)*100,2)
        recall_dict["precision"]=round(precision_score(targets,preds)*100,2)
        recall_dict["f1"]=round(f1_score(targets,preds)*100,2)
        recall_dict["confusion_matrix"]=str(confusion_matrix(targets,preds))
        return recall_dict


    def report_dp(self,dp,max_turns):
        """Get the validation result and judgment result for each turn of a data point"""
        first_turn_success=dp["turns"][0]["success"]
        turn_success=[first_turn_success]
        for turn_id in range(1,max_turns):
            if turn_id >= len(dp["turns"]):  # If it exceeds the maximum turns
                turn_success.append(turn_success[-1])
                continue
            turn_success.append(dp["turns"][turn_id]["success"])
        assert len(turn_success)==max_turns
        return turn_success


    def report_refine(self):
        """Statistics the refine results"""
        refine={}
        refine_worse_ids=[]
        refine_better_ids=[]
        refine_success_tie_ids=[]
        refine_fail_tie_ids=[]
        for dp_id,dp in enumerate(self.infered_data["logs"]):
            if len(dp["turns"])==1:  # without refine
                continue
            init_success=dp["turns"][0]["success"]
            refine_success=dp["final_response"]["success"]
            if init_success and refine_success:
                refine_success_tie_ids.append(dp["id"])
            elif not init_success and not refine_success:
                refine_fail_tie_ids.append(dp["id"])
            elif not init_success and refine_success:
                refine_better_ids.append(dp["id"])
            elif init_success and not refine_success:
                refine_worse_ids.append(dp["id"])
        refine["refine_dict"]={"all":len(refine_worse_ids)+len(refine_better_ids)+len(refine_success_tie_ids)+len(refine_fail_tie_ids),
                                         "better":len(refine_better_ids),"worse":len(refine_worse_ids),
                                         "success_tie":len(refine_success_tie_ids),"fail_tie":len(refine_fail_tie_ids)}
        refine["refine_better_ids"]=refine_better_ids
        refine["refine_worse_ids"]=refine_worse_ids
        refine["refine_success_tie_ids"]=refine_success_tie_ids
        refine["refine_fail_tie_ids"]=refine_fail_tie_ids
        return refine
        

    def report(self):
        """Report the ACC, recall, and refine for each turn"""
        result={}
        # Final ACC
        result["init_acc"]=round(sum([dp["turns"][0]["success"] for dp in self.infered_data["logs"]])*100/len(self.infered_data["logs"]),2)
        result["final_acc"]=round(sum([dp["final_response"]["success"] for dp in self.infered_data["logs"]])*100/len(self.infered_data["logs"]),2)
        # Each turn ACC
        result["turn_acc"]={}
        max_turns=max([len(dp["turns"]) for dp in self.infered_data["logs"]])
        turns_acc=[0]*max_turns
        init_failed_ids=[dp["id"] for dp in self.infered_data["logs"] if not dp["turns"][0]["success"]]
        final_failed_ids=[dp["id"] for dp in self.infered_data["logs"] if not dp["turns"][-1]["success"]]
        for dp_id,dp in enumerate(tqdm(self.infered_data["logs"])):
            # Get the evaluation result for each turn of a data point
            dp_turns_acc=self.report_dp(dp,max_turns)
            for i in range(max_turns):
                if dp_turns_acc[i]:
                    turns_acc[i]+=1
        turns_acc=[round(turn_acc*100/len(self.infered_data["logs"]),2) for turn_acc in turns_acc]
        result["turn_acc"]["turns_acc"]=turns_acc
        result["turn_acc"]["init_failed_ids"]=init_failed_ids
        result["turn_acc"]["final_failed_ids"]=final_failed_ids
        # log recall
        result["recall"]=self.report_recall()
        # log refine
        result["refine"]=self.report_refine()
        self.infered_data["result"]=result
        # Save
        print("save_file_path:",self.evaled_data_path)
        save_file(self.evaled_data_path,self.infered_data)

    def __call__(self):
        self.eval()
        self.report()

def main_fuc(infered_file_path):
    print("start eval:",infered_file_path)
    evaluator=Evaluator(infered_file_path)
    evaluator()

if __name__=="__main__":
    fire.Fire(main_fuc)
        
    

