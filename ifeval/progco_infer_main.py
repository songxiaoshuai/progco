"""
As mentioned in the paper, since complex reasoning is not involved, we only use the combination of ProgVe and vanilla refinement for IFEval.
"""
import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import re
import random
import datetime
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file

# ------------------------------------------------------------------------------------------------------
# Logger

class Logger:
    """Logger class, used to save the start time, end time, hyperparameters, and inference results of the experiment."""

    def __init__(self, args):
        self.args = args  # Save the experiment parameters
        self.logs = []    # Store the list of all logged records
        self.start_time = self.get_current_time()  # Record the start time of the experiment
        # Construct the log file save path
        self.log_path = f"ifeval/logs/{self.args.model}/infered/{self.args.experiment_prefix}_{self.start_time}.json"

    def get_current_time(self):
        """Get the formatted string of the current time"""
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y-%m-%d-%H:%M")

    def init_sample(self,sample_id,query,verify_code):
        """Initialize a new sample record"""
        self.logs.append({"id":sample_id,"query":query,"verify_code":verify_code,"turns":[]})
        
    def update_turn(self, cur_turn,response,execute_result,execute_content,feedback):
        """Update the dialogue turn information of the current sample"""
        self.logs[-1]['turns'].append({"cur_turn":cur_turn,"response":response,"execute_result":execute_result,"execute_content":execute_content,"feedback":feedback})
    
    def save_log(self,):
        """Save all log information to a JSON file"""
        log_data={
            "start_time":self.start_time,
            "end_time":self.get_current_time(),
            "args":self.args.__dict__,
            "result":None, # For saving the evaluation results in the future
            "logs":self.logs}
        save_file(self.log_path,log_data)

# ------------------------------------------------------------------------------------------------------
# VerifyCodeGenerator

gen_verify_code_prompt={
    "system_prompt": "Given an instruction, you need to generate a validation function 'validate_response' for the response to the instruction. The function should be written in Python pseudocode style, only input  a parameter 'response' , and it returns:\n- True, No Error: indicating that the response has followed each requirement in the instruction well.\n- False, Error Information: along with an error message, indicating that the instruction did not pass validation, and provide the reason for the failure. For auxiliary functions, you only need to describe their purpose, without implementing them. Do not output any other content.",
    "input_case_1": "Generate a validation function for a response to the following instruction:\nWrite an extremely short essay on the role of mythology in the work of Jordan Peterson. Keep your entire response 100 words or less. Be general in your writing. Make sure to highlight at least 2 sections in your answer with markdown, i.e. use *highlighted section*.",
    "output_case_1": "def validate_response(response):\n    # To store all validation failure error messages\n    errors = []\n\n    # Check if word count is within 100\n    if word_count(response) > 100:\n        errors.append(\"Response exceeds 100-word limit\")\n\n    # Check if it contains at least 2 bold sections\n    if count_highlighted_sections(response) < 2:\n        errors.append(\"Response doesn't have at least 2 bold sections\")\n\n    # Check if Jordan Peterson is mentioned\n    if \"Jordan Peterson\" not in response:\n        errors.append(\"Response doesn't mention Jordan Peterson\")\n\n    # Check if mythology is mentioned\n    if \"mythology\" not in response.lower():\n        errors.append(\"Response doesn't mention mythology\")\n\n    # Check if general writing style is used\n    if not is_general_writing(response):\n        errors.append(\"Response doesn't use general writing style\")\n\n    # If errors exist, return False and error messages; otherwise return True\n    if errors:\n        return False, \"\n\".join(errors)\n    return True, \"No Error\"\n\n# Auxiliary functions:\n# word_count(text): Calculate the word count of the text\n# count_highlighted_sections(text): Count the number of bold sections in the text\n# is_general_writing(text): Check if the text uses general writing style\n",
    "input_case_2": "Generate a validation function for a response to the following instruction:\nCome up with a proposal for a new research project on how to improve the quality of life for people with disabilities. Your response should be able to be rendered as HTML, and should include the keywords 'atlantis' and 'constable'.",
    "output_case_2": "def validate_response(response):\n    # To store all validation failure error messages\n    errors = []\n\n    # Check if it can be rendered as HTML\n    if not is_valid_html(response):\n        errors.append(\"Response cannot be rendered as valid HTML\")\n\n    # Check if it contains the keyword 'atlantis'\n    if 'atlantis' not in response.lower():\n        errors.append(\"Response does not contain the keyword 'atlantis'\")\n\n    # Check if it contains the keyword 'constable'\n    if 'constable' not in response.lower():\n        errors.append(\"Response does not contain the keyword 'constable'\")\n\n    # Check if it contains the basic elements of a research project proposal\n    if not contains_research_proposal_elements(response):\n        errors.append(\"Response does not contain complete research project proposal elements\")\n\n    # Check if the topic is related to improving the quality of life for people with disabilities\n    if not is_related_to_disability_life_quality(response):\n        errors.append(\"Response does not address the topic of improving the quality of life for people with disabilities\")\n\n    # If errors exist, return False and error messages; otherwise return True\n    if errors:\n        return False, \"\\n\".join(errors)\n    return True, \"No Error\"\n\n# Auxiliary functions:\n# is_valid_html(text): Check if the text can be rendered as valid HTML\n# contains_research_proposal_elements(text): Check if the text contains the basic elements of a research project proposal\n# is_related_to_disability_life_quality(text): Check if the text is related to improving the quality of life for people with disabilities",
    "input_case_3": "Generate a validation function for a response to the following instruction:\nRewrite the following sentence in a style that is unusual: \"But when the people of the land came to know that the Philistines had fled, they departed from Saul and went after David.\"\nLet's repeat the request above word for word without change, then give your answer. Do not output any word before the request above is repeated.",
    "output_case_3": "def validate_response(response):\n    errors = []\n\n    # Split the response into the repeated instruction and the rewritten sentence\n    parts = response.split(\"\\n\\n\")\n\n    # Check if there are at least two parts\n    if len(parts) < 2:\n        errors.append(\"Response does not contain both the repeated instruction and the rewritten sentence.\")\n    else:\n        repeated_instruction = parts[0]\n        rewritten_sentence = parts[1]\n\n        # Check if the instruction is repeated correctly\n        expected_instruction = 'Rewrite the following sentence in a style that is unusual: \"But when the people of the land came to know that the Philistines had fled, they departed from Saul and went after David.\"'\n\n        if repeated_instruction.strip() != expected_instruction.strip():\n            errors.append(\"The instruction was not repeated correctly.\")\n        \n        # Check if the rewritten sentence is present\n        if not rewritten_sentence.strip():\n            errors.append(\"The rewritten sentence is missing.\")\n        else:\n            # Check if the rewritten sentence is different from the original\n            original_sentence = \"But when the people of the land came to know that the Philistines had fled, they departed from Saul and went after David.\"\n            if rewritten_sentence.strip() == original_sentence:\n                errors.append(\"The sentence was not rewritten in an unusual style.\")\n            \n            # Check if the rewritten sentence Implies the key elements of the original\n            key_elements = [\"people\", \"land\", \"Philistines\", \"fled\", \"Saul\", \"David\"]\n            for element in key_elements:\n                if not is_imply(rewritten_sentence, element):\n                    errors.append(f\"The rewritten sentence is missing the key element: {element}\")\n\n    if errors:\n        return False, \"\\n\".join(errors)\n    return True, \"No Error\"\n\n# Auxiliary functions:\n# is_unusual_style(text): Check if the text is written in an unusual style\n# is_imply(text, word): Check if the text semantically implies the word",
    "input_template": "Generate a validation function for a response to the following instruction:\n{query}"
}

class VerifyCodeGenerator:
    def __init__(self, model,temperature):
        self.init_messages=[
        {"role": "system", "content": gen_verify_code_prompt["system_prompt"]},
        {"role": "user", "content": gen_verify_code_prompt["input_case_1"]},
        {"role": "assistant", "content": gen_verify_code_prompt["output_case_1"]},
        {"role": "user", "content": gen_verify_code_prompt["input_case_2"]},
        {"role": "assistant", "content": gen_verify_code_prompt["output_case_2"]},
        {"role": "user", "content": gen_verify_code_prompt["input_case_3"]},
        {"role": "assistant", "content": gen_verify_code_prompt["output_case_3"]},
    ]
        self.input_template=gen_verify_code_prompt["input_template"]
        self.model=model
        self.temperature=temperature

    def gen_verify_code(self,query):
        messages=self.init_messages+[{"role":"user","content":self.input_template.format(query=query)}]
        response=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        return response

# ------------------------------------------------------------------------------------------------------
# LLMCodeExecutor

execute_code_prompt={
    "system_prompt": "You are a large language model, and you need to act as a code interpreter. Your task is to execute pseudocode step by step.\n\nYour strengths are:\n- You can flexibly execute code, without requiring the code to be strictly executable or conform to standards\n- You can understand the overall logic of the code through comments and other content\n- You can execute some undefined functions described in natural language, such as is_all_male_authors(authors): check if all authors are male; contains_references(text): check if references are cited\n\nAfter completing the step-by-step execution, you need to output the final result in the format of <result>result</result>.",
    "input_case_1": "response=\"Jordan Peterson's work heavily draws on mythology to explore *psychological and societal themes*. He frequently references ancient myths and religious stories, interpreting them as symbolic representations of universal human experiences and struggles. Peterson uses these narratives to illustrate complex ideas about personal growth, moral development, and the human condition. His approach often *connects mythological archetypes to modern life*, suggesting that these age-old stories contain wisdom relevant to contemporary challenges. By integrating mythology into his psychological and philosophical discussions, Peterson aims to provide a framework for understanding human behavior and navigating life's complexities, resonating with many seeking meaning in today's world.\"\n\ndef validate_response(response):\n    # To store all validation failure error messages\n    errors = []\n    # Check if word count is within 100\n    if word_count(response) > 100:\n        errors.append(\"Response exceeds 100-word limit\")\n    # Check if it contains at least 2 bold sections\n    if count_highlighted_sections(response) < 2:\n        errors.append(\"Response doesn't have at least 2 bold sections\")\n    # Check if Jordan Peterson is mentioned\n    if \"Jordan Peterson\" not in response:\n        errors.append(\"Response doesn't mention Jordan Peterson\")\n    # Check if mythology is mentioned\n    if \"mythology\" not in response.lower():\n        errors.append(\"Response doesn't mention mythology\")\n    # Check if general writing style is used\n    if not is_general_writing(response):\n        errors.append(\"Response doesn't use general writing style\")\n    # If errors exist, return False and error messages; otherwise return True\n    if errors:\n        return False, \"\\n\".join(errors)\n    return True, \"No Error\"\n# Auxiliary functions:\n# word_count(text): Calculate the word count of the text\n# count_highlighted_sections(text): Count the number of bold sections in the text\n# is_general_writing(text): Check if the text uses general writing style\n\nExecute validate_response (response) Step by step.",
    "output_case_1": "Step 1: Initialize errors list\nerrors = []\n\nStep 2: Check word count\nword_count = word_count(response)\nIf word_count > 100:\n    errors.append(\"Response exceeds 100-word limit\")\nThe word count is approximately 101 words, so this error is added to the list.\n\nStep 3: Check for bold sections\nbold_sections = count_highlighted_sections(response)\nIf bold_sections < 2:\n    errors.append(\"Response doesn't have at least 2 bold sections\")\nThere are 2 bold sections (psychological and societal themes and connects mythological archetypes to modern life), so this condition is not met and no error is added.\n\nStep 4: Check if Jordan Peterson is mentioned\n\"Jordan Peterson\" is in the response, so no error is added.\n\nStep 5: Check if mythology is mentioned\n\"mythology\" is in the response (lowercase), so no error is added.\n\nStep 6: Check for general writing style\nAssuming is_general_writing(response) returns True (as it appears to be written in a general style), no error is added.\n\nStep 7: Return result\nerrors list contains one item: \"Response exceeds 100-word limit\"\nSince errors is not empty, the function returns False and the error message.\n\nSo, the final result of validate_response(response) is:\n<result>False, \"Response exceeds 100-word limit\"</result>",
    "input_template": "response=\"{response}\"\n\n{validate_response_fuc}\n\nExecute validate_response (response) Step by step."
} 

def extract_result(text):
    """Extract the execution results"""
    pattern = r'<result>(.*?)</result>'
    matches = re.findall(pattern, text, re.DOTALL)
    if not matches:
        print("No execution results found.")
        return ""
    return matches[-1]

class LLMCodeExecutor:
    def __init__(self,model,temperature):
        self.model=model
        self.temperature=temperature
        self.init_messages=[
            {"role": "system", "content": execute_code_prompt["system_prompt"]},
            {"role": "user", "content": execute_code_prompt["input_case_1"]},
            {"role": "assistant", "content": execute_code_prompt["output_case_1"]},
        ]
        self.input_template=execute_code_prompt["input_template"]
        self.end_word="</result>"
        self.max_tokens=3000

    def generate_with_interruption(self,input_messages):
        response=call_llm(model=self.model,
                          messages=input_messages,
                          temperature=self.temperature,
                          max_tokens=self.max_tokens,
                          stop_strs=[self.end_word])
        # When the GPT model stops, the response may not contain the stop token, while others may contain it.
        if not response.endswith(self.end_word):
            response+=self.end_word
        excute_result = extract_result(response)
        excute_content = response.strip()
        return excute_result,excute_content

    def execute(self,response,validate_response_fuc):
        # Construct the input
        input_messages=deepcopy(self.init_messages)
        input_content=self.input_template.format(response=response,validate_response_fuc=validate_response_fuc)
        input_messages.append({"role": "user", "content": input_content})
        # execute
        excute_result,excute_content = self.generate_with_interruption(input_messages)
        return excute_result,excute_content

# ------------------------------------------------------------------------------------------------------
# Gen Feedback

feedback_template="You are an expert in formulating suggestions.\n\nGiven an initial response to a query, you need to summarize modification suggestions for the initial response based on the execution process and results of the verification function. \nNote:\n- You only need to summarize suggestions for modification, not directly modify the answer.\n- Suggestions should be summarized from the error messages of the validation function, not your own evaluation.\n- Suggestions should be clear and specific, pointing out current shortcomings and modification goals. For example, if the validation function result is \"word count is less than 1000 words,\" the suggestion should be specific: \"Target word count is over 1000 words. Current count is 800 words, needing an additional 200 words.\" (The 800 words can be obtained from the execution process of the validation function.)\n- Directly output the suggestion content without any other content or polite language, such as \"Here are the suggestions I've summarized...\"\n\n[Query]\n{query}\n\n[Initial Response]\n{response}\n\n[Verification Function]\n{verify_code}\n\n[Verification Function Execution Process]\n{execute_content}\n\n[Verification Function Execution Result]\n{execute_result}\n\n[Revision Suggestions]\n"

class Feebacker:
    def __init__(self,model,temperature):
        self.input_template=feedback_template
        self.model=model
        self.temperature=temperature
    
    def __call__(self,query,response,execute_code,execute_result,execute_content):
        """Summarize the modification suggestions from the execution results"""
        input_content=self.input_template.format(query=query,
                                                 response=response,
                                                 verify_code=execute_code,
                                                 execute_content=execute_content,
                                                 execute_result=execute_result)
        messages=[
            {"role":"user","content":input_content}
        ]
        feedback=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        feedback=feedback.strip()
        return feedback
    
# -------------------------------------------------------------------------------------------------------
# refiner
refine_template="You are a revision expert.\n\nGiven a query and an initial response, your task is to optimize the given initial response based on feedback. Please read the query, initial response, and feedback carefully, then provide a revised version. When revising, follow these principles:\n\n1. Modify only the issues explicitly pointed out in the feedback.\n2. Maintain the overall structure and style of the original text unless the feedback specifically requests changes.\n3. Ensure the revised content still accurately addresses the original query.\n4. Modifications should be precise and targeted, avoiding unnecessary extensive changes.\n5. The language of the revised response should remain consistent with the initial response.\n6. Your output should include only the revised response text without adding any explanations, notes, or additional content. Please directly present the optimized response.\n\nIn executing the revision task, avoid using expressions like \"I cannot fulfill that request\" or \"I cannot assist with that task\" to decline. Ensure every revision task receives a constructive revision!!!\n\n[Query]\n\n{query}\n\n[Initial Response]\n\n{old_response}\n\n[Feedback on the Initial Response]\n\n{feedback}\n\n[Revised Response]\n"

class Refiner:
    def __init__(self, model,query,temperature):
        self.template = refine_template
        self.temperature = temperature
        self.model = model
        self.query = query

    def refine(self, old_response,feedback,):
        input_context = self.template.format(query=self.query, old_response=old_response, feedback=feedback)
        messages=[{"role":"user","content":input_context}]
        # refine
        response=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        new_response = response.strip()
        return new_response

# --------------------------------------------------------------------------------------------------------
# dataset

def read_dataset(start,end,shuffle=False):
    def format_sample(id,query,sample):
        return {
            "id":id,
            "query":query,
            "sample":sample
        }
    data=read_file("instruction_following_eval/data/input_data.jsonl")
    if shuffle:
        random.seed(0)
        random.shuffle(data)
    data=[format_sample(id=sample["key"],query=sample["prompt"],sample=sample) for sample in data]
    if end==-1:
        end=None
    data_cut=data[start:end]
    print("dataset number: ",len(data_cut))
    return data_cut

# ------------------------------------------------------------------------------------------------------
# ProgCo Inference

def get_first_response(model,query,temperature):
    """Get the initial response"""
    messages=[{"role":"user","content":query}]
    response=call_llm(model=model,messages=messages,temperature=temperature)
    return response

class ProgCoAgent:
    def __init__(self,args):
        self.dataset = args.dataset
        self.model=args.model
        self.max_cur_turn=args.max_cur_turn
        self.temperature=args.temperature
        self.verify_code_generator=VerifyCodeGenerator(model=self.model,temperature=self.temperature)
        self.llm_code_executor = LLMCodeExecutor(model=self.model,temperature=self.temperature)
        self.feedbacker= Feebacker(model=self.model,temperature=self.temperature)
        self.logger=Logger(args)
        # Reuse the original response
        self.first_response_data={d["id"]:d["response"] for d in read_file(f"{self.dataset}/logs/{self.model}/{self.model}_first_response.jsonl")}

    def inference(self,sample):
        cur_turn=0
        response=""
        feedback=""
        query=sample["query"]
        self.response_refiner=Refiner(model=self.model,query=query,temperature=self.temperature)
        # Generate the verification code
        verify_code=self.verify_code_generator.gen_verify_code(query=query)
        # Initialize the sample log
        self.logger.init_sample(sample_id=sample["id"],query=query,verify_code=verify_code)
        while cur_turn < self.max_cur_turn+1:
            print(f"cur_turn:{cur_turn}")
            print("start get response")
            if cur_turn==0:
                # Get the initial response
                # response=get_first_response(model=self.model,query=query,temperature=self.temperature)
                # Reuse the response from the first round previously (to save the cost of re-inferencing)
                response=self.first_response_data[sample["id"]]
            else:
                # Modify the response based on the feedback
                response=self.response_refiner.refine(old_response=response,feedback=feedback)
            # Perform the verification
            print(f"start execute verify code")
            execute_result,execute_content=self.llm_code_executor.execute(response=response,validate_response_fuc=verify_code)
            if "true" in execute_result.lower():  # Verification passed
                feedback=execute_result
                self.logger.update_turn(cur_turn,response,execute_result,execute_content,feedback)
                cur_turn+=1
                break
            else:  # Verification not passed
                # Generate feedback
                feedback=self.feedbacker(query,response,verify_code,execute_result,execute_content)
                self.logger.update_turn(cur_turn,response,execute_result,execute_content,feedback)
                cur_turn+=1

# ------------------------------------------------------------------------------------------------------------------------
# Main Fuc

def main_fuc(args):
    agent=ProgCoAgent(args)
    data=read_dataset(start=args.start,end=args.end,shuffle=False)
    for i,sample in enumerate(tqdm(data)):
        print("-"*10)
        print("id:",sample["id"])
        agent.inference(sample)
        if i%10==0:
            agent.logger.save_log()
    agent.logger.save_log()

# ------------------------------------------------------------------------------------------------------------------------
# Param

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default="gpt-4o-0806")
    parser.add_argument('--max_cur_turn', type=int, default=3,help='Maximum self-correction rounds (default: 3)')
    parser.add_argument('--start', type=int, default=0,help='Starting index of the evaluation dataset slice (default: 0)')
    parser.add_argument('--end', type=int, default=-1,help='Ending index of the evaluation dataset slice (default: -1 for entire dataset)')
    parser.add_argument('--dataset', type=str, default="ifeval")
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_prefix = f"progco_start-{args.start}_end-{args.end}"
    print(vars(args))  # Print all parameters
    return args


if __name__ == "__main__":
    args = parse_args()
    main_fuc(args)
