import os
import sys
sys.path.append(os.environ["PROJECT_PATH"])
import re
import time
import datetime
import random
from tqdm import tqdm
from copy import deepcopy
from argparse import ArgumentParser
from extract_answer import extract_answer
from utils.call_llm import call_llm
from utils.process_file import read_file,save_file

# ------------------------------------------------------------------------------------------------------------------------
# Logger

class Logger:
    def __init__(self, args):
        self.args = args
        self.logs=[]
        self.start_time=self.get_current_time()
        save_model_name=self.args.model
        self.log_path=f"MATH/logs/{save_model_name}/infered/{self.args.experiment_prefix}_{self.start_time}.json"

    def get_current_time(self):
        current_time = datetime.datetime.now()
        return current_time.strftime("%Y-%m-%d-%H:%M")

    def init_sample(self,sample_id,query):
        self.logs.append({"id":sample_id,"query":query,"turns":[]})
        
    def update_turn(self,cur_turn,verify_code,response,result,execute_result,execute_content,feedback,reflection,code_feedback):
        self.logs[-1]['turns'].append({"cur_turn":cur_turn,"verify_code":verify_code,"response":response,"result":result,
         "execute_result":execute_result,"execute_content":execute_content,"feedback":feedback,"reflection":reflection,"code_feedback":code_feedback})
    
    def update_final_response(self,response,result):
        self.logs[-1]["final_response"]=response
        self.logs[-1]["final_result"]=result

    def update_insights(self,insights):
        self.logs[-1]["insights"]=insights

    def save_log(self,):
        log_data={
            "start_time":self.start_time,
            "end_time":self.get_current_time(),
            "args":self.args.__dict__,
            "result":None,
            "logs":self.logs}
        save_file(self.log_path,log_data)

# ------------------------------------------------------------------------------------------------------------------------
# VerifyCodeGenerator

gen_verify_code_dict =  {
    "system_prompt": "You are an expert in reverse reasoning verification. Given a problem, you need to generate a reverse verification executable Python function for that problem.\n\n[Reverse Reasoning Introduction]\n* Reverse reasoning is a method of thinking that starts from the result and verifies the problem backwards. Specifically, it involves:\n\n1. Starting with the given answer, rather than the initial conditions of the problem.\n2. Assuming this answer is correct, then conducting reverse deduction based on this assumption and the known conditions in the problem.\n3. Through this reverse deduction, checking whether other known conditions or constraints in the problem can be satisfied.\n4. If the results of the reverse deduction can satisfy all conditions, then the original answer can be considered correct.\n\nFor example, using the answer and known conditions 1 and 2 as assumptions, reverse reason to check if it satisfies known condition 3 in the problem statement.\n\nTo illustrate, consider a problem of solving a quadratic equation ax^2 + bx + c = 0:\n- Forward thinking would start with a, b, c values and use the quadratic formula to calculate x.\n- Reverse reasoning would:\n1. Start with a possible solution x\n2. Substitute it into ax^2 + bx + c\n3. Check if the result equals 0\n4. If it equals 0, the solution is verified as cor\n\n[Requirements]\n1. The verification function should start with the input answer and use reverse reasoning to validate the correctness of the answer.\n2. The verification function should only accept one input (the answer) and output the verification result as True/False.\n3. After generating the verification function name, please first write the reverse analysis verification approach as code comments, then generate the content of the verification function. Please do not output any other content.",
    "input_case_1": "Generate a validation function for an answer to the following question:\nThe length of one leg of a right triangle is 9 meters. The lengths of the other two sides are consecutive integer numbers of meters. What is the number of meters in the perimeter of the triangle?",
    "output_case_1": "def verify_triangle_perimeter(perimeter):\n    # Reverse verification approach:\n    # 1. Start with the given perimeter\n    # 2. Calculate the possible lengths of the hypotenuse and the other leg\n    # 3. Verify if the lengths are consecutive integers\n    # 4. Check if the Pythagorean theorem holds\n    # 5. Verify if the sum of the sides equals the given perimeter\n\n    # Known condition\n    leg1 = 9\n\n    # Calculate the sum of the other two sides\n    other_sides_sum = perimeter - leg1\n\n    # Iterate over possible lengths for the other leg\n    for leg2 in range(1, other_sides_sum):\n        hypotenuse = other_sides_sum - leg2\n\n        # Check if leg2 and hypotenuse are consecutive integers\n        if abs(hypotenuse - leg2) != 1:\n            continue\n\n        # Check if the Pythagorean theorem holds\n        if leg1**2 + leg2**2 != hypotenuse**2:\n            continue\n\n        # If all conditions are satisfied, the perimeter is correct\n        return True\n\n    # If no valid configuration is found, the perimeter is incorrect\n    return False",
    "input_case_2": "Generate a validation function for an answer to the following question(**The verification function should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):\nWhich of the following numbers is the smallest? Your answer is the letter in front of the smallest number.\n\n$\\bullet$ A. $14.45001$ rounded to the nearest tenth.\n\n$\\bullet$ B. $14.54999$ rounded to the nearest tenth.\n\n$\\bullet$ C. $14.44999$ rounded to the nearest tenth.",
    "output_case_2": "def verify_smallest_number_choice(choice):\n    # Reverse verification approach:\n    # 1. Start with the given choice (A, B, or C)\n    # 2. Determine the rounded value for each option\n    # 3. Compare the rounded values to identify the smallest\n    # 4. Verify if the given choice corresponds to the smallest rounded value\n\n    # Calculate the rounded values for each option\n    rounded_A = round(14.45001, 1)\n    rounded_B = round(14.54999, 1)\n    rounded_C = round(14.44999, 1)\n\n    # Determine the smallest rounded value\n    smallest_value = min(rounded_A, rounded_B, rounded_C)\n\n    # Verify if the given choice corresponds to the smallest rounded value\n    if choice == 'A' and rounded_A == smallest_value:\n        return True\n    elif choice == 'B' and rounded_B == smallest_value:\n        return True\n    elif choice == 'C' and rounded_C == smallest_value:\n        return True\n    else:\n        return False",
    "input_case_3": "Generate a validation function for an answer to the following question(**The verification function should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):\nIn how many ways can the letters of the word ``COPYRIGHT'' be arranged?",
    "output_case_3": "def verify_arrangement_count(arrangement_count):\n    # Reverse verification approach:\n    # 1. Start with the given arrangement count\n    # 2. Calculate the factorial of the number of letters in \"COPYRIGHT\"\n    # 3. Verify if the calculated factorial matches the given arrangement count\n    # 4. Check if all conditions in the problem are satisfied\n\n    from math import factorial\n\n    # Known condition: the word \"COPYRIGHT\" has 9 unique letters\n    num_letters = 9\n\n    # Calculate the number of arrangements using factorial\n    calculated_arrangement_count = factorial(num_letters)\n\n    # Verify if the calculated arrangement count matches the given arrangement count\n    return calculated_arrangement_count == arrangement_count",
    "input_template": "Generate a validation function for an answer to the following question(**The verification function should start with the input answer and use reverse reasoning to validate the correctness of the answer.**):\n{query}"
}

class VerifyCodeGenerator:
    def __init__(self, model,temperature):
        self.init_messages=[
            {"role": "system", "content": gen_verify_code_dict["system_prompt"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_1"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_1"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_2"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_2"]},
            {"role": "user", "content": gen_verify_code_dict["input_case_3"]},
            {"role": "assistant", "content": gen_verify_code_dict["output_case_3"]},
        ]
        self.input_template=gen_verify_code_dict["input_template"]
        self.model=model
        self.temperature=temperature

    def gen_verify_code(self,query):
        messages=self.init_messages+[{"role":"user","content":self.input_template.format(query=query)}]
        response=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        return response

# ------------------------------------------------------------------------------------------------------------------------
# LLMCodeExecutor

execute_verify_code_dict = {
    "system_prompt": "You are a verification expert proficient in code execution and possessing extensive world knowledge, and you have the following advantages:\n\n1. Ability to execute code flexibly, unrestricted by strict syntax or standards\n2. Capacity to comprehend code purpose, overall logic, and potential issues through analysis of comments and context\n\nYour task is to verify a given mathematical problem and its solution. A data annotator has written verification code for this problem. Please proceed as follows:\n\n1. Carefully read the provided problem, solution process, and answer\n2. Execute the verification code step by step\n3. If you discover any issues in the verification code, make appropriate revisions before continuing execution\n4. Derive the final verification result\n\nInput information:\n\n[Problem]\n{query}\n\n[Solution Process]\n{response}\n\n[Solution Answer]\n{result}\n\n[Verification Code]\n{validate_response_fuc}\n\nPlease output your analysis and results strictly in the following format, without adding any additional content:\n\n[Execution of Verification Code]\n{Detailed step-by-step execution process}\n\n[Verification Result]\n{True or False}",
    "input_case_1": "[Problem]\nThe product $ab = 1200$, $a$ is an integer, and $b$ is an odd integer. What is the largest possible value of $b$?\n\n[Solution Process]\nTo solve for the largest possible value of \\( b \\) given that \\( ab = 1200 \\), \\( a \\) is an integer, and \\( b \\) is an odd integer, we start by factoring 1200.\n\nFirst, we find the prime factorization of 1200:\n\\[\n1200 = 12       imes 100 = (2^2         imes 3)         imes (2^2       imes 5^2) = 2^4         imes 3  imes 5^2\n\\]\n\nSince \\( b \\) is an odd integer, it cannot have any factors of 2. Therefore, \\( b \\) must be composed only of the odd prime factors of 1200, which are 3 and 5.\n\nThe prime factorization of 1200 without the factor of 2 is:\n\\[\n3       imes 5^2 = 3    imes 25 = 75\n\\]\n\nThus, the largest possible value of \\( b \\) is 75. We verify this by checking if \\( a \\) is an integer when \\( b = 75 \\).\n\nCalculate \\( a \\) when \\( b = 75 \\):\n\\[\na = \n    rac{1200}{b} = \n                   rac{1200}{75}\n\\]\n\nPerform the division:\n\\[\n1200 \\div 75 = 16\n\\]\n\nSince \\( a = 16 \\) is an integer, the conditions are satisfied. Therefore, the largest possible value of \\( b \\) is \\oxed{75}\\).\n\n[Solution Answer]\n75\n\n[Verification Code]\ndef verify_largest_odd_b(b):\n    # Reverse verification approach:\n    # 1. Start with the given value of b\n    # 2. Verify that b is an odd integer\n    # 3. Calculate a = 1200 / b and check if a is an integer\n    # 4. Ensure that b is the largest possible odd integer satisfying the conditions\n\n    # Known condition: the product ab = 1200\n    product = 1200\n\n    # Check if b is an odd integer\n    if b % 2 == 0:\n        return False\n\n    # Calculate a and check if it is an integer\n    if product % b != 0:\n        return False\n\n    # Check if b is the largest possible odd integer\n    # Iterate over possible odd values greater than b to ensure b is the largest\n    for potential_b in range(b + 2, product + 1, 2):\n        if product % potential_b == 0:\n            return False\n\n    # If all conditions are satisfied, the given b is correct\n    return True",
    "output_case_1": "[Execution of Verification Code]\n\n1. The function verify_largest_odd_b is called with the argument b = 75.\n\n2. Check if b is an odd integer:\n   75 % 2 = 1, which is not equal to 0, so b is odd. This check passes.\n\n3. Calculate a and check if it's an integer:\n   product % b = 1200 % 75 = 0\n   This means 1200 is divisible by 75 without a remainder, so a is an integer. This check passes.\n\n4. Check if b is the largest possible odd integer:\n   The function iterates over odd numbers greater than 75, starting from 77:\n   - For 77: 1200 % 77 ≠ 0\n   - For 79: 1200 % 79 ≠ 0\n   - ...\n   - For 1199: 1200 % 1199 ≠ 0\n   \n   No odd number greater than 75 divides 1200 evenly, so 75 is indeed the largest.\n\n5. Since all conditions are satisfied, the function returns True.\n\n[Verification Result]\nTrue",
    "input_case_2": "[Problem]\nAt the candy store, a chocolate bar costs $c$ dollars and a vanilla bar costs $2$ dollars more than a chocolate bar. Jamie buys a chocolate bar and three vanilla bars, and Kevin buys five chocolate bars. How much money, in total, do Jamie and Kevin spend at the candy store in terms of $c$?\n\n[Solution Process]\nJamie spends a total of $c + 3(c + 2) = 3c + 6$ dollars. \nKevin spends a total of $5c$ dollars. \nThe total spent by both Jamie and Kevin is $3c + 6 + 5c = 8c + 6$ dollars.\n\n[Solution Answer]\n8c + 6\n\n[Verification Code]\ndef verify_total_spent(total_spent, c):\n    # Reverse verification approach:\n    # 1. Start with the given total spent\n    # 2. Calculate the cost of a vanilla bar in terms of c\n    # 3. Calculate the total cost for Jamie's purchase\n    # 4. Calculate the total cost for Kevin's purchase\n    # 5. Sum the costs of both purchases\n    # 6. Verify if the calculated total matches the given total spent\n\n    # Calculate the cost of a vanilla bar\n    vanilla_cost = c + 2\n\n    # Calculate the total cost for Jamie's purchase\n    jamie_cost = c + 3 * vanilla_cost\n\n    # Calculate the total cost for Kevin's purchase\n    kevin_cost = 5 * c\n\n    # Calculate the total spent by both Jamie and Kevin\n    calculated_total_spent = jamie_cost + kevin_cost\n\n    # Verify if the calculated total matches the given total spent\n    return calculated_total_spent == total_spent",
    "output_case_2": "[Execution of Verification Code]\n\nThe function verify_total_spent is called with the arguments total_spent = 8c + 6 and c.\n\n1. Calculate the cost of a vanilla bar:\n   vanilla_cost = c + 2\n\n2. Calculate the total cost for Jamie's purchase:\n   jamie_cost = c + 3 * vanilla_cost\n   jamie_cost = c + 3 * (c + 2)\n   jamie_cost = c + 3c + 6\n   jamie_cost = 4c + 6\n\n3. Calculate the total cost for Kevin's purchase:\n   kevin_cost = 5 * c\n\n4. Calculate the total spent by both Jamie and Kevin:\n   calculated_total_spent = jamie_cost + kevin_cost\n   calculated_total_spent = 4c + 6 + 5c\n   calculated_total_spent = 9c + 6\n\n5. Verify if the calculated total matches the given total spent:\n   Check if 9c + 6 is equal to 8c + 6.\n   This equation is not true in general, so the verification code is incorrect.\n\n[Verification Result]\nFalse",
    "input_template": "[Problem]\n{query}\n\n[Solution Process]\n{response}\n\n[Solution Answer]\n{result}\n\n[Verification Code]\n{validate_response_fuc}"
}

def split_content_result(response):
    if "[Verification Result]" in response:
        content,result=response.split("[Verification Result]")
        content=content.strip()
        result=result.strip()
        if "true" in result.lower():
            result=True
        else:
            result=False
        return content,result
    else:
        return response,True

class LLMCodeExecutor:
    def __init__(self,model,temperature):
        self.model=model
        self.temperature=temperature
        self.init_messages=[
            {"role": "system", "content": execute_verify_code_dict["system_prompt"]},
            {"role": "user", "content": execute_verify_code_dict["input_case_1"]},
            {"role": "assistant", "content": execute_verify_code_dict["output_case_1"]},
            {"role": "user", "content": execute_verify_code_dict["input_case_2"]},
            {"role": "assistant", "content": execute_verify_code_dict["output_case_2"]},
            ]
        self.input_template=execute_verify_code_dict["input_template"]

    def execute(self,query,response,result,validate_response_fuc):
        retries = 0
        max_retries = 10
        while retries < max_retries:
            try:
                # init
                self.messages=deepcopy(self.init_messages)
                self.excute_content=""
                # execute
                input_content=self.input_template.format(query=query,
                                                         response=response,
                                                         result=result,
                                                         validate_response_fuc=validate_response_fuc)
                self.messages.append({"role": "user", "content": input_content})
                execute_response=call_llm(model=self.model,
                                    messages=self.messages,
                                    temperature=self.temperature)
                
                excute_content,excute_result=split_content_result(execute_response)
                return excute_result,excute_content
            except KeyboardInterrupt:
                print("Operation canceled by user.")
                break
            except Exception as e:
                print(f"Someting wrong:{e}. Retrying in {retries*10+10} seconds...")
                time.sleep(retries*10)
                retries += 1
        return "Execute failed", "Execute failed"

# ------------------------------------------------------------------------------------------------------------------------
# Gen Feedback

feedback_dict={
    "input_case_1": "Extract key information from the execution process of the verification code and convert it into natural language form.\n\n[Problem]\nIn right triangle $EAD$ with right angle at $A$, $AE=4$ units, $AB=BC=CD$ and the area of triangle $ABE=6$ sq units. What is the length of segment $CE$? Express your answer as a decimal to the nearest tenth. [asy]\ndraw((0,0)--(0,4)--(9,0)--cycle);\ndraw((0,4)--(3,0));\ndraw((0,4)--(6,0));\ndraw(rightanglemark((9,0),(0,0),(0,4)));\nlabel(\"A\",(0,0),S);\nlabel(\"B\",(3,0),S);\nlabel(\"C\",(6,0),S);\nlabel(\"D\",(9,0),S);\nlabel(\"E\",(0,4),N);\nlabel(\"4\",(0,0)--(0,4),W);\n[/asy]\n\n[Execution of Verification Code]\nThe predicted length of segment \\( CE \\) is 9.4. Now, let's execute the `verify_segment_length_CE(CE_length)` function with `CE_length = 9.4`.\n\nStep 1: Initialize known conditions\n```python\nAE = 4  # length of AE\narea_ABE = 6  # area of triangle ABE\n```\n\nStep 2: Calculate the length of \\( AB \\) using the area of triangle \\( ABE \\)\n```python\nAB = (2 * area_ABE) / AE\n```\nThis is a simple calculation, so I can handle it myself:\n- \\( AB = (2 * 6) / 4 = 3 \\)\n\nStep 3: Calculate the total length of \\( AD \\)\n```python\nAD = 3 * AB\n```\nSince \\( AB = 3 \\), we have:\n- \\( AD = 3 * 3 = 9 \\)\n\nStep 4: Calculate the length of \\( CE \\) using the Pythagorean theorem in triangle \\( EAD \\)\n```python\ncalculated_CE_length = (AE**2 + AD**2)**0.5\n```\nThis involves a square root calculation, so I will use PythonInterpreterTool.\n<code>\ndef execute():\n    AE = 4\n    AD = 9\n    calculated_CE_length = (AE**2 + AD**2)**0.5\n    return round(calculated_CE_length, 1)\nexecute()\n</code>\nThe execution result of PythonInterpreterTool is execute_result:\n\n\n9.8\n\nThe execution result of the PythonInterpreterTool indicates that the calculated length of \\( CE \\) is 9.8, which does not match the given length of 9.4.\n\nStep 5: Verify if the calculated \\( CE \\) length matches the given \\( CE \\) length\n```python\nreturn round(calculated_CE_length, 1) == round(CE_length, 1)\n```\nSince the calculated \\( CE \\) length is 9.8 and the given \\( CE \\) length is 9.4, the verification code will return `False`.\n\n[Verification Result]\nThe final result of the execution is:\n<result>False</result>",
    "output_case_1": "In the problem, we are given a right triangle \\( EAD \\) with a right angle at \\( A \\), where \\( AE = 4 \\) units, and segments \\( AB = BC = CD \\). The area of triangle \\( ABE \\) is 6 square units. We need to find the length of segment \\( CE \\) and express it as a decimal to the nearest tenth.\n\nThe verification process begins by calculating the length of segment \\( AB \\) using the area of triangle \\( ABE \\). Given the area formula for a triangle, we have:\n\n\\[ AB = \frac{2 \times \text{area\\_ABE}}{AE} = \frac{2 \times 6}{4} = 3 \\]\n\nNext, since \\( AB = BC = CD \\), the total length of \\( AD \\) is:\n\n\\[ AD = 3 \times AB = 3 \times 3 = 9 \\]\n\nUsing the Pythagorean theorem in triangle \\( EAD \\), we calculate the length of \\( CE \\):\n\n\\[ CE = \\sqrt{AE^2 + AD^2} = \\sqrt{4^2 + 9^2} = \\sqrt{16 + 81} = \\sqrt{97} \\]\n\nThe calculated length of \\( CE \\) is approximately 9.8 when rounded to the nearest tenth.\n\nThe verification process checks if this calculated length matches the given length of 9.4. Since 9.8 does not equal 9.4, the verification code returns `False`, indicating a discrepancy between the calculated and given lengths of segment \\( CE \\).",
    "input_case_2": "Extract key information from the execution process of the verification code and convert it into natural language form.\n\n[Problem]\nEvaluate $\\sqrt{5\\cdot10\\cdot14\\cdot21\\cdot15\\cdot20}$.\n\n[Execution of Verification Code]\nThe predicted answer is \\(1050\\sqrt{2}\\). Now, let's execute the `verify_sqrt_evaluation(answer)` function using this value.\n\nStep 1: Calculate the product of the numbers\n```python\nproduct = 5 * 10 * 14 * 21 * 15 * 20\n```\nSince this is a complex calculation, I will use PythonInterpreterTool.\n<code>\ndef execute():\n    product = 5 * 10 * 14 * 21 * 15 * 20\n    return product\nexecute()\n</code>\nThe execution result of PythonInterpreterTool is execute_result:\n\n\n4410000\n\nThe calculated product of the numbers is 4410000.\n\nStep 2: Square the given answer\nThe given answer is \\(1050\\sqrt{2}\\). We need to square this expression:\n\\[\n(1050\\sqrt{2})^2 = 1050^2 \times (\\sqrt{2})^2 = 1050^2 \times 2\n\\]\n\nSince this is a complex calculation, I will use PythonInterpreterTool.\n<code>\ndef execute():\n    answer = 1050 * (2 ** 0.5)\n    squared_answer = (1050 ** 2) * 2\n    return squared_answer\nexecute()\n</code>\nThe execution result of PythonInterpreterTool is execute_result:\n\n\n2205000\n\nThe squared answer is 2205000.\n\nStep 3: Verify if the squared answer matches the calculated product\nWe compare the squared answer (2205000) with the calculated product (4410000).\n\nSince 2205000 does not equal 4410000, the verification fails.\n\n[Verification Result]\nThe final result of the execution is:\n<result>False</result>",
    "output_case_2": "The problem involves evaluating the expression \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\).\n\n1. **Calculate the Product of the Numbers:**\n   - The product of the numbers is calculated as \\(5 \times 10 \times 14 \times 21 \times 15 \times 20\\).\n   - Using a computational tool, the product is found to be 4,410,000.\n\n2. **Square the Given Answer:**\n   - The predicted answer is \\(1050\\sqrt{2}\\).\n   - To verify, we square this expression: \\((1050\\sqrt{2})^2 = 1050^2 \times 2\\).\n   - Using a computational tool, the squared value is calculated to be 2,205,000.\n\n3. **Verification:**\n   - Compare the squared answer (2,205,000) with the calculated product (4,410,000).\n   - Since 2,205,000 does not equal 4,410,000, the verification fails.\n\n**Conclusion:**\nThe predicted answer \\(1050\\sqrt{2}\\) is incorrect because its square does not match the calculated product of the numbers. The verification process returns a result of \"False.\"",
    "input_template": "Extract key information from the execution process of the verification code and convert it into natural language form.\n\n[Problem]\n{query}\n\n[Execution of Verification Code]\n{execute_content}\n"
}

class Feebacker:
    def __init__(self,model,temperature):
        self.init_messages=[
            {"role":"user","content":feedback_dict["input_case_1"]},
            {"role":"assistant","content":feedback_dict["output_case_1"]},
            {"role":"user","content":feedback_dict["output_case_1"]},
            {"role":"assistant","content":feedback_dict["output_case_2"]},
        ]
        self.input_template=feedback_dict["input_template"]
        self.model=model
        self.temperature=temperature

    def __call__(self, query,response,execute_code,execute_result,execute_content):
        execute_content=execute_content.replace("[Execution of Verification Code]","").strip()
        input_content=self.input_template.format(query=query,
                                                response=response,
                                                execute_content=execute_content)
        messages=self.init_messages+[{"role":"user","content":input_content}]
        feedback=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        feedback=feedback.strip()
        return feedback

# ------------------------------------------------------------------------------------------------------------------------
# VerifyCodeRefiner

code_refine_dict={
    "system_prompt": "Your task is to reflex whether a verification code is correct.\n\n[Reverse Reasoning Introduction]\n* Reverse reasoning is a method of thinking that starts from the result and verifies the problem backwards. Specifically, it involves:\n\n1. Starting with the given answer, rather than the initial conditions of the problem.\n2. Assuming this answer is correct, then conducting reverse deduction based on this assumption and the known conditions in the problem.\n3. Through this reverse deduction, checking whether other known conditions or constraints in the problem can be satisfied.\n4. If the results of the reverse deduction can satisfy all conditions, then the original answer can be considered correct.\n\nFor example, using the answer and known conditions 1 and 2 as assumptions, reverse reason to check if it satisfies known condition 3 in the problem statement.\n\nTo illustrate, consider a problem of solving a quadratic equation ax^2 + bx + c = 0:\n- Forward thinking would start with a, b, c values and use the quadratic formula to calculate x.\n- Reverse reasoning would:\n1. Start with a possible solution x\n2. Substitute it into ax^2 + bx + c\n3. Check if the result equals 0\n4. If it equals 0, the solution is verified as cor\n\nGiven a problem [Problem], a reverse reasoning validation expert generated an executable Python function for reverse validation [Initial Verification Code] of this problem. However, the verification code did not pass when verifying a solution [Solution] of the problem.\n\n**The error may come from the solution or from the verification code.**\n\nYou need to carefully analyze and complete the following tasks:\n1. Reflect on the initial verification code:\n   - First, examine the logic and accuracy of the verification code **step by step**\n   - If errors or inadequacies are found, provide detailed feedback and suggestions for improvement\n2. Provide a new verification code:\n   - If the initial verification code is correct, you can use the original verification code\n   - If errors or inadequacies are found, revise the verification code based on your reflection\n   - Keep detailed comments in the revised validation code\n\nPlease strictly output your analysis and revision according to the following format, without any additional content:\n\n[Reflection]\n{Detailed reflection process}\n\n[New Verification Code]\n{Complete verification code based on the reflection}\n",
    "input_case_1": "Reflect on the initial verification code and provide a new verification code.\n(Note: If the verification code is correct, you also need to repeat the initial verification code word for word in the new verification code section [New Verification Code].)\n\n[Problem]\nThe four-digit number $25AB$ is divisible by nine, with $A$ the tens digit and $B$ the units digit. How many different such four-digit numbers could $25AB$ represent?\n\n[Initial Verification Code]\ndef verify_divisible_by_nine_count(count):\n    # Reverse verification approach:\n    # 1. Start with the given count of numbers\n    # 2. Iterate over all possible values of A and B (0 to 9)\n    # 3. Calculate the sum of the digits for each combination of A and B\n    # 4. Check if the sum is divisible by 9\n    # 5. Count the number of valid combinations\n    # 6. Verify if the counted number of combinations matches the given count\n\n    # Known digits in the number\n    fixed_digits_sum = 2 + 5\n\n    # Initialize a counter for valid numbers\n    valid_count = 0\n\n    # Iterate over all possible values of A and B\n    for A in range(10):\n        for B in range(10):\n            # Calculate the sum of the digits\n            total_sum = fixed_digits_sum + A + B\n\n            # Check if the sum is divisible by 9\n            if total_sum % 9 == 0:\n                valid_count += 1\n\n    # Verify if the counted number of combinations matches the given count\n    return valid_count == count\n\n[Solution]\nTo determine how many different four-digit numbers \\(25AB\\) could represent, where \\(25AB\\) is divisible by 9, we need to use the divisibility rule for 9. A number is divisible by 9 if the sum of its digits is divisible by 9.\n\nThe number \\(25AB\\) has the digits 2, 5, A, and B. The sum of these digits is:\n\n\\[\n2 + 5 + A + B = 7 + A + B\n\\]\n\nWe require that this sum be divisible by 9. Therefore, we need:\n\n\\[\n7 + A + B \\equiv 0 \\pmod{9}\n\\]\n\nThis implies:\n\n\\[\nA + B \\equiv -7 \\equiv 2 \\pmod{9}\n\\]\n\nThus, we need:\n\n\\[\nA + B \\equiv 2 \\pmod{9}\n\\]\n\nNow, we will find all pairs \\((A, B)\\) where \\(A\\) and \\(B\\) are digits (i.e., \\(0 \\leq A, B \\leq 9\\)) that satisfy this condition.\n\nWe will check each possible value of \\(A\\) from 0 to 9 and find the corresponding \\(B\\) such that \\(A + B \\equiv 2 \\pmod{9}\\).\n\n- If \\(A = 0\\), then \\(B \\equiv 2 \\pmod{9}\\), so \\(B = 2\\).\n- If \\(A = 1\\), then \\(B \\equiv 1 \\pmod{9}\\), so \\(B = 1\\).\n- If \\(A = 2\\), then \\(B \\equiv 0 \\pmod{9}\\), so \\(B = 0\\).\n- If \\(A = 3\\), then \\(B \\equiv 8 \\pmod{9}\\), so \\(B = 8\\).\n- If \\(A = 4\\), then \\(B \\equiv 7 \\pmod{9}\\), so \\(B = 7\\).\n- If \\(A = 5\\), then \\(B \\equiv 6 \\pmod{9}\\), so \\(B = 6\\).\n- If \\(A = 6\\), then \\(B \\equiv 5 \\pmod{9}\\), so \\(B = 5\\).\n- If \\(A = 7\\), then \\(B \\equiv 4 \\pmod{9}\\), so \\(B = 4\\).\n- If \\(A = 8\\), then \\(B \\equiv 3 \\pmod{9}\\), so \\(B = 3\\).\n- If \\(A = 9\\), then \\(B \\equiv 2 \\pmod{9}\\), so \\(B = 2\\).\n\nEach value of \\(A\\) from 0 to 9 gives a unique corresponding value of \\(B\\), resulting in 10 valid pairs \\((A, B)\\).\n\nThus, there are \\(\boxed{10}\\) different four-digit numbers \\(25AB\\) that are divisible by 9.\n\n[Verification Process]\nThe predicted count of valid numbers is 10. Now, let's execute the `verify_divisible_by_nine_count(count)` function with `count = 10`.\n\nStep 1: Initialize the known digits sum\n```python\nfixed_digits_sum = 2 + 5\n```\nThis is a simple calculation, so I can handle it myself. The sum of the fixed digits is `7`.\n\nStep 2: Initialize a counter for valid numbers\n```python\nvalid_count = 0\n```\nThe counter is initialized to zero.\n\nStep 3: Iterate over all possible values of A and B\n```python\nfor A in range(10):\n    for B in range(10):\n        # Calculate the sum of the digits\n        total_sum = fixed_digits_sum + A + B\n\n        # Check if the sum is divisible by 9\n        if total_sum % 9 == 0:\n            valid_count += 1\n```\nThis involves iterating over all combinations of `A` and `B` from 0 to 9, calculating the total sum, and checking divisibility by 9. Since this is a complex iterative calculation, I will use PythonInterpreterTool.\n\n<code>\ndef execute():\n    fixed_digits_sum = 7\n    valid_count = 0\n    for A in range(10):\n        for B in range(10):\n            total_sum = fixed_digits_sum + A + B\n            if total_sum % 9 == 0:\n                valid_count += 1\n    return valid_count\nexecute()\n</code>\nThe execution result of PythonInterpreterTool is execute_result:\n\n\n11\n\nThe execution result indicates that the function `verify_divisible_by_nine_count(count)` returns `11` for the count of valid numbers, which does not match the predicted count of `10`.\n\n[Verification Result]\nThe final result of the execution is:\n<result>False</result>",
    "output_case_1": "[Reflection]\nUpon reviewing the initial verification code, it accurately implements the process needed to verify the total number of four-digit numbers $25AB$ divisible by nine:\n\n1. **Check the Known Digits**: The given digits, 2 and 5, are correctly summed to fix the partial sum of 7.\n2. **Iterating through Possible Digits for A and B**: The code iterates through all possible values (0 to 9) for A and B, forming valid combinations as mandated.\n3. **Divisibility Check**: For each (A, B) pair, the sum of all digits in $25AB$ is checked for divisibility by 9. This correctly reflects the condition that $2 + 5 + A + B$ must be divisible by 9.\n4. **Counting Valid Outcomes**: Each combination that meets this criterion increments a counter.\n5. **Validation**: The final valid count is checked against the expected count to determine if they match.\n\nThe problem arises because the solution logic resulting in 10 valid combinations did not account for potential off-by-one or computational errors in certain ranges of A and B.\n\nUpon executing the code using Python, the correct execution confirmed 11 valid combinations, indicating a misunderstanding or logical oversight in either the reasoning or the count during initial expectations.\n\n[New Verification Code]\ndef verify_divisible_by_nine_count(count):\n    # Reverse verification approach:\n    # 1. Start with the given count of numbers\n    # 2. Iterate over all possible values of A and B (0 to 9)\n    # 3. Calculate the sum of the digits for each combination of A and B\n    # 4. Check if the sum is divisible by 9\n    # 5. Count the number of valid combinations\n    # 6. Verify if the counted number of combinations matches the given count\n\n    # Known digits in the number\n    fixed_digits_sum = 2 + 5\n\n    # Initialize a counter for valid numbers\n    valid_count = 0\n\n    # Iterate over all possible values of A and B\n    for A in range(10):\n        for B in range(10):\n            # Calculate the sum of the digits\n            total_sum = fixed_digits_sum + A + B\n\n            # Check if the sum is divisible by 9\n            if total_sum % 9 == 0:\n                valid_count += 1\n\n    # Verify if the counted number of combinations matches the given count\n    return valid_count == count",
    "input_case_2": "Reflect on the initial verification code and provide a new verification code.\n\n[Problem]\nAmber, Micah, and Ahito ran 52 miles in total. Amber ran 8 miles. Micah ran 3.5 times what Amber ran. How many miles did Ahito run?\n\n[Initial Verification Code]\ndef verify_ahito_miles(ahito_miles):\n    # Reverse verification approach:\n    # 1. Start with the given miles run by Ahito\n    # 2. Calculate the miles run by Micah (should be 3.5 times Amber's miles)\n    # 3. Calculate the miles run by Amber\n    # 4. Check if the sum of all three runners' miles equals the total\n    # 5. Verify if all conditions in the problem are satisfied\n\n    # Known conditions\n    total_miles = 52\n    amber_miles = 8\n    micah_multiplier = 3.5\n\n    # Calculate the miles run by Micah\n    micah_miles = amber_miles * micah_multiplier\n\n    # Calculate the miles run by Ahito\n    calculated_ahito_miles = total_miles - amber_miles - micah_miles\n\n    # Check if the calculated miles run by Ahito matches the given answer\n    if abs(calculated_ahito_miles - ahito_miles) > 0.01:  # Allow for small rounding errors\n        return False\n\n    # All conditions are satisfied\n    return True\n\n[Solution]\nMicah ran 8 x 3.5 = <<8*3.5=28>>28 miles.\nSo, Ahito ran 52 - 8 - 28 = <<52-8-28=16>>16 miles. Answer: \boxed{16}.\n\n[Verification Process]\nThe function verify_ahito_miles is called with the argument ahito_miles = 16.\n\nThe code calculates the miles run by Micah using the given information:\nmicah_miles = amber_miles * micah_multiplier = 8 * 3.5 = 28\n\nThe code then calculates the miles run by Ahito:\ncalculated_ahito_miles = total_miles - amber_miles - micah_miles = 52 - 8 - 28 = 16\n\nThe code checks if the calculated miles run by Ahito matches the given answer:\nabs(calculated_ahito_miles - ahito_miles) = abs(16 - 16) = 0, which is within the tolerance of 0.01. This check passes.\n\nSince all checks have passed, the function returns True.",
    "output_case_2": "[Reflection]\nThe initial verification code accurately checks whether the solution for Ahito's miles is correct using reverse reasoning. The logic follows these steps:\n1. It calculates the miles run by Micah based on Amber's miles and the provided multiplier (3.5 times Amber's miles). This is correctly calculated as `micah_miles = 8 * 3.5 = 28`.\n2. It computes Ahito's miles by subtracting Amber and Micah's miles from the total miles run by all three participants. This calculation is correctly reflected as `calculated_ahito_miles = 52 - 8 - 28 = 16`.\n3. Finally, it compares the computed Ahito's miles with the provided solution for Ahito's miles (16). The comparison allows for a small tolerance which is good practice to handle potential floating-point arithmetic issues with a condition of `abs(calculated_ahito_miles - ahito_miles) > 0.01`.\n\nThese steps ensure the logic is sound and matches the calculations from the solution. \n\nTherefore, there are no logical errors or inadequacies in the initial verification code that need correction. The initial reasoning logic is correct and complies with the reverse reasoning method.\n\n[New Verification Code]\ndef verify_ahito_miles(ahito_miles):\n    # Reverse verification approach:\n    # 1. Start with the given miles run by Ahito\n    # 2. Calculate the miles run by Micah (should be 3.5 times Amber's miles)\n    # 3. Calculate the miles run by Amber\n    # 4. Check if the sum of all three runners' miles equals the total\n    # 5. Verify if all conditions in the problem are satisfied\n\n    # Known conditions\n    total_miles = 52\n    amber_miles = 8\n    micah_multiplier = 3.5\n\n    # Calculate the miles run by Micah\n    micah_miles = amber_miles * micah_multiplier\n\n    # Calculate the miles run by Ahito\n    calculated_ahito_miles = total_miles - amber_miles - micah_miles\n\n    # Check if the calculated miles run by Ahito matches the given answer\n    if abs(calculated_ahito_miles - ahito_miles) > 0.01:  # Allow for small rounding errors\n        return False\n\n    # All conditions are satisfied\n    return True",
    "input_template": "Reflect on the initial verification code and provide a new verification code.\n(Note: If the verification code is correct, you also need to repeat the initial verification code word for word in the new verification code section [New Verification Code].)\n\n[Problem]\n{query}\n\n[Initial Verification Code]\n{verify_code}\n\n[Solution]\n{response}\n\n[Verification Process]\n{verify_process}\n"
}

class CodeRefiner:
    def __init__(self, model,query,temperatue):
        self.init_messages=[
            {"role":"system","content":code_refine_dict["system_prompt"]},
            {"role":"user","content":code_refine_dict["input_case_1"]},
            {"role":"assistant","content":code_refine_dict["output_case_1"]},
            {"role":"user","content":code_refine_dict["input_case_2"]},
            {"role":"assistant","content":code_refine_dict["output_case_2"]},
        ]
        self.template = code_refine_dict["input_template"]
        self.model = model
        self.query = query
        self.memory = []
        self.temperatue=temperatue
    
    def extract_new_solution(self,new_response):
        if "[New Verification Code]" not in new_response:
            return "",new_response
        else:
            return new_response.split("[New Verification Code]")[0].strip(),new_response.split("[New Verification Code]")[1].strip()

    def refine(self, response,verify_code,verify_process):
        verify_process=verify_process.replace("[Execution of Verification Code]","").strip()
        input_context = self.template.format(query=self.query,response=response,verify_code=verify_code,verify_process=verify_process)
        messages=self.init_messages+[{"role":"user","content":input_context}]
        # refine
        output=call_llm(model=self.model,messages=messages,temperature=self.temperatue)
        output = output.strip()
        reflection,new_code=self.extract_new_solution(output) 
        if "```python" in new_code:
            new_code=new_code.replace("```python","")
        if "```" in new_code:
            new_code=new_code.replace("```","")
        return reflection,new_code


# ------------------------------------------------------------------------------------------------------------------------
# ResponseRefiner

refine_dict={
    "system_prompt": "Your task is to reflex whether a solution is correct.\n\nGiven a problem [Problem], a reverse reasoning validation expert generated an executable Python function for reverse validation [Initial Verification Code] of this problem. However, the verification code did not pass when verifying a solution [Solution] of the problem.\n\nGiven a problem, a corresponding solution, and feedback from a verification function.\n\nHowever, the solution has not been validated by the verification function. This means that either the solution or the verification function, or both, contain errors.\n\nYou need to carefully analyze and complete the following tasks:\n1. Reflect on the initial solution process:\n   - Compare in detail the solution and the feedback, examine the logic and accuracy of the solution approach **step by step**, consider whether the errors lie with the feedback or with the solution.\n   - If errors or inadequacies are found, provide detailed feedback and suggestions for improvement\n   - If it is found that there are no errors in the solution but rather errors exist in the feedback, state that the solution is correct.\n2. Provide a new solution:\n   - If the initial solution is correct, please use the original solution process (Note: If the verification code is correct, you also need to repeat the initial verification code word for word)\n   - If errors or inadequacies are found, revise the solution based on your reflection.\n\nNote: \n- You cannot refuse to generate a new solution due to missing information or other reasons.\n\nPlease strictly output your analysis and revision according to the following format, without any additional content:\n\n[Reflection]\n{{Detailed reflection process}}\n\n[New Solution]\n{{Complete solution process based on the reflection}}",
    "input_case_1": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\nEvaluate $\\sqrt{5\\cdot10\\cdot14\\cdot21\\cdot15\\cdot20}$.\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\nTo evaluate \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\), we start by simplifying the expression inside the square root. We can factor each number into its prime factors:\n\n\\[\n\begin{align*}\n5 &= 5, \\\n10 &= 2 \\cdot 5, \\\n14 &= 2 \\cdot 7, \\\n21 &= 3 \\cdot 7, \\\n15 &= 3 \\cdot 5, \\\n20 &= 2^2 \\cdot 5.\n\\end{align*}\n\\]\n\nNow, substitute these factorizations into the product:\n\n\\[\n5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20 = 5 \\cdot (2 \\cdot 5) \\cdot (2 \\cdot 7) \\cdot (3 \\cdot 7) \\cdot (3 \\cdot 5) \\cdot (2^2 \\cdot 5).\n\\]\n\nCombine all the factors:\n\n\\[\n= 2^3 \\cdot 3^2 \\cdot 5^4 \\cdot 7^2.\n\\]\n\nNow, take the square root of the entire expression:\n\n\\[\n\\sqrt{2^3 \\cdot 3^2 \\cdot 5^4 \\cdot 7^2}.\n\\]\n\nThe square root of a product is the product of the square roots:\n\n\\[\n= \\sqrt{2^3} \\cdot \\sqrt{3^2} \\cdot \\sqrt{5^4} \\cdot \\sqrt{7^2}.\n\\]\n\nSimplify each square root:\n\n\\[\n\\sqrt{2^3} = \\sqrt{2^2 \\cdot 2} = 2 \\sqrt{2},\n\\]\n\\[\n\\sqrt{3^2} = 3,\n\\]\n\\[\n\\sqrt{5^4} = 5^2 = 25,\n\\]\n\\[\n\\sqrt{7^2} = 7.\n\\]\n\nCombine these results:\n\n\\[\n= (2 \\sqrt{2}) \\cdot 3 \\cdot 25 \\cdot 7.\n\\]\n\nCalculate the product of the integers:\n\n\\[\n3 \\cdot 25 = 75,\n\\]\n\\[\n75 \\cdot 7 = 525,\n\\]\n\\[\n2 \\cdot 525 = 1050.\n\\]\n\nThus, the expression simplifies to:\n\n\\[\n1050 \\sqrt{2}.\n\\]\n\nTherefore, the value of \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\) is \\(\boxed{1050 \\sqrt{2}}\\).\n\n[Feedback]\nThe problem involves evaluating the expression \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\).\n\n1. **Calculate the Product of the Numbers:**\n   - The product of the numbers is calculated as \\(5 \times 10 \times 14 \times 21 \times 15 \times 20\\).\n   - Using a computational tool, the product is found to be 4,410,000.\n\n2. **Square the Given Answer:**\n   - The predicted answer is \\(1050\\sqrt{2}\\).\n   - To verify, we square this expression: \\((1050\\sqrt{2})^2 = 1050^2 \times 2\\).\n   - Using a computational tool, the squared value is calculated to be 2,205,000.\n\n3. **Verification:**\n   - Compare the squared answer (2,205,000) with the calculated product (4,410,000).\n   - Since 2,205,000 does not equal 4,410,000, the verification fails.\n\n**Conclusion:**\nThe predicted answer \\(1050\\sqrt{2}\\) is incorrect because its square does not match the calculated product of the numbers. The verification process returns a result of \"False.\"",
    "output_case_1": "[Reflection]\nThe initial solution attempts to simplify the expression \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\) by factoring each number into its prime factors and then taking the square root of the product. However, there is a mistake in the calculation process:\n\n1. The initial solution correctly factors each number into its prime components.\n2. The combination of these factors is also correct: \\(2^3 \\cdot 3^2 \\cdot 5^4 \\cdot 7^2\\).\n3. The error occurs in the simplification of the square root. The square root of a product of powers should be simplified by taking half of each exponent:\n   - \\(\\sqrt{2^3} = 2^{3/2} = 2\\sqrt{2}\\)\n   - \\(\\sqrt{3^2} = 3\\)\n   - \\(\\sqrt{5^4} = 5^2 = 25\\)\n   - \\(\\sqrt{7^2} = 7\\)\n\n4. The multiplication of these simplified terms is incorrect. The correct multiplication should be:\n   - \\(2\\sqrt{2} \\cdot 3 \\cdot 25 \\cdot 7 = 1050\\sqrt{2}\\), but the verification shows this is incorrect.\n\nThe feedback correctly identifies that the squared result of \\(1050\\sqrt{2}\\) does not match the product of the numbers, indicating an error in the initial solution.\n\n[New Solution]\nTo correctly evaluate \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\), we should:\n\n1. Calculate the product of the numbers directly:\n   \\[\n   5 \times 10 \times 14 \times 21 \times 15 \times 20 = 4,410,000\n   \\]\n\n2. Take the square root of the product:\n   \\[\n   \\sqrt{4,410,000}\n   \\]\n\n3. Simplify the square root:\n   - Factor the product into its prime factors:\n     \\[\n     4,410,000 = 2^3 \times 3^2 \times 5^4 \times 7^2\n     \\]\n   - Take the square root:\n     \\[\n     \\sqrt{2^3 \times 3^2 \times 5^4 \times 7^2} = 2^{3/2} \times 3 \times 5^2 \times 7 = 2\\sqrt{2} \times 3 \times 25 \times 7\n     \\]\n\n4. Calculate the product:\n   \\[\n   2 \times 3 \times 25 \times 7 = 1050\n   \\]\n\n5. The correct answer is:\n   \\[\n   \boxed{2100}\n   \\]\n\nThus, the value of \\(\\sqrt{5 \\cdot 10 \\cdot 14 \\cdot 21 \\cdot 15 \\cdot 20}\\) is \\(\boxed{2100}\\).",
    "input_case_2": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\nWhat is the perimeter of trapezoid $ABCD$? [asy]\nsize(3inch, 1.5inch);\npair a=(0,0), b=(18,24), c=(68,24), d=(75,0), f=(68,0), e=(18,0);\ndraw(a--b--c--d--cycle);\ndraw(b--e);\ndraw(shift(0,2)*e--shift(2,2)*e--shift(2,0)*e);\nlabel(\"30\", (9,12), W);\nlabel(\"50\", (43,24), N);\nlabel(\"25\", (71.5, 12), E);\nlabel(\"24\", (18, 12), E);\nlabel(\"$A$\", a, SW);\nlabel(\"$B$\", b, N);\nlabel(\"$C$\", c, N);\nlabel(\"$D$\", d, SE);\nlabel(\"$E$\", e, S);\n[/asy]\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\nTo find the perimeter of trapezoid \\(ABCD\\), we need to determine the lengths of all its sides: \\(AB\\), \\(BC\\), \\(CD\\), and \\(DA\\).\n\nGiven:\n- \\(AB = 30\\)\n- \\(BC = 50\\)\n- \\(CD = 25\\)\n- \\(BE = 24\\)\n\nSince \\(BE\\) is perpendicular to \\(AE\\) and \\(ED\\), \\(AE\\) and \\(ED\\) are horizontal segments. We can find \\(AE\\) and \\(ED\\) using the Pythagorean theorem.\n\nFirst, calculate \\(AE\\):\n\\[\nAE = \\sqrt{AB^2 - BE^2} = \\sqrt{30^2 - 24^2} = \\sqrt{900 - 576} = \\sqrt{324} = 18\n\\]\n\nNext, calculate \\(ED\\):\nSince \\(ED\\) is the remaining horizontal segment from \\(E\\) to \\(D\\), and \\(AD = AE + ED\\), we need to find \\(AD\\) first.\n\nThe total horizontal distance from \\(A\\) to \\(D\\) is the sum of \\(AE\\) and \\(ED\\). Since \\(AD\\) is the base of the trapezoid, we can use the fact that \\(AD = AE + ED\\).\n\nNow, calculate \\(AD\\):\n\\[\nAD = \\sqrt{CD^2 - BC^2} = \\sqrt{25^2 - 50^2} = \\sqrt{625 - 2500} = \\sqrt{0} = 0\n\\]\n\nThis calculation seems incorrect because \\(AD\\) cannot be zero. Let's re-evaluate the problem:\n\nSince \\(BE\\) is perpendicular to \\(AE\\) and \\(ED\\), and \\(BE = 24\\), the horizontal distance from \\(A\\) to \\(D\\) is the sum of \\(AE\\) and \\(ED\\).\n\nGiven that \\(AE = 18\\), and the total horizontal distance from \\(A\\) to \\(D\\) is \\(75\\) (as \\(D\\) is directly above \\(A\\) on the x-axis), we have:\n\\[\nED = 75 - 18 = 57\n\\]\n\nNow, calculate the perimeter of trapezoid \\(ABCD\\):\n\\[\n\text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75 = 180\n\\]\n\nThus, the perimeter of trapezoid \\(ABCD\\) is \\(\boxed{180}\\).",
    "output_case_2": "[Reflection]\nThe initial solution attempts to calculate the perimeter of trapezoid \\(ABCD\\) by summing up its sides. However, there is a fundamental misunderstanding about the dimensions and configuration of the trapezoid in both the solution and the feedback provided.\n\nUpon carefully reviewing the problem, it appears there were some confusions around the naming and use of the lengths. The mistake lies in both interpreting the problem description and illustrating the steps of calculations. Here's a detailed assessment of what went wrong:\n\n1. The initial solution incorrectly sought the length of \\(AD\\) and \\(ED\\) through a misapplied Pythagorean theorem calculation, which led to an error.\n2. There seemed also to be a misunderstanding about the position of some of the sides when delineating the perimeter.\n3. The feedback, although criticizing the solution, still miscalculated the perimeter with the wrong side lenghs.\n\nWe must reassess and correctly identify all lengths, then simply sum them to find the perimeter of trapezoid \\(ABCD\\).\n\n[New Solution]\n1. Identify each side of the trapezoid as clearly described by the problem:\n    - \\(AB = 30\\) (as indicated in the labels)\n    - \\(BC = 50\\)\n    - \\(CD = 25\\)\n    - \\(DA = 75\\) (since \\(AD = 18 + 57 = 75\\) from the corrected understanding of horizontal arrangement.)\n\n2. Sum all sides to calculate the perimeter:\n   \\[\n   \text{Perimeter} = AB + BC + CD + DA = 30 + 50 + 25 + 75\n   \\]\n   \\[\n   = 180\n   \\]\n\nThus, the correctly determined perimeter of trapezoid \\(ABCD\\) is \\(\boxed{180}\\).",
    "input_template": "Reflect on the initial solution and provide a new solution.\n\n[Problem]\n{query}\n\n[Initial Solution]\n(Remember, feedback is not always be correct.)\n{old_response}\n\n[Feedback]\n{feedback}\n"
}

class Refiner:
    def __init__(self,model,query):
        self.init_messages=[
            {"role":"system","content":refine_dict["system_prompt"]},
            {"role":"user","content":refine_dict["input_case_1"]},
            {"role":"assistant","content":refine_dict["output_case_1"]},
            {"role":"user","content":refine_dict["input_case_2"]},
            {"role":"assistant","content":refine_dict["output_case_2"]},
        ]
        self.template = refine_dict["input_template"]
        self.model = model
        self.query = query
        self.temperature=0
    
    def extract_new_solution(self,new_response):
        if "[New Solution]" not in new_response:
            return "",new_response
        else:
            return new_response.split("[New Solution]")[0].strip(),new_response.split("[New Solution]")[1].strip()

    def refine(self, old_response,feedback):
        input_context = self.template.format(query=self.query, old_response=old_response,feedback=feedback)
        messages=self.init_messages+[{"role":"user","content":input_context}]
        # refine
        response=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        new_response = response.strip()
        feedback,new_response=self.extract_new_solution(new_response) 
        return feedback,new_response

# ------------------------------------------------------------------------------------------------------------------------
# Contrast and Regenerate

contrast_response_dict= {
    "system_prompt": "You are an expert at comparing and extracting key points.\n\nTask: Analyze the differences between two solutions and extract key points\n\nBackground: For the same problem, two solutions have provided different answers. We need to analyze these differences in depth to identify the key aspects of the problem.\n\nSteps:\n1. Carefully read the problem and both solutions\n2. Ignore surface differences in expression, focus on substantial differences in content and method\n3. Compare the core ideas, key steps, and final results of both solutions\n4. Summarize the essence of the problem reflected by these differences and the key points to note when solving\n\nOutput requirements:\n1. Concisely list 1-3 key points\n2. Each point should be specific and helpful for regenerating a better solution\n\nOutput format:\n[Comparative Analysis Process]\n{{Your comparative analysis process}}\n\n[Core Differences in Solutions]\n{{Summarize the differences in solutions based on the comparative analysis process, answer in bullet points}}\n\n[Key Points to Note When Solving the Problem]\n{{Summarize the key points to note when solving the problem based on the differences in solutions, answer in bullet points}}",
    "input_template": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\n{query}\n\n[Solution 1]\n{response_1}\n\n[Solution 2]\n{response_2}\n\n",
    "input_case_1": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nEvaluate $\\sqrt{3\\cdot6\\cdot10\\cdot15\\cdot21\\cdot28}$.\n\n[Solution 1]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each of the numbers:\n\n- \\(3\\) is already a prime number.\n- \\(6 = 2 \\cdot 3\\)\n- \\(10 = 2 \\cdot 5\\)\n- \\(15 = 3 \\cdot 5\\)\n- \\(21 = 3 \\cdot 7\\)\n- \\(28 = 2^2 \\cdot 7\\)\n\nNow, let's write the product \\(3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28\\) using these factorizations:\n\n\\[\n3 \\cdot (2 \\cdot 3) \\cdot (2 \\cdot 5) \\cdot (3 \\cdot 5) \\cdot (3 \\cdot 7) \\cdot (2^2 \\cdot 7)\n\\]\n\nCombine the factors:\n\n- The factor of \\(2\\) appears \\(1 + 1 + 2 = 4\\) times.\n- The factor of \\(3\\) appears \\(1 + 1 + 1 + 1 = 4\\) times.\n- The factor of \\(5\\) appears \\(1 + 1 = 2\\) times.\n- The factor of \\(7\\) appears \\(1 + 1 = 2\\) times.\n\nThus, the product can be rewritten as:\n\n\\[\n2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2\n\\]\n\nNow, take the square root of the entire expression:\n\n\\[\n\\sqrt{2^4 \\cdot 3^4 \\cdot 5^2 \\cdot 7^2} = \\sqrt{(2^2 \\cdot 3^2 \\cdot 5 \\cdot 7)^2}\n\\]\n\nThe square root of a square is the base itself:\n\n\\[\n2^2 \\cdot 3^2 \\cdot 5 \\cdot 7 = 4 \\cdot 9 \\cdot 5 \\cdot 7\n\\]\n\nCalculate the product:\n\n- \\(4 \\cdot 9 = 36\\)\n- \\(36 \\cdot 5 = 180\\)\n- \\(180 \\cdot 7 = 1260\\)\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{1260}\\).\n\n[Solution 2]\nTo evaluate \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\), we start by factoring each number into its prime factors:\n\n\\[\n3 = 3\n\\]\n\n\\[\n6 = 2 \\times 3\n\\]\n\n\\[\n10 = 2 \\times 5\n\\]\n\n\\[\n15 = 3 \\times 5\n\\]\n\n\\[\n21 = 3 \\times 7\n\\]\n\n\\[\n28 = 2^2 \\times 7\n\\]\n\nNow, we multiply all these factorizations together:\n\n\\[\n3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28 = 3 \\times (2 \\times 3) \\times (2 \\times 5) \\times (3 \\times 5) \\times (3 \\times 7) \\times (2^2 \\times 7)\n\\]\n\nCombine the factors:\n\n\\[\n= 2^3 \\times 3^4 \\times 5^2 \\times 7^2\n\\]\n\nNow, take the square root of the product:\n\n\\[\n\\sqrt{2^3 \\times 3^4 \\times 5^2 \\times 7^2} = \\sqrt{2^3} \\times \\sqrt{3^4} \\times \\sqrt{5^2} \\times \\sqrt{7^2}\n\\]\n\nCalculate each square root:\n\n\\[\n\\sqrt{2^3} = \\sqrt{8} = 2^{3/2} = 2 \\sqrt{2}\n\\]\n\n\\[\n\\sqrt{3^4} = \\sqrt{81} = 3^2 = 9\n\\]\n\n\\[\n\\sqrt{5^2} = 5\n\\]\n\n\\[\n\\sqrt{7^2} = 7\n\\]\n\nCombine these results:\n\n\\[\n\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28} = (2 \\sqrt{2}) \\times 9 \\times 5 \\times 7\n\\]\n\nCalculate the product of the integers:\n\n\\[\n9 \\times 5 = 45\n\\]\n\n\\[\n45 \\times 7 = 315\n\\]\n\nThus, the expression becomes:\n\n\\[\n2 \\sqrt{2} \\times 315 = 630 \\sqrt{2}\n\\]\n\nTherefore, the value of \\(\\sqrt{3 \\cdot 6 \\cdot 10 \\cdot 15 \\cdot 21 \\cdot 28}\\) is \\(\\boxed{630}\\).",
    "output_case_1": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly factorized each number into its prime components.\n2. Both solutions attempted to simplify the expression under the square root by combining the prime factors.\n3. The key difference lies in the handling of the square root:\n   - Solution 1 correctly identified that the product under the square root is a perfect square and simplified it directly to an integer.\n   - Solution 2 incorrectly calculated the square root of \\(2^3\\) as \\(2\\sqrt{2}\\) instead of simplifying the entire expression as a perfect square.\n\n[Core Differences in Solutions]\n1. Handling of the square root:\n   - Solution 1: Correctly identified the expression as a perfect square and simplified it to an integer.\n   - Solution 2: Incorrectly handled the square root of \\(2^3\\) and did not recognize the entire expression as a perfect square.\n2. Final result:\n   - Solution 1: Correctly calculated the result as 1260.\n   - Solution 2: Incorrectly calculated the result as \\(630\\sqrt{2}\\).\n\n[Key Points to Note When Solving the Problem]\n1. Verify whether the expression under the square root is a perfect square by checking the exponents of all prime factors.\n2. Simplify the expression under the square root completely before calculating the final result to ensure accuracy.\n3. Recognize that if the product of factors results in a perfect square, the square root should yield an integer, not an expression involving radicals.",
    "input_case_2": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nThe trisectors of angles $B$ and $C$ of scalene triangle $ABC$ meet at points $P$ and $Q$, as shown. Angle $A$ measures 39 degrees and angle $QBP$ measures 14 degrees. What is the measure of angle $BPC$?\n\n[asy]unitsize(2cm);\nlabel(\"$B$\",(0,0),W);\n\nlabel(\"$A$\",(1.2,1.5),N);\nlabel(\"$C$\",(1,0),E);\nlabel(\"$Q$\",(.8,.6),N);\nlabel(\"$P$\",(.7,.2),N);\n\ndraw((0,0)--(1.2,1.5)--(1,0)--cycle,linewidth(1));\ndraw((0,0)--(.8,.6)--(1,0),linewidth(1));\ndraw((0,0)--(.7,.2)--(1,0),linewidth(1));\n[/asy]\n\n[Solution 1]\nTo solve for the measure of angle \\( BPC \\), we start by analyzing the given information about the triangle and the angles.\n\nWe know:\n- \\( \\angle A = 39^\\circ \\).\n- \\( \\angle QBP = 14^\\circ \\).\n\nSince \\( \\triangle ABC \\) is a triangle, the sum of its angles is \\( 180^\\circ \\). Therefore, we can find the sum of angles \\( B \\) and \\( C \\):\n\n\\[\n\\angle B + \\angle C = 180^\\circ - \\angle A = 180^\\circ - 39^\\circ = 141^\\circ.\n\\]\n\nThe problem involves trisectors of angles \\( B \\) and \\( C \\). Let the angles at \\( B \\) be trisected into \\( x, x, x \\) and at \\( C \\) into \\( y, y, y \\). Thus, we have:\n\n\\[\n3x = \\angle B \\quad \\text{and} \\quad 3y = \\angle C.\n\\]\n\nFrom the equation \\( \\angle B + \\angle C = 141^\\circ \\), we have:\n\n\\[\n3x + 3y = 141^\\circ \\quad \\Rightarrow \\quad x + y = 47^\\circ.\n\\]\n\nThe angle \\( \\angle QBP = 14^\\circ \\) is given. Since \\( Q \\) and \\( P \\) are points on the trisectors, \\( \\angle QBP \\) is the angle between the first trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, we have:\n\n\\[\n\\angle QBP = x + y = 47^\\circ.\n\\]\n\nHowever, this contradicts the given \\( \\angle QBP = 14^\\circ \\). Therefore, we need to reconsider the trisector configuration. The correct interpretation is that \\( \\angle QBP = x + y - 33^\\circ \\) (since \\( x + y = 47^\\circ \\) and \\( \\angle QBP = 14^\\circ \\)).\n\nNow, we need to find \\( \\angle BPC \\). The angle \\( \\angle BPC \\) is the angle between the second trisector of \\( \\angle B \\) and the second trisector of \\( \\angle C \\). Therefore, it is:\n\n\\[\n\\angle BPC = 180^\\circ - (x + y) = 180^\\circ - 47^\\circ = 133^\\circ.\n\\]\n\nThus, the measure of angle \\( BPC \\) is \\(\\boxed{133^\\circ}\\).\n\n[Solution 2]\nTo solve this problem, we need to find the measure of angle \\( BPC \\) in the given triangle configuration. We are given that angle \\( A \\) measures 39 degrees and angle \\( QBP \\) measures 14 degrees. The points \\( P \\) and \\( Q \\) are the intersections of the trisectors of angles \\( B \\) and \\( C \\).\n\nFirst, let's denote the angles of triangle \\( ABC \\) as follows:\n- \\( \\angle A = 39^\\circ \\)\n- Let \\( \\angle B = x \\)\n- Let \\( \\angle C = y \\)\n\nSince the sum of angles in a triangle is 180 degrees, we have:\n\\[\nx + y + 39^\\circ = 180^\\circ\n\\]\n\\[\nx + y = 141^\\circ\n\\]\n\nNext, consider the trisectors of angles \\( B \\) and \\( C \\). Since \\( P \\) and \\( Q \\) are points where the trisectors meet, we can express the angles at \\( B \\) and \\( C \\) in terms of their trisectors:\n- The trisectors of \\( \\angle B \\) divide it into three equal parts, so each part is \\( \\frac{x}{3} \\).\n- The trisectors of \\( \\angle C \\) divide it into three equal parts, so each part is \\( \\frac{y}{3} \\).\n\nThe angle \\( QBP \\) is given as 14 degrees. This angle is formed by one trisector of \\( \\angle B \\) and one trisector of \\( \\angle C \\). Therefore, we can express \\( \\angle QBP \\) as:\n\\[\n\\angle QBP = \\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, we have two equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nFrom the second equation, multiply through by 3 to clear the fractions:\n\\[\nx + y = 42^\\circ\n\\]\n\nHowever, this seems inconsistent with the first equation. Let's re-evaluate the setup. The correct interpretation should be:\n- The angle \\( QBP \\) is actually the angle between the trisectors, not the sum of the trisectors.\n\nThus, the correct setup should be:\n\\[\n\\frac{x}{3} + \\frac{y}{3} = 14^\\circ\n\\]\n\nNow, solve the system of equations:\n1. \\( x + y = 141^\\circ \\)\n2. \\( \\frac{x}{3} + \\frac{y}{3} = 14^\\circ \\)\n\nMultiply the second equation by 3:\n\\[\nx + y = 42^\\circ\n\\]\n\nThis is incorrect, so let's correct the interpretation:\n- The angle \\( QBP \\) is the angle between the trisectors, not the sum of the trisectors.\n\nRe-evaluate:\n- The angle \\( BPC \\) is the external angle formed by the trisectors.\n\nThe correct interpretation is:\n- The angle \\( BPC \\) is the external angle to \\( \\angle QBP \\).\n\nThus, the measure of angle \\( BPC \\) is:\n\\[\n180^\\circ - 14^\\circ = 166^\\circ\n\\]\n\nTherefore, the measure of angle \\( BPC \\) is \\(\\boxed{166^\\circ}\\).",
    "output_case_2": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Both solutions correctly identified the given angles and the relationship between the angles in triangle \\(ABC\\).\n2. Both solutions attempted to use the properties of trisectors to find the measure of angle \\(BPC\\).\n3. The key difference lies in the interpretation and calculation of the angles formed by the trisectors:\n   - Solution 1 attempted to use the sum of the trisected angles and derived a contradiction, leading to a reevaluation and a final angle of \\(133^\\circ\\).\n   - Solution 2 initially misinterpreted the relationship between the trisectors and the given angle \\(QBP\\), leading to an incorrect intermediate step, but eventually corrected the interpretation to find the external angle \\(BPC\\) as \\(166^\\circ\\).\n\n[Core Differences in Solutions]\n1. Interpretation of angle \\(QBP\\):\n   - Solution 1: Initially assumed \\(QBP\\) was the sum of trisected angles, leading to a contradiction and reevaluation.\n   - Solution 2: Initially miscalculated the sum of trisected angles, but corrected to find the external angle.\n2. Calculation of angle \\(BPC\\):\n   - Solution 1: Concluded \\(BPC\\) as \\(133^\\circ\\) after reevaluation.\n   - Solution 2: Concluded \\(BPC\\) as \\(166^\\circ\\) by considering it as an external angle.\n\n[Key Points to Note When Solving the Problem]\n1. Carefully interpret the geometric configuration and relationships between angles, especially when dealing with trisectors or other subdivisions.\n2. Verify the consistency of derived equations with given conditions to avoid contradictions.\n3. Consider alternative geometric interpretations, such as external angles, when initial calculations lead to inconsistencies.",
    "input_case_3": "Based on the following information, please analyze the differences and extract key points:\n(The key points should be detailed, not vague.)\n\n[Problem]\nWhich of the following is closest to 67.4?\n\nA.      67.332\nB.      67.473\nC.      67.526\nD.      67.445\nE.      67.346\n\nExpress your answer as A, B, C, D, or E.\n\n[Solution 1]\nB. 67.473\n\n[Solution 2]\nTo determine which option is closest to 67.4, we need to calculate the absolute difference between 67.4 and each of the given options. Let's do that:\n\nA. |67.4 - 67.332| = 0.068\nB. |67.4 - 67.473| = 0.073\nC. |67.4 - 67.526| = 0.126\nD. |67.4 - 67.445| = 0.045\nE. |67.4 - 67.346| = 0.054\n\nNow, let's compare these differences:\n\n- A: 0.068\n- B: 0.073\n- C: 0.126\n- D: 0.045\n- E: 0.054\n\nThe smallest difference is 0.045, which corresponds to option D (67.445).\n\nTherefore, the option closest to 67.4 is D.",
    "output_case_3": "[Comparative Analysis Process]\nI analyzed both solutions to identify the differences in their approaches and calculations:\n\n1. Solution 1 directly selected option B without showing any calculations or reasoning.\n2. Solution 2 provided a detailed calculation of the absolute differences between 67.4 and each of the given options, then identified the smallest difference to determine the closest value.\n\n[Core Differences in Solutions]\n1. Methodology:\n- Solution 1: Provided an answer without any supporting calculations or reasoning.\n- Solution 2: Used a systematic approach by calculating the absolute differences and comparing them to find the closest value.\n2. Accuracy:\n- Solution 1: Incorrectly identified option B as the closest value.\n- Solution 2: Correctly identified option D as the closest value based on the smallest absolute difference.\n\n[Key Points to Note When Solving the Problem]\n1. Always calculate the absolute differences when determining which number is closest to a given value to ensure accuracy.\n2. Compare the calculated differences to identify the smallest one, which indicates the closest value.\n3. Provide clear reasoning and calculations to support the chosen answer, ensuring transparency and correctness.",
    "gen_response_with_insights_template": "Answer the following question:\n\n[Problem]\n{query}\n\n[Key Points to Note When Solving the Problem]\n{insights}\n\n[Solution]\n{{Your solution}}\n"
}

class ContrastResponse:
    def __init__(self,model,query,temperature=0) -> None:
        self.init_messages=[
            {"role":"system","content":contrast_response_dict["system_prompt"]},
            {"role":"user","content":contrast_response_dict["input_case_1"]},
            {"role":"assistant","content":contrast_response_dict["output_case_1"]},
            {"role":"user","content":contrast_response_dict["input_case_2"]},
            {"role":"assistant","content":contrast_response_dict["output_case_2"]},
            {"role":"user","content":contrast_response_dict["input_case_3"]},
            {"role":"assistant","content":contrast_response_dict["output_case_3"]},
        ]
        self.contrast_response_template=contrast_response_dict["input_template"]
        self.gen_response_with_insights_template=contrast_response_dict["gen_response_with_insights_template"]
        self.model=model
        self.temperature=temperature
        self.query=query
        self.memory=[]

    def extract_insights(self,output):
        if "[Key Points to Note When Solving the Problem]" in output:
            insights=output.split("[Key Points to Note When Solving the Problem]")[-1].strip()
        else:
            insights=output
        pattern = r'\d+\.\s*(.*)'
        matches = re.findall(pattern, insights)
        self.memory = []  
        for match in matches:
            self.memory.append(match)

    def contrast_response(self,response_1,response_2):
        responses=[response_1,response_2]
        random.shuffle(responses)
        input_content=self.contrast_response_template.format(query=self.query,response_1=responses[0],response_2=responses[1])
        messages=self.init_messages+[{"role":"user","content":input_content}]
        output=call_llm(model=self.model,messages=messages,temperature=self.temperature)
        self.extract_insights(output)
        return output

    def gen_response_with_insights(self):
        insights_str=""
        for i,insight in enumerate(self.memory):
            insights_str+=f"{i+1}. {insight}\n"
        input_content=self.gen_response_with_insights_template.format(query=self.query,insights=insights_str)
        new_response=call_llm(model=self.model,messages=[{"role":"user","content":input_content}],temperature=self.temperature)
        new_response=new_response.strip()
        return new_response
    
    def __call__(self,response_1,response_2):
        contrast_result=self.contrast_response(response_1,response_2)
        new_response=self.gen_response_with_insights()
        return contrast_result,new_response

# ------------------------------------------------------------------------------------------------------------------------
# Dataset

def read_dataset(start,end,shuffle=False):
    def format_sample(id,query,sample):
        return {"id":id,"query":query,"sample":sample}
    data=read_file("MATH/data/all_random_500.jsonl")
    data=[format_sample(id=dp["id"],query=dp["query"],sample=dp) for dp in data]
    if shuffle:
        random.seed(0)
        random.shuffle(data)
    if end==-1:
        return data[start:]
    else:
        return data[start:end]


# ------------------------------------------------------------------------------------------------------------------------
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
        self.llm_code_executor = LLMCodeExecutor(model=self.model,temperature=self.temperature)
        self.feedbacker = Feebacker(model=args.model,temperature=self.temperature)
        self.logger=Logger(args)
        # Reuse the original response
        self.first_response_data=read_file(f"MATH/logs/{self.model}/all_random_500_first_response.jsonl")
        self.first_response_data={item["id"]: item for item in self.first_response_data}


    def inference(self,sample):
        cur_turn=0
        query=sample["query"]
        self.response_refiner=Refiner(model=self.model,query=query)
        self.code_refiner=CodeRefiner(model=self.model,query=query,temperatue=self.temperature)
        self.contraster=ContrastResponse(model=self.model,query=query,temperature=self.temperature)
        old_result=None
        old_response=None
        # Initialize the sample log
        self.logger.init_sample(sample_id=sample["id"],query=query)
        while cur_turn < self.max_cur_turn+1:
            print(f"cur_turn:{cur_turn}")
            print("start get response")
            if cur_turn==0:
                # Get the initial response
                # response=get_first_response(model=self.model,query=query,temperature=self.temperature)
                # Reuse the response from the first round previously (to save the cost of re-inferencing)
                response=self.first_response_data[sample["id"]]['response']
                result=self.first_response_data[sample["id"]]['pred']
                # Generate the verification code
                verify_code_generator=VerifyCodeGenerator(model=self.model,temperature=self.temperature)
                verify_code=verify_code_generator.gen_verify_code(query=query)
            else:
                old_response=response
                old_result=result
                response=new_response
                verify_code=new_verify_code
                result=extract_answer(query,response)
            if cur_turn>0:
                if result!=old_result:
                    # contrast insights
                    contrast_result,response=self.contraster(old_response,new_response)
                    result=extract_answer(query,response)
            # Perform the verification
            print(f"start execute verify code")
            execute_result,execute_content=self.llm_code_executor.execute(query=query,response=response,result=result,validate_response_fuc=verify_code)
            if execute_result:  # Verification passed
                feedback="verify passed"
                # log
                self.logger.update_turn(cur_turn,verify_code,response,result,execute_result,execute_content,feedback,reflection="",code_feedback="")
                cur_turn+=1
                break
            else:  # Verification not passed
                # Generate feedback and refine
                print(f"start refine")
                feedback_summary=self.feedbacker(query=query,response=response,execute_code=verify_code,execute_result=execute_result,execute_content=execute_content)
                reflection,new_response=self.response_refiner.refine(old_response=response,feedback=feedback_summary)
                # code refine
                code_feedback,new_verify_code=self.code_refiner.refine(response=response,verify_code=verify_code,verify_process=execute_content)
                # log
                self.logger.update_turn(cur_turn,verify_code,response,result,execute_result,execute_content,feedback_summary,reflection,code_feedback)
                cur_turn+=1
        final_response=response
        final_result=result
        self.logger.update_final_response(final_response,final_result)
        self.logger.update_insights(self.contraster.memory)

# ------------------------------------------------------------------------------------------------------------------------
# Main Fuc

def main_fuc(args):
    agent=ProgCoAgent(args)
    data=read_dataset(start=args.start,end=args.end,shuffle=False)
    print("dataset size: ",len(data))
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
    parser.add_argument('--dataset', type=str, default="MATH")
    parser.add_argument('--temperature', type=float, default=0.0)
    args = parser.parse_args()
    args.experiment_prefix = f"progco_start-{args.start}_end-{args.end}"
    print(vars(args))  # Print all parameters
    return args
                

if __name__ == "__main__":
    args = parse_args()
    main_fuc(args)