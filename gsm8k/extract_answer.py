from utils.call_llm import call_llm

extract_prompt="Your task is to extract the final numerical answer from the given problem and solution process. Please carefully read the problem and solution process to find the final calculation result or answer.\\nPlease note that you only need to extract the final numerical answer, without including units or other textual explanations.\n\nImportant notes:\n1. Only extract the final answer, without units or other textual explanations.\n2. The answer sometimes begins with \"Answer:\" or is wrapped in \\boxed{{}}.\n3. Do not output any additional content, such as \\boxed{{}} wrapping, etc.\n4. For percentage answers, only extract the numeric prefix.\n\n\nHere are some output examples:\n10, -1.5, 20, 3, 10.8\n\nHere is the problem:\n{query}\n\nHere is the solution:\n{response}\n\nAgain, ensure that your output contains only the final answer, without any additional explanations or formatting."

def extract_answer(query,response):
    prompt = extract_prompt.format(query=query,response=response)
    messages=[
        {"role": "user", "content": prompt},
    ]
    result = call_llm(model="gpt-4o-mini-0718",messages=messages,temperature=0)
    result=result.strip()
    return result