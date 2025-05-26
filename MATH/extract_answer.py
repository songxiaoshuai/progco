"""
Extract predicted answers from LLM response
"""
from utils.call_llm import call_llm


# MATH
extract_prompt="Your task is to accurately extract the final answer from the given problem and solution process. Please carefully read the entire question and solution process to find the final answer.\n\nImportant notes:\n1. Only extract the final answer, without units or other textual explanations.\n2. Do not output any additional content, such as \\boxed{{}} wrapping, etc.\n3. For multiple answers, use list format for output.\n\nThe answers may take various forms, including but not limited to:\n\n1. Integers or decimals: 10, -1.5\n2. Fractions: \\frac{{1}}{{2}}, -\\frac{{3}}{{4}}\n3. Text answers: even, odd\n4. Square root expressions: \\sqrt{{10}}, \\sqrt[3]{{8}}\n5. Constants: π, e\n6. Algebraic expressions: a+1, y = x + 1\n7. Multiple choice options: A, B, C, D\n8. Matrices (using LaTeX format):\n   \\begin{{bmatrix}}\n   1 & 2 \\\\\n   3 & 4\n   \\end{{bmatrix}}\n\nHere is the problem:\n{query}\n\nHere is the solution:\n{response}\n\nAgain, ensure that your output contains only the final answer, without any additional explanations or formatting."

def extract_answer(query,response):
    prompt = extract_prompt.format(query=query,response=response)
    messages=[
        {"role": "user", "content": prompt},
    ]
    result = call_llm(model="gpt-4o-mini-0718",messages=messages,temperature=0)
    result=result.strip()
    return result

