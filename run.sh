export PROJECT_PATH="progco_project_path"
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_api_base"

# IFEval
# 1. Generate Initial Inference Responses (We have provided the results of GPT-3.5 and GPT-4 in the code.)
# python ifeval/get_first_response.py --model=gpt-4o-0806
# 2. Execute ProgCo Main Program
# python ifeval/progco_infer_main.py --model=gpt-4o-0806 --max_cur_turn=3 --start=0 --end=-1
# 3. Evaluate Inference Results
# python ifeval/progco_eval_main.py --infered_file_path=ifeval/logs/gpt-4o-0806/infered/your_infered_file_name.json

# GSM8K
# 1. Generate Initial Inference Responses (We have provided the results of GPT-3.5 and GPT-4 in the code.)
# python gsm8k/get_first_response.py --model=gpt-4o-0806
# 2. Execute ProgCo Main Program
# python gsm8k/progco_infer_main.py --model=gpt-4o-0806 --max_cur_turn=3 --start=0 --end=-1
# 3. Evaluate Inference Results
# python gsm8k/progco_eval_main.py --infered_file_path=gsm8k/logs/gpt-4o-0806/infered/your_infered_file_name.json


# MATH
# 1. Generate Initial Inference Responses (We have provided the results of GPT-3.5 and GPT-4 in the code.)
# python MATH/get_first_response.py --model=gpt-4o-0806
# 2. Execute ProgCo Main Program
# python MATH/progco_infer_main.py --model=gpt-4o-0806 --max_cur_turn=3 --start=0 --end=-1
# 3. Evaluate Inference Results
# python MATH/progco_eval_main.py --infered_file_path=MATH/logs/your_model/infered/your_infered_file_name.json
