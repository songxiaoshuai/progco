import time
import os
from openai import OpenAI
from typing import Optional, List

api_key=os.environ["OPENAI_API_KEY"]
base_url=os.environ["OPENAI_BASE_URL"]

def huggingface_llm_inference(
                        model: str, 
                        messages: List[dict],
                        temperature: float = None, 
                        stop_strs: Optional[List[str]] = None,
                        max_tokens: int = None):
    raise NotImplementedError

def openai_llm_inference(
                        model: str, 
                        messages: List[dict],
                        temperature: float = None, 
                        stop_strs: Optional[List[str]] = None,
                        max_tokens: int = None):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        stop=stop_strs,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
            output=response.choices[0].message.content
            return output
        except KeyboardInterrupt:
            print("Operation canceled by user.")
            break
        except Exception as e:
            print(f"Someting wrong:{e}. Retrying in {retries*10+10} seconds...")
            time.sleep(retries*10) 
            # gpt-3.5-16k does not output certain data points in MATH dataset at temperature 0
            if model=='gpt-3.5-16K' and retries>3:
                temperature=0.5
            retries += 1
    return ''


def call_llm(
            model: str, 
            messages: List[dict],
            temperature: float = None, 
            stop_strs: Optional[List[str]] = None,
            max_tokens: int = None):
    if model in ['gpt-3.5-turbo','gpt-3.5-16K','gpt-4o-0806','gpt-4o-mini-0718']:
        if model=='gpt-4o-0806':
            model='gpt-4o-0806-global'
        response=openai_llm_inference(model,messages,temperature,stop_strs,max_tokens)
    else:
        response=huggingface_llm_inference(model,messages,temperature,stop_strs,max_tokens)
    return response

if __name__=="__main__":
    messages=[{'role': 'user', 'content': "write a essay about time"}]
    stop_strs=["time"]
    response=call_llm(model="gpt-3.5-16K",messages=messages,temperature=0,stop_strs=stop_strs)
    print(response)
    