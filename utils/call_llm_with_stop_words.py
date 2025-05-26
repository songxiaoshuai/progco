"""
Stream call to OpenAI API and determine whether to stop through string matching
Note: The stop parameter in OpenAI API does not work with Claude model
https://github.com/MacPaw/OpenAI/issues/198 
https://community.openai.com/t/interrupting-completion-stream-in-python/30628/7
"""
import time
import os
from openai import OpenAI
from typing import Optional, List

api_key=os.environ["OPENAI_API_KEY"]
base_url=os.environ["OPENAI_BASE_URL"]

client = OpenAI(
    api_key=api_key,
    base_url=base_url,
)

def stream_openai(model,messages,temperature,stop_words,max_tokens):
    accumulated_text = ""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=temperature,
        max_tokens=max_tokens
    )
    is_stop=False
    gen_stop_word=None
    try:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                accumulated_text += content
                for stop_word in stop_words:
                    if stop_word in accumulated_text:
                        is_stop=True
                        gen_stop_word=stop_word
                        accumulated_text = accumulated_text.split(gen_stop_word)[0]+gen_stop_word
                        break 
                if is_stop:
                    break
    finally:
        response.close()
    return accumulated_text,is_stop,gen_stop_word

def call_llm_with_stop_words(
            model: str, 
            messages: List[dict],
            temperature: float = None, 
            stop_strs: Optional[List[str]] = None,
            max_tokens: int = None):
    if model in ['gpt-3.5-turbo','gpt-3.5-16K','gpt-4o-0806','gpt-4o-mini-0718']:
        retries = 0
        max_retries = 10
        while retries < max_retries:
            try:
                response,is_stop,gen_stop_word = stream_openai(model=model,messages=messages,temperature=temperature,stop_words=stop_strs,max_tokens=max_tokens)
                return response,is_stop,gen_stop_word
            except KeyboardInterrupt:
                print("Operation canceled by user.")
                return ''
            except Exception as e:
                print(f"Someting wrong:{e}. Retrying in {retries*10+10} seconds...")
                time.sleep(retries*10) 
                retries += 1
        return '',False,None
    else:
        raise

if __name__ == "__main__":
    MESSAGES = [
        {"role": "user", "content": "Write an essay about summer"},
    ]
    STOP_WORD = ["time","hot"]  # Set your stop words
    response,is_stop,gen_stop_word = call_llm_with_stop_words(model="gpt-4o-0806",messages=MESSAGES,temperature=0,stop_strs=STOP_WORD)
    print(response)
    print(is_stop)
    print(gen_stop_word)
