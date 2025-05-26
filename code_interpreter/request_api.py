import requests

def request_code_interpreter(code):
    try:
        url = 'http://localhost:4200/api/code_interpreter'
        data = {'code': code}
        response = requests.post(url, json=data)
        result=response.json()['result'].replace("```","").replace("execute_result:","").strip()
        if result.endswith(".0"):
            result=result.replace(".0","")
        return result
    except:
        print("request code interpreter failed")
        return "request code interpreter failed"
    
if __name__ == '__main__':
    code="def execute():\n    worker_bees = 400\n    baby_bees = worker_bees / 2\n    return baby_bees\nexecute()"
    result=request_code_interpreter(code)
    print(result)
    print(type(result))