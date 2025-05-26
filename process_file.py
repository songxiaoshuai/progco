import os
import json

def _read_json(read_file_path):
    with open(read_file_path,encoding='utf-8') as file:
        raw_data=json.load(file)
    return raw_data

def _save_json(save_file_path,data,mode='w'):
    with open(save_file_path,mode,encoding='utf-8')as fp:
        json.dump(data,fp,ensure_ascii=False,indent=4)

def _read_jsonl(read_file_path):
    data = []
    with open(read_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def _save_jsonl(save_file_path, data,mode='w'):
    with open(save_file_path, mode, encoding='utf-8') as f:
        for record in data:
            f.write(json.dumps(record,ensure_ascii=False) + '\n')

def _read_txt(read_file_path):
    with open(read_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
    
def _save_txt(save_file_path, data,mode='w'):
    with open(save_file_path, mode) as file:
        file.write(data)

def read_file(read_file_path):
    if read_file_path.endswith('.json'):
        return _read_json(read_file_path)
    elif read_file_path.endswith('.jsonl'):
        return _read_jsonl(read_file_path)
    elif read_file_path.endswith('.txt'):
        return _read_txt(read_file_path)
    else:
        raise ValueError('file_type is not supported')


def _ensure_dir(file_path):
    # Get directory path
    directory = os.path.dirname(file_path)
    # Create directory if it doesn't exist and is not empty
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    
def save_file(save_file_path,data,mode='w'):
    _ensure_dir(save_file_path)
    if save_file_path.endswith('.json'):
        return _save_json(save_file_path,data,mode)
    elif save_file_path.endswith('.jsonl'):
        return _save_jsonl(save_file_path,data,mode)
    elif save_file_path.endswith('.txt'):
        return _save_txt(save_file_path,data,mode)
    else:
        raise ValueError('file_type is not supported')

