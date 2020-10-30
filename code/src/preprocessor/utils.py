import json
import gzip

def read_json_gz(fname):
    with gzip.GzipFile(fname, 'r') as reader:
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]
    return input_data

def write_jsonl(data, fname):
    with open(fname, 'w') as f:
        for d in data:
            json.dump(d, f)
            f.write('\n')
    return 0

def read_json(fname):
    f = open(fname, 'r')
    data = json.load(f)
    f.close()
    return data

def write_json(data, fname):
    with open(fname, 'w') as f:
        json.dump(data, f)
    return 0
