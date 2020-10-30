import random
import argparse
from tqdm import tqdm
import utils
from collections import defaultdict

def get_doc_with_only_one_sf(data):
    new_data = []
    for i in range(len(data)):
        d = data[i]
        
        title2lines = defaultdict(lambda: [])
        for title, ln in d['supporting_facts']:
            title2lines[title].append(ln)
        sup = [[title, lns[0]] for title, lns in title2lines.items() if len(lns) == 1]
        
        if len(sup) != 0:
            d['supporting_facts'] = sup
            new_data.append(d)
    return new_data

def merge_passages(titles, title2context, s):
    target_title, target_line = s[0], s[1]
    
    sindex = 0; eindex = 0
    answer_indices = None
    passage_text = ""
    line2charindex = []
    
    answer_text = title2context[target_title][target_line].strip()
    for title in titles:
        lines = [l.strip() for l in title2context[title]]
        # Remove the last line if it is empty
        if lines[-1] == "":
            lines.pop()
        
        text = " ".join(lines)
        se_indices = []
        for line in lines:
            se_indices.append((sindex, sindex + len(line) - 1))
            sindex += len(line) + 1

        if title == target_title:
            answer_indices = se_indices[target_line]
        
        line2charindex += se_indices
        passage_text += text + " "
    passage_text = passage_text.strip()
    sindex, eindex = answer_indices
    
    assert answer_text == passage_text[sindex:eindex+1]
    assert line2charindex[-1][1] == len(passage_text)-1

    return line2charindex, passage_text, sindex, eindex, answer_text

def to_mrqa(data):
    new_data = []
    example_id = 100000000
    for d in tqdm(data):
        contexts = d['context']
        question = d['question']
        sup = d['supporting_facts']
        
        title2context = {c[0]: c[1] for c in contexts}
        
        target_sup = {s[0]:0 for s in sup}
        default_titles = [title for title in list(title2context.keys()) if title not in target_sup]
        
        for s in sup:
            example_id += 1
            stitle = s[0]
            
            target_titles = default_titles + [stitle]
            random.shuffle(target_titles)
           
            line2charindex, passage_text, sindex, eindex, answer_text = merge_passages(target_titles, title2context, s)
            
            example = {
                'context': passage_text,
                'sentences': line2charindex,
                'qas': [{
                    'qid': example_id,
                    'question': question,
                    'detected_answers': [{
                        'char_spans': [[sindex, eindex]], 
                        'text': answer_text
                    }],
                    'answers': [answer_text],
                }]
            }
            new_data.append(example)
    return new_data

def filter_data(data):
    new_data = get_doc_with_only_one_sf(data)
    new_data = to_mrqa(new_data)
    return new_data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)

    args = parser.parse_args()
    return args

def main(args):
    data = utils.read_json(args.source)
    new_data = filter_data(data)
    _ = utils.write_jsonl(new_data, args.output)

if __name__ == '__main__':
    random.seed(1000)
    args = parse_args()
    main(args)
