import argparse
import random
import utils

class Preprocessor:
    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        print("Set random seed: {:d}".format(seed))
        random.seed(seed)
        return 0

    def read_data(self, fname):
        "Not implemented yet"
    
    def split_data(self, data):
        "Not implemented yet"

    def write_data(self, data, fname):
        "Not implemented yet"

class MRQAPreprocessor(Preprocessor):
    def __init__(self):
        super().__init__(1234)
    
    def read_data(self, fname):
        data = utils.read_json_gz(fname)
        return data

    def split_data(self, data):
        n = len(data)
        p = int(n * 0.1)
        random.shuffle(data)
        train_data = data[p:]
        dev_data = data[:p]
        return (train_data, dev_data)

    def write_data(self, data, fname):
        _ = utils.write_jsonl(data, fname)
        return 0

class SQUAD_V1Preprocessor(Preprocessor):
    def __init__(self):
        super().__init__(8473)
    
    def read_data(self, fname):
        data = utils.read_json(fname)
        return data
    
    def split_data(self, data):
        ndata = data['data']
        n = len(ndata)
        p = int(n * 0.1)
        random.shuffle(ndata)
        
        train_data = {
            'data': [], 
            'version': 1.1
        }
        dev_data = {
            'data': [], 
            'version': 1.1
        }
        train_data['data'] = ndata[p:]
        dev_data['data'] = ndata[:p]
        return (train_data, dev_data)

    def write_data(self, data, fname):
        _ = utils.write_json(data, fname)
        return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=str)
    parser.add_argument("--train_output", required=True, type=str)
    parser.add_argument("--dev_output", required=True, type=str)
    parser.add_argument("--data_type", required=True, type=str)

    args = parser.parse_args()
    return args

def main(args):
    data_type2prep = {
        "mrqa": MRQAPreprocessor,
        "squad_v1": SQUAD_V1Preprocessor,
    }
    prep = data_type2prep[args.data_type]()
    data = prep.read_data(args.source)
    train, dev = prep.split_data(data)
    prep.write_data(train, args.train_output)
    prep.write_data(dev, args.dev_output)
    return 0

if __name__ == '__main__':
    args = parse_args()
    main(args)
