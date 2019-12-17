import argparse
import faiss
import time
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and save an empty FAISS index')
    parser.add_argument('index_type', type=str, help='String for index_factory()')
    parser.add_argument('train_data', type=str, help='Path to train data')
    parser.add_argument('index_file', type=str, help='Output Index file')        
    args = parser.parse_args()

    x = utils.load_features(args.train_data, 'rmac')[...]
    n, d = x.shape
    
    index = faiss.index_factory(d, args.index_type)
    train_time = time.time()
    index.train(x)
    train_time = time.time() - train_time
    print('Training Time:', train_time)
    faiss.write_index(index, args.index_file)
    

