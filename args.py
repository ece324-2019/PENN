import argparse

def get_args():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='Terminal Arguments for PENN')
    
    # for pre_train.py
    parser.add_argument('--model', type=str, choices=["mlp", "average", "cnn", "rnn"], default="cnn")
    parser.add_argument('--batch-size', type=int, default=200)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval-every', type=int, default=-1)
    parser.add_argument('--save-as', type=str, default=None)
    parser.add_argument('--overfit', action="store_true", default=False)
    
    # for fine_tune.py and demo.py
    parser.add_argument('--model-name', type=str, default="trained_model")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)
