import argparse

def get_args():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='Terminal Arguments for PENN')
    
    parser.add_argument('--model', type=str, choices=["mlp", "average", "cnn", "rnn"], default="cnn")
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--lr', type=float, default=-1)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=-1)
    parser.add_argument('--overfit', action="store_true", default=False)
    parser.add_argument('--save', action="store_true", default=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)
