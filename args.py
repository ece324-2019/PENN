import argparse

def get_args():
    # Command Line Arguments
    parser = argparse.ArgumentParser(description='Terminal Arguments for PENN')
    
    parser.add_argument('--model', type=str, choices=["baseline", "cnn"], default="cnn")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--overfit', action="store_true", default=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    print(args)
