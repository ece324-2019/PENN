# pip install my-voice-analysis
# copy the contents of the `__init__.py` function and change the name to `myspsolution.py`
# Also need to copy `mysolution.praat` to the same directory

import myspsolution as mysp

class baseline(object):

    def __init__(self):
        pass

    def __call__(self, x):
        print("hi")


if __name__ == "__main__":

    model = baseline()
    model(10)
