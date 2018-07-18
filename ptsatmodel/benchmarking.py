import argparse
import ptsatmodel.classifiers as classifiers


def main(args):
    data = classifiers.Data(args.input)
    for c in classifiers.Classifiers:
        clf = c.value(data, args.storage)
        clf.run()
        print(clf.result)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--storage')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())