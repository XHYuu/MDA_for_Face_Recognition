import argparse
import numpy as np
from sklearn.metrics import classification_report

from dataLoader import ORLDataset
from model.Eigenface import Eigenface
from model.Fisherface import Fisherface
from model.MDA import MDA
from model.LDA import LDAProcessor

np.random.seed(42)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ORL")
    parser.add_argument("--data_path", type=str, default="data/ORL")
    parser.add_argument("--model", type=str, default="LDA",
                        help="choose MDA, eigenface, fisherface, LDA")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--pca_energy", type=float, default=0.95)
    parser.add_argument("--lda_outdim", type=int, default=30,
                        help="Hyperparameter only for fisherface")

    args = parser.parse_args()
    return args


def train_test(args, train_set, train_label, test_set, test_label):
    train_set = train_set.reshape(train_set.shape[0] * train_set.shape[1], *train_set.shape[2:])
    train_label = train_label.reshape(train_label.shape[0] * train_label.shape[1], *train_label.shape[2:])
    test_set = test_set.reshape(test_set.shape[0] * test_set.shape[1], *test_set.shape[2:])
    test_label = test_label.reshape(test_label.shape[0] * test_label.shape[1], *test_label.shape[2:])

    if args.model == "MDA":
        model = MDA(input_dim=list(train_set[0].shape),
                    output_dim=[40, 40],
                    epochs=args.epochs,
                    epsilon=args.epsilon)
    elif args.model == "eigenface":
        model = Eigenface(components=25)
    elif args.model == "fisherface":
        model = Fisherface(pca_energy=args.pca_energy, out_dim=args.lda_outdim)
    elif args.model == "LDA":
        model = LDAProcessor(n_components=30)
    else:
        raise ValueError(f"Without model:{args.model}")
    model.fit(train_set, train_label)

    pred = model.predict(test_set)
    print(classification_report(test_label, pred))


def main():
    args = arg_parser()
    dataset = ORLDataset(args.data_path)
    dataset.load_data()
    # TODO：use different m
    # for m in [5, 4, 3, 2]:
    # print(f"Train set : Test set = {m} : {10 - m}")
    train_set, train_label, test_set, test_label = dataset.split_personal_image(5)
    train_test(args, train_set, train_label, test_set, test_label)


if __name__ == '__main__':
    main()
