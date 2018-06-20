#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function
import logging
import argparse
import random

from data.handler import DataHandler
from ml.supervised.classes.classifier.logistic_neural_network import LogisticNeuralNetwork
from ml.supervised.evaluation import crossvalidation, get_statistics


def setup_logger():

    class MyFilter(object):
        def __init__(self, level):
            self.__level = level

        def filter(self, log_record):
            return log_record.levelno <= self.__level

    main_logger = logging.getLogger("main")

    formatter = logging.Formatter("%(levelname)s: %(message)s")

    handler = logging.FileHandler("output.log", mode="w")
    handler.setLevel(logging.DEBUG)
    handler.filter(MyFilter(logging.DEBUG))
    handler.setFormatter(formatter)
    main_logger.addHandler(handler)

    main_logger.setLevel(logging.INFO)

    return main_logger


if __name__ == '__main__':
    logger = setup_logger()

    parser = argparse.ArgumentParser()
    #parser.add_argument("operation", metavar="O", type=str, help="operation to be executed: 'backpropagation', 'validate', 'classify'")
    parser.add_argument("struct_file", metavar="N", type=str, help="network structure .txt file")
    parser.add_argument("weights_file", metavar="W", type=str, help="network initial weights .txt file")
    parser.add_argument("dataset_file", metavar="D", type=str, help="dataset .txt file")
    parser.add_argument("--verbose", help="enables debugging", action="store_true")
    parser.add_argument("--normalize", help="normalize the dataset", action="store_true")
    parser.add_argument("--seed", type=int, help="the seed to consider in random numbers generation")
    parser.add_argument("--alpha", type=float, help="the alpha to be considered in the neural network")
    parser.add_argument("--beta", type=float, help="the beta to be considered in the neural network")
    parser.add_argument("--ins_per_batch", type=int, help="the number of instances to use per mini batch in the training")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    print("Reading dataset file...")

    data_handler = DataHandler(DataHandler.parse(args.dataset_file, args.normalize))

    print("Reading strucuture files...")

    regularization_factor = 0
    layers = []
    initial_weights = []

    with open(args.struct_file) as sf:
        lines = [line.rstrip().lstrip() for line in sf.readlines()]

        regularization_factor = float(lines[0])

        for i in range(1, len(lines)):
            layers.append(int(lines[i]))

    with open(args.weights_file) as wf:
        lines = [line.rstrip().lstrip() for line in wf.readlines()]

        if len(lines) > 0:
            for idx, line in enumerate(lines):
                layer_weights = line.split(";")
                if len(layer_weights) != layers[idx + 1]:
                    raise ImportError("Number of weights for the layer do not match with the number of neurons!")

                parsed_layer_weights = []

                for n_weights in layer_weights:
                    parsed_n_weights = [float(weight) for weight in n_weights.rstrip().lstrip().split(",")]
                    parsed_layer_weights.append(parsed_n_weights)

                    if len(parsed_n_weights) != (layers[idx] + 1):
                        raise ImportError("Number of weights do not match with next layer neurons")

                initial_weights.append(parsed_layer_weights)
        else:
            r = 4 * ((6/(layers[0] + layers[len(layers) - 1]))**0.5)

            for idx_layer in range(0, len(layers) - 1):
                weights = [random.uniform(-r, r) for _ in range(0, layers[idx_layer] + 1)]
                layer_weights = [list(weights) for _ in range(0, layers[idx_layer + 1])]

                initial_weights.append(layer_weights)

    alpha = 0.9
    beta = 0.95
    ins_per_batch = 50

    if args.alpha is not None:
        alpha = args.alpha

    if args.beta is not None:
        beta = args.beta

    if args.ins_per_batch is not None:
        ins_per_batch = args.ins_per_batch

    print("Classifying...")

    data = crossvalidation(10, data_handler, LogisticNeuralNetwork(layers, initial_weights, regularization_factor, alpha, beta, ins_per_batch))

    print(get_statistics(data))
