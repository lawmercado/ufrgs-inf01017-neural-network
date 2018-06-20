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

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    data_handler = DataHandler(DataHandler.parse(args.dataset_file, args.normalize))

    # print(data_handler)

    print("\nReading strucuture files...\n")

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
                weights = [round(random.uniform(-r, r), 5) for _ in range(0, layers[idx_layer] + 1)]
                layer_weights = [list(weights) for _ in range(0, layers[idx_layer + 1])]

                initial_weights.append(layer_weights)

    LogisticNeuralNetwork(data_handler, layers, initial_weights, regularization_factor)
