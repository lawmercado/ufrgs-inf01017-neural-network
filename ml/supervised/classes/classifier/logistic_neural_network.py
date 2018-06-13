#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from .classifier import Classifier
import numpy as np
import math


class LogisticNeuralNetwork(Classifier):

    __layers = []
    __weights = []
    __regularization_factor = 0
    __activations = []
    __alpha = 0.001

    def __init__(self, data_handler, layers, initial_weights, regularization_factor=0):
        self.__layers = list(layers)
        self.__weights = initial_weights
        self.__regularization_factor = regularization_factor

        activations = []

        for num_neuron in self.__layers:
            layer_activation = []

            for i in range(0, num_neuron):
                layer_activation.append(None)

            activations.append(layer_activation)

        self.__activations = activations

        self.__train(data_handler)

    def __train(self, data_handler):
        print("Cost: " + str(self.__cost(data_handler)))

        self.__backpropagation(data_handler)

    def __cost(self, data_handler):
        instances = data_handler.as_instances()
        j = None

        for instance in instances:
            fw = self.__propagate(instance[0])

            if j is not None:
                j = j + np.array(
                    [-instance[1] * math.log(output, 10) - (1 - instance[1]) * math.log(1 - output) for output in fw])
            else:
                j = np.array(
                    [-instance[1] * math.log(output, 10) - (1 - instance[1]) * math.log(1 - output) for output in fw])

        j = np.sum(j)/len(instances)

        s = 0

        for weights_per_layer in self.__weights:
            for weights_per_neuron in weights_per_layer:
                for i_weights in range(1, len(weights_per_neuron)):
                    s += weights_per_neuron[i_weights] ** 2

        s = (self.__regularization_factor / (2 * len(instances))) * s

        return j + s

    def __propagate(self, instance):
        self.__activations[0] = list(instance)

        # Add the bias
        self.__activations[0].insert(0, 1.0)

        for i_hl in range(1, len(self.__activations) - 1):
            # terms = np.array(self.__weights[i_hl - 1]) * np.array([self.__activations[i_hl - 1], ] * self.__layers[i_hl])
            # zs = np.sum(terms, axis=1).tolist()
            weights = np.array(self.__weights[i_hl - 1])
            act = np.array(self.__activations[i_hl - 1])
            terms = weights.dot(act)
            zs = terms.tolist()

            # Applies sigmoid
            self.__activations[i_hl] = [self.__activation(z) for z in zs]

            # Add the bias
            self.__activations[i_hl].insert(0, 1.0)

        i_ol = len(self.__activations) - 1

        # terms = self.__weights[i_ol - 1] * np.array([self.__activations[i_ol - 1], ] * self.__layers[i_ol])
        # zs = np.sum(terms, axis=1).tolist()
        weights = np.array(self.__weights[i_ol - 1])
        act = np.array(self.__activations[i_ol - 1])
        terms = weights.dot(act)
        zs = terms.tolist()

        self.__activations[i_ol] = [self.__activation(z) for z in zs]

        return self.__activations[i_ol]

    def __activation(self, z):
        return 1 / (1 + math.exp(-z))

    def __backpropagation(self, data_handler):
        instances = data_handler.as_instances()
        deltas = []
        d = [[] for i in range(1, len(self.__activations) - 2)]

        for instance in instances:
            fw = self.__propagate(instance[0])

            deltas_ol = []
            for output in fw:
                deltas_ol.append(output - instance[1])

            deltas.append(deltas_ol)

            for i_hl in range(len(self.__activations) - 2, 2):
                weights = np.array(self.__weights[i_hl])
                weights = np.transpose(weights)

                d_kplus1 = np.array(deltas[len(deltas) - 1])
                # terms = weights * d_kplus1
                # terms_sum = np.sum(terms, axis=1)
                terms = weights.dot(d_kplus1)

                a_terms = np.ones((len(self.__activations[i_hl]))) - np.array(self.__activations[i_hl])

                # delta_hl = np.array(terms_sum * a_terms).tolist()
                delta_hl = np.array(terms * a_terms).tolist()
                delta_hl.pop()
                deltas.append(delta_hl)

            deltas.reverse()

            for i_hl in range(len(self.__activations) - 2, 1):
                d_aux = np.array(deltas[i_hl + 1]) * np.array(self.__activations[i_hl])
                d_aux = np.array(d[i_hl]) + d_aux
                d[i_hl] = d_aux.tolist()

        for i_hl in range(len(self.__activations) - 2, 1):
            p_aux = np.array(self.__weights[i_hl])
            p_aux[:, 0] = 0
            p_aux = p_aux * self.__regularization_factor
            # p[i_hl] = p_aux.tolist()

            d_aux = (np.array(d[i_hl]) + p_aux) / len(instances)
            d[i_hl] = d_aux.tolist()

        for i_hl in range(len(self.__activations) - 2, 1):
            weights = np.array(self.__weights[i_hl]) - np.array(d[i_hl]) * self.__alpha
            self.__weights[i_hl] = weights

    def classify(self, test_data_handler, test_instances):
        raise NotImplementedError