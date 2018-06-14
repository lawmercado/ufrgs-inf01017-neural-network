#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from .classifier import Classifier
import numpy as np
import math


class LogisticNeuralNetwork(Classifier):

    __layers = []
    __weights = []
    __old_weights = []
    __regularization_factor = 0
    __activations = []
    __gradients = []
    __alpha = 0.1

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

        gradients = []
        # print(self.__weights)
        for weights_per_layer in self.__weights:
            gradients_per_layer = []
            for weights_per_neuron in weights_per_layer:
                zeros = np.zeros(len(weights_per_neuron))
                gradients_per_layer.append(zeros.tolist())

            gradients.append(gradients_per_layer)

        self.__gradients = gradients

        self.__train(data_handler)

    def __train(self, data_handler):
        print("Cost: " + str(self.__cost(data_handler)))

        self.__backpropagation(data_handler)
        while not self.__stop(data_handler):
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

        print("Activations:")
        print(self.__activations)

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
        last_layer = len(self.__activations) - 1
        print("Starting backpropagation")
        instances = data_handler.as_instances()
        # print("Instances:")
        # print(instances)

        for instance in instances:
            print("Instance:")
            print(instance)
            fw = self.__propagate(instance[0])
            print("Output after propagating one instance:")
            print(fw)

            deltas = []
            deltas_out = []
            for output in fw:
                deltas_out.append(output - instance[1])

            deltas.append(deltas_out)

            for k in range(last_layer - 1, 0, -1):
                # print("k=" + str(k))
                weights = np.array(self.__weights[k])
                weights = np.transpose(weights)
                # print("Weights layer-" + str(k))
                # print(weights)

                d_kplus1 = np.array(deltas[len(deltas) - 1])
                # print("delta k+1")
                # print(d_kplus1)
                # terms = weights * d_kplus1
                # terms_sum = np.sum(terms, axis=1)
                terms = weights.dot(d_kplus1)
                # print("Weights x Delta_k+1:")
                # print(terms)

                # print("a:")
                # print(self.__activations[k])
                a_terms = np.ones((len(self.__activations[k]))) - np.array(self.__activations[k])
                # print("1 - a:")
                # print(a_terms)
                a_terms = a_terms * np.array(self.__activations[k])
                # print("a x (1 - a):")
                # print(a_terms)

                # delta_hl = np.array(terms_sum * a_terms).tolist()
                delta_in = np.array(terms * a_terms).tolist()
                # print("Weights x Delta_k+1 x a x (1 - a):")
                # print(delta_in)
                delta_in.pop(0)
                deltas.append(delta_in)

            deltas.reverse()
            print("Deltas:")
            print(deltas)

            for k in range(last_layer - 1, -1, -1):
                # print("k=" + str(k))
                # print("d(k+1):")
                # print(deltas[k])
                # print("a(k):")
                # print(self.__activations[k])
                delt = np.array(deltas[k])
                act = np.array(self.__activations[k])
                d_aux = np.outer(delt, act)
                # print("d(k+1) x a(k)")
                # print(d_aux)
                # print("D(k):")
                # print(self.__gradients[k])
                d_aux = np.array(self.__gradients[k]) + d_aux
                self.__gradients[k] = d_aux.tolist()
                # print("D(k) updated:")
                # print(self.__gradients[k])

        for k in range(last_layer - 1, -1, -1):
            # print("k=" + str(k))
            p_aux = np.array(self.__weights[k])
            p_aux[:, 0] = 0
            # print("P(k) with first column nullified:")
            # print(p_aux)
            p_aux = p_aux * self.__regularization_factor
            # print("P(k) x Regularization Factor:")
            # print(p_aux)
            # p[k] = p_aux.tolist()

            # print("D(k):")
            # print(self.__gradients[k])
            # print("n:")
            # print(len(instances))
            d_aux = (np.array(self.__gradients[k]) + p_aux) / len(instances)
            self.__gradients[k] = d_aux.tolist()
            # print("D(k) updated:")
            # print(self.__gradients[k])

            self.__old_weights = list(self.__weights)
        for k in range(last_layer - 1, -1, -1):
            print("k=" + str(k))
            print("Old weights:")
            print(self.__weights[k])
            weights = np.array(self.__weights[k]) - np.array(self.__gradients[k]) * self.__alpha
            self.__weights[k] = weights.tolist()
            print("New weights:")
            print(self.__weights[k])

    def __stop(self, data_handler):
        stop = True

        instances = data_handler.as_instances()
        for instance in instances:

            fw = self.__propagate(instance[0])
            for output in fw:
                diff = abs(output - instance[1])

                #print(diff)
                # If error bigger than NUMBER then must continue backpropagating
                if diff > 0.1:
                    stop = False
                    return stop

        return stop


    def classify(self, test_data_handler, test_instances):
        raise NotImplementedError
