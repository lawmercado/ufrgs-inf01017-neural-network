#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
import logging
import random
import math
import copy

logger = logging.getLogger("main")


class DataHandler(object):
    __instances = []

    def __init__(self, instances):
        self.__instances = copy.deepcopy(instances)

    def __str__(self):
        return str(self.as_instances())

    def as_instances(self):
        if self.__instances:
            return copy.deepcopy(self.__instances)

        return []

    def classes(self):
        return list(self.__by_class_attr_values().keys())

    def stratify(self, k_folds):
        """
        Divide the data into k stratified folds, maintaining the main proportion

        :param integer k_folds: Number of folds
        :return: The folds
        :rtype: list
        """

        data = self.as_instances()
        classes = self.__by_class_attr_values()

        folds = [[] for _ in range(0, k_folds)]

        instances_per_fold = round(len(data) / k_folds)

        for yi in classes:
            yi_proportion = len(classes[yi]) / len(data)

            counter = round(yi_proportion * instances_per_fold)

            while counter > 0:
                try:
                    for idx in range(0, k_folds):
                        instance = data[classes[yi].pop(random.randint(0, len(classes[yi]) - 1))]

                        folds[idx].append(instance)

                    counter -= 1

                except (ValueError, IndexError):
                    break

        return folds

    def __by_class_attr_values(self):
        instances = self.as_instances()

        data = {str(instance[1]): [] for instance in instances}

        for idx, instance in enumerate(instances):
            data[str(instance[1])].append(idx)

        return data

    @staticmethod
    def parse(file, normalize=False):
        instances = []

        try:
            with open(file) as f:
                for line in f:
                    attrs, outputs = line.rstrip().lstrip().split(";")

                    parsed_attrs = []
                    parsed_outputs = []

                    for item in attrs.split(","):
                        parsed_attrs.append(float(item))

                    for item in outputs.split(","):
                        parsed_outputs.append(float(item))

                    instances.append((tuple(parsed_attrs), tuple(parsed_outputs)))

            if normalize:
                instances = DataHandler.normalize(instances)

        except IndexError:
            print("Error when parsing the dataset")

        return instances

    @staticmethod
    def normalize(instances):
        """
        Normalizes the data represented by an dictionary
        :return: A list with the normalized data
        :rtype: list
        """

        average_instance = DataHandler.__get_averages(instances)
        std_deviation_instance = DataHandler.__get_std_deviations(instances, average_instance)
        n_instances = []

        for instance in instances:
            n_instance_attrs = [0 for _ in instances[0][0]]
            n_instance_outputs = [0 for _ in instances[0][1]]

            for idx_value, value in enumerate(instance[0]):
                if std_deviation_instance[0][idx_value] != 0:
                    n_instance_attrs[idx_value] = (value - average_instance[0][idx_value]) / std_deviation_instance[0][idx_value]
                else:
                    n_instance_attrs[idx_value] = (value - average_instance[0][idx_value])

            for idx_value, value in enumerate(instance[1]):
                n_instance_outputs[idx_value] = value

                '''if std_deviation_instance[1][idx_value] != 0:
                    n_instance_outputs[idx_value] = (value - average_instance[1][idx_value]) / std_deviation_instance[1][idx_value]
                else:
                    n_instance_outputs[idx_value] = (value - average_instance[0][idx_value])'''

            n_instances.append((tuple(n_instance_attrs), tuple(n_instance_outputs)))

        return n_instances

    @staticmethod
    def __get_averages(instances):
        instance_attrs = [0 for _ in instances[0][0]]
        instance_outputs = [0 for _ in instances[0][1]]

        for instance in instances:
            for idx_value, value in enumerate(instance[0]):
                instance_attrs[idx_value] += value

            for idx_value, value in enumerate(instance[1]):
                instance_outputs[idx_value] += value

        for idx_value, value in enumerate(instance_attrs):
            instance_attrs[idx_value] = value / len(instances)

        for idx_value, value in enumerate(instance_outputs):
            instance_outputs[idx_value] = value / len(instances)

        return tuple(instance_attrs), tuple(instance_outputs)

    @staticmethod
    def __get_std_deviations(instances, average_instance):
        instance_attrs = [0 for _ in instances[0][0]]
        instance_outputs = [0 for _ in instances[0][1]]

        for instance in instances:
            for idx_value, value in enumerate(instance[0]):
                instance_attrs[idx_value] += (value - average_instance[0][idx_value]) ** 2

            for idx_value, value in enumerate(instance[1]):
                instance_outputs[idx_value] += (value - average_instance[1][idx_value]) ** 2

        for idx_value, value in enumerate(instance_attrs):
            instance_attrs[idx_value] = (value / len(instances)) ** 0.5

        for idx_value, value in enumerate(instance_outputs):
            instance_outputs[idx_value] = (value / len(instances)) ** 0.5

        return tuple(instance_attrs), tuple(instance_outputs)
