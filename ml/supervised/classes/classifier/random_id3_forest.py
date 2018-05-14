#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from .classifier import Classifier
from .id3_decision_tree import ID3DecisionTree
import logging

logger = logging.getLogger("main")


class RandomID3Forest(Classifier):

    __num_trees = 0

    def __init__(self, num_trees):
        self.__num_trees = num_trees

    def classify(self, test_data_handler, test_instances):
        trees = []
        classified = []

        bag = test_data_handler.bagging(self.__num_trees)

        for bootstrap in bag:
            trees.append(ID3DecisionTree(bootstrap))

        for test_instance in test_instances:
            classifications = []

            for tree in trees:
                logger.debug("Testing: " + str(test_instance))
                tree_classification = tree.classify(test_instance)
                logger.debug("Classified (by one of the trees) as " + str(tree_classification))

                classifications.append(tree_classification)

            counter = {tree_classification: classifications.count(tree_classification) for tree_classification in classifications}

            ensemble_result = max(counter, key=lambda key: counter[key])

            classified.append((test_instance, ensemble_result))

        return classified
