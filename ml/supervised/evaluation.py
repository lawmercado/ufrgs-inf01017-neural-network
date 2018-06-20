#! /usr/bin/python

from __future__ import division
import copy
from .classes.measure import Measure
from data.handler import DataHandler


def repeated_crossvalidation(num_repetitions, num_folds, data_handler, algorithm):
    """
    :param num_repetitions: Number of repetitions
    :param num_folds: Number of folds to generate
    :param DataHandler data_handler: Data for the cross validation
    :param ClassificationAlgorithm algorithm: The algorithm to evaluate
    :return: Dict with values for Accuracy, F1-measure
    :rtype: dict
    """

    measures = {"acc": [], "f1-measure": []}

    for i in range(0, num_repetitions):
        cv_measures = crossvalidation(num_folds, data_handler, algorithm)

        measures["acc"] += cv_measures["acc"]
        measures["f1-measure"] += cv_measures["f1-measure"]

    return measures


def crossvalidation(num_folds, data_handler, algorithm):
    """
    :param num_folds: Number of folds to generate
    :param DataHandler data_handler: Data for the cross validation
    :param ClassificationAlgorithm algorithm: The algorithm to evaluate
    :return: A dict containing the measures for each fold
    :rtype: dict
    """

    measures = {"acc": [], "f1-measure": []}

    folds = data_handler.stratify(num_folds)

    for index_fold, fold in enumerate(folds):
        aux_folds = copy.deepcopy(folds)  # Copy the folds
        test_fold = aux_folds.pop(index_fold)

        # Train the algorithm & classify the test fold
        train_instances = []

        for aux_fold in aux_folds:
            for instance in aux_fold:
                train_instances.append(instance)

        test_instances = [instance[0] for instance in test_fold]
        train_handler = DataHandler(train_instances)
        test_handler = DataHandler(test_fold)

        classified_instances = algorithm.classify(train_handler, test_instances)

        measure = Measure()
        measure.calculate(test_handler.as_instances(), classified_instances, data_handler.classes())

        measures["acc"].append(measure.accuracy)
        measures["f1-measure"].append(measure.f_measure(1))

    return measures


def get_statistics(measures):
    """
    With a set of measures, calculates the average and de standard

    :param dict measures: The name of the measures and a list of measurement
    :return: A list of tuples containing the average and the standard deviation associated with the measure
    :rtype: list [(measure, (average, standard deviation)), ...]
    """

    statistics = []

    keys = list(measures.keys())
    keys.sort()

    for id_measure in keys:
        acc = 0
        for measure in measures[id_measure]:
            acc += measure

        avg = acc / len(measures[id_measure])

        f_acc = 0
        for measure in measures[id_measure]:
            f_acc += (measure - avg) ** 2

        std_deviation = (f_acc / (len(measures[id_measure]) - 1)) ** 0.5

        statistics.append((id_measure, (avg, std_deviation)))

    return statistics
