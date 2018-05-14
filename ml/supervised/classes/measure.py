#! /usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division


class Measure(object):

    accuracy = None
    precision = None
    recall = None

    def calculate(self, instances, classified_instances, classes):
        # Initialize confusion matrix
        correct_classifications = 0
        true_positives = {}
        true_negatives = {}
        false_positives = {}
        false_negatives = {}
        for a_class in classes:
            true_positives[a_class] = 0
            true_negatives[a_class] = 0
            false_positives[a_class] = 0
            false_negatives[a_class] = 0

        # Compare classified samples with the test set
        for (predicted_sample, test_sample) in zip(classified_instances, instances):
            # If right prediction
            if predicted_sample[1] == test_sample[1]:
                correct_classifications += 1
                true_positives[predicted_sample[1]] += 1

                for a_class in classes:
                    if a_class != predicted_sample[1]:
                        true_negatives[a_class] += 1

            else:
                for a_class in classes:
                    if a_class == predicted_sample[1]:
                        false_positives[a_class] += 1
                    else:
                        false_negatives[a_class] += 1

        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        for a_class in classes:
            total_true_positives += true_positives[a_class]
            total_false_positives += false_positives[a_class]
            total_false_negatives += false_negatives[a_class]

            if len(classes) <= 2:
                break

        self.accuracy = correct_classifications / len(classified_instances)

        # Micro average for 3 or more possible classes
        self.recall = total_true_positives / (total_true_positives + total_false_negatives)
        self.precision = total_true_positives / (total_true_positives + total_false_positives)

    def f_measure(self, beta):
        return ((beta**2 + 1) * self.precision * self.recall)/((beta**2) * self.precision + self.recall)

    def __str__(self):
        text = "Acc: %s\nPrec: %s\nRec: %s\n"

        return text.format(str(self.accuracy), str(self.precision), str(self.recall))
