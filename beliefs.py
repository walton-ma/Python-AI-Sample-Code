# ----------------------------------------------------------------------
# Name:     beliefs
# Purpose:  Homework 8
#
# Author(s):
#
# ----------------------------------------------------------------------
"""
Module to track the belief distribution over all possible grid positions

Your task for homework 8 is to implement:
1.  update
2.  recommend_sensing
"""
import utils


class Belief(object):
    """
    Belief class used to track the belief distribution based on the
    sensing evidence we have so far.
    Arguments:
    size (int): the number of rows/columns in the grid

    Attributes:
    open (list of tuples): list containing all the positions that have not
        been observed so far.
    current_distribution (dictionary): probability distribution based on
        the evidence observed so far.
        The keys of the dictionary are the possible grid positions
        The values represent the (conditional) probability that the
        treasure is found at that position given the evidence
        (sensor data) observed so far.
    """

    def __init__(self, size):
        # Initially all positions are open - have not been observed
        self.open = [(x, y) for x in range(size)
                     for y in range(size)]
        # Initialize to a uniform distribution
        self.current_distribution = {pos: 1 / (size ** 2) for pos in self.open}

    def update(self, color, sensor_position, model):
        """
        Update the belief distribution based on new evidence:  our agent
        detected the given color at sensor location: sensor_position.
        :param color: (string) color detected
        :param sensor_position: (tuple) position of the sensor
        :param model (Model object) models the relationship between the
             treasure location and the sensor data
        :return: None
        """
        # Iterate over ALL positions in the grid and update the
        # probability of finding the treasure at that position - given
        # the new evidence.
        # The probability of the evidence given the Manhattan distance
        # to the treasure is given by calling model.pcolorgivendist.
        # Don't forget to normalize.
        # Don't forget to update self.open since sensor_position has
        # now been observed.

        for pos in self.current_distribution:  # Iterate over ALL positions in the grid
            # Just have information for one position, therefore, P(C|T) is just the probability
            # of getting the color if treasure is at this position.
            # Then, P(T)*P(C|T) is the original P * (what I get from model.pcolorgivendist).
            # Finally, I need to update the value in the current_distribution dictionary for all positions.
            dist = utils.manhattan_distance(pos, sensor_position)
            self.current_distribution[pos] = self.current_distribution[pos] * model.pcolorgivendist(color, dist)

        # Normalize the answer by dividing the sum of all values in the dictionary.
        total = sum(self.current_distribution.values())
        for pos in self.current_distribution:
            self.current_distribution[pos] = self.current_distribution[pos] / total

        self.open.remove(sensor_position)  # Remove current point in the open list.

    def recommend_sensing(self):
        """
        Recommend where we should take the next measurement in the grid.
        The position should be the most promising unobserved location.
        If all remaining unobserved locations have a probability of 0,
        return the unobserved location that is closest to the (observed)
        location with he highest probability.
        If there are no remaining unobserved locations return the
        (observed) location with the highest probability.

        :return: tuple representing the position where we should take
            the next measurement
        """
        if not self.open:  # If the list is empty, means no more open points
            # Return the max of all probabilities of the current_distribution dictionary.
            return max(self.current_distribution.keys(), key=lambda x: self.current_distribution[x])

        if sum(self.current_distribution[pos] for pos in self.open) == 0:
            # The open list is not empty, and the sum of probabilities
            # for all positions in the list is zero.(zero for every position).

            # Return the position that is closest to the position with the max probability.
            max_pos = max(self.current_distribution.keys(), key=lambda x: self.current_distribution[x])
            return utils.closest_point(max_pos, self.open)

        else:
            # Return the position in the open list that has the highest probability.
            return max(self.open, key=lambda x: self.current_distribution[x])
