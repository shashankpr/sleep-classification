import numpy as np
import logging
import threading

from sklearn.metrics import confusion_matrix


class Metrics(object):
    def __init__(self, predicted_values, true_values):
        self.predicted_values = predicted_values
        self.true_values = true_values
        self.logger = logging.getLogger(__name__)

    def build_conf_matrix(self):
        """Build Confusion matrix given True values and Predicted Values.

        Returns: confusion matrix with specified labels.

        """
        pred_vals = []
        for prediction in self.predicted_values:
            pred_vals.append(np.argmax(prediction))

        print confusion_matrix(self.true_values, pred_vals, labels=[1, 2, 3, 4])

        self.logger.info("Confusion Matrix : {}".format(confusion_matrix(self.true_values, pred_vals,
                                                                         labels=[1, 2, 3, 4])))


class ThreadSafe(object):
    """Make a generator thread safe.

    Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.

    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):  # Py2
        with self.lock:
            return self.it.next()

def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.

    Args:
        f (generator): A generator function

    Returns:

    """
    def g(*a, **kw):
        return ThreadSafe(f(*a, **kw))
    return g