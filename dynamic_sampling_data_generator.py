import tensorflow as tf
import numpy as np
import math

class DynamicSamplingDataGenerator(tf.compat.v2.keras.utils.Sequence):
    """
    Data generator class definition to generate data on the fly when requested,
    instead of storing the entire dataset in RAM. This class is also useful in transforming the dataset
    during training for restricted dynamic sampling to take place.

    Example:
        '''python
        train_data_generator = DynamicSamplingDataGenerator(x_train=training_data, y_train=training_labels,
            batch_size=64, dimensions=(28,28), classes=[0,1,2,3,4,5])
        '''
    """

    def __init__(self,
                 x_train, y_train,
                 batch_size,
                 dimensions,
                 classes):
        """Initializes `DynamicSamplingDataGenerator` class.

            Args:
              x_train: training set.
              y_train: training labels.
              batch_size: size of each training batch.
              dimensions: array dimensions of a single instance in the training set.
              classes: all predictable classes of the dataset.
            """
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = batch_size
        num_instances = len(x_train)
        self.num_batches = math.ceil(num_instances / self.batch_size)

        # Separate class-wise instances and calculate average number of instances per class
        self.classes = classes
        self.num_classes = len(classes)
        self.class_indices = dict(zip(classes, range(self.num_classes)))
        self.instances_by_class = dict()
        for _class in classes:
            self.instances_by_class[_class] = x_train[(y_train == _class)]
        self.average_num_instances_per_class = math.ceil(num_instances / self.num_classes)

        self.dimensions = dimensions
        self.indices = np.arange(num_instances)
        np.random.shuffle(self.indices)

    def __len__(self):
        """Number of batches in the generator.

            Returns:
                The number of batches in the generator.
        """
        return self.num_batches  # Return the number of batches in the dataset

    def __getitem__(self, index):
        """Gets batch at position `index`.

            Arguments:
                index: position of the batch in the Sequence.

            Returns:
                A batch
        """
        # Generate indices of instances in a batch
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        x = self.get_x_train_batch(batch_indices)
        y = self.get_y_train_batch(batch_indices)
        return x, y

    def get_x_train_batch(self, batch_indices):
        """Generates and return a batch of data.

            Arguments:
                batch_indices: indices of the data to fetch from training set.

            Returns:
                batch
        """
        x = np.empty((self.batch_size, *self.dimensions))
        for i, batch_index in enumerate(batch_indices):
            x[i] = self.x_train[batch_index]
        return x[:, :, :, np.newaxis]

    def get_y_train_batch(self, batch_indices):
        """Generates and return a batch of labels.

            Arguments:
                batch_indices: indices of the labels to fetch from training labels.

            Returns:
                batch
        """
        y = np.empty(self.batch_size)
        for i, batch_index in enumerate(batch_indices):
            y[i] = self.y_train[batch_index]
        return y
