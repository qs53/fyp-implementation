import streamlit as st
import tensorflow as tf
import numpy as np
import math
from sklearn.metrics import confusion_matrix
from keras.metrics import SparseCategoricalAccuracy

final_status = ""

class ModelTrainer:
    """The class definition to train models with the loss and restricted dynamic smapling methods specified.
    
    Example:
        ```python
        model_trainer = ModelTrainer()
        model_trainer.train_model(model, epochs, train_generator, validation_generator, loss_fn, optimizer,
                        dynamic_sampling_start_epoch, session_state)
        ```
    """
    def train_model(self, model: tf.keras.Model, epochs, train_generator, validation_generator, loss_fn, optimizer,
                    dynamic_sampling_start_epoch, session_state):
        """Trains the model
        
            Args:
                model: model instance for training.
                epochs: number of training epochs.
                train_generator: training batch data generator.
                validation_generator: validation batch data generator.
                loss_fn: instance of the `SparseCategoricalFocalLoss` class.
                optimizer: the minimization technique to update model's traininable weights.
                dynamic_sampling_start_epoch: start epoch to start sampling dynamically.
                session_state: `SessionState` instance to store the trained model after successful training completion. It also stores
                the data augmentation class object
        """
        train_acc_metric = SparseCategoricalAccuracy()
        val_acc_metric = SparseCategoricalAccuracy()
        progress_bar = st.progress(0)
        percent_complete = 0
        status_message = st.empty()
        status = ""
        for epoch in range(epochs): # Loop through the training set with the epochs specified
            num_batches = len(train_generator)
            status += "  \nStart of epoch %d  \nNumber of batches in training set: %d  \nTraining...  \n" % (
                epoch, num_batches)
            with status_message.beta_container():
                st.info(status)
            # Iterate over the batches of the dataset.
            # Train model batch-wise
            for step, (x_batch_train, y_batch_train) in enumerate(train_generator):
                # Calculate gradients and update network weights
                with tf.GradientTape() as tape:
                    x_batch_train = tf.convert_to_tensor(x_batch_train)
                    y_batch_train = tf.convert_to_tensor(y_batch_train)
                    logits = model(x_batch_train, training=True)
                    loss_value = loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

                # Update training metric.
                train_acc_metric.update_state(y_batch_train, logits)

            # Display metrics at the end of each epoch.
            train_acc = train_acc_metric.result()
            status += "Training accuracy over epoch: %.4f   \n" % (float(train_acc),)
            with status_message.beta_container():
                st.info(status)

            # Reset training metrics at the end of each epoch
            train_acc_metric.reset_states()

            # Run a validation loop at the end of each epoch.
            for x_batch_val, y_batch_val in validation_generator:
                val_logits = model(x_batch_val, training=False)
                # Update validation metrics
                val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = val_acc_metric.result()
            val_acc_metric.reset_states()
            status += "Validation accuracy: %.4f   \n" % (float(val_acc),)
            with status_message.beta_container():
                st.info(status)
            # Update training set based on F1 scores
            if epoch < epochs - 1 and epoch >= dynamic_sampling_start_epoch: # Check if any need to sample dynamically
                status += "Calculating class-wise F1 scores for dynamic sampling...  \n"
                self.reformulate_training_set(model, train_generator, validation_generator, session_state.data_augmenter)
                with status_message.beta_container():
                    st.info(status)
            if epoch == epochs - 1:
                percent_complete = 100
            else:
                percent_complete += (int(100 / epochs))
            progress_bar.progress(percent_complete)
        final_status = status
        session_state.is_model_trained = percent_complete == 100
        if session_state.is_model_trained:
            session_state.trained_model = model

    def reformulate_training_set(self, model, train_generator, validation_generator, data_augmenter):
        """Mutate training set after calculating class-wise F1 scores.
        
            Args:
                model: model to make predictions on validation set.
                train_generator: dynamic data generator that produces batch training data.
                validation_generator: dynamic data generator that produces batch validation data.
                Used to calculate F1 scores after validation predictions and labels compared.
                data_augmenter: the data augmentation class object to perform real-time data augmentation
        """
        y_true = np.array([])
        y_pred = np.array([])
        # Compute class-wise performance using validation data
        for validation_data in validation_generator:
            x_batch, y_batch = validation_data
            batch_pred = model.predict(x_batch)
            for i in range(len(x_batch)):
                y_true = np.append(y_true, y_batch[i])
                y_pred = np.append(y_pred, np.argmax(batch_pred[i]))
        cf = confusion_matrix(y_true, y_pred) # Calculate the confusion matrix
        precision = np.array([])
        recall = np.array([])
        # Calculate one-versus-all precision and recall
        for i in range(len(cf)):
            if np.sum(cf[i]) == 0:
                precision = np.append(precision, 1)
            else:
                precision = np.append(precision, cf[i][i] / np.sum(cf[i]))
            if np.sum(cf, axis=0)[i] == 0:
                recall = np.append(recall, 0)
            else:
                recall = np.append(recall, cf[i][i] / np.sum(cf, axis=0)[i])
        # Calculate class-wise F1 scores
        f1 = np.zeros((len(cf)))
        for i in range(len(cf)):
            if precision[i] + recall[i] == 0:
                f1[i] = 0
            else:
                f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
        # Calculate weight to mutate training set to reflect minority instances more
        weight = 1 - f1
        weight_sum = sum(weight)
        if weight_sum == 0:
            train_generator.indices = np.arange(len(train_generator.x_train))
            np.random.shuffle(train_generator.indices)
        else:
            num_instances_per_class = (weight / weight_sum) * train_generator.average_num_instances_per_class
            for i, num_instances_in_class in enumerate(num_instances_per_class):
                num_instances_per_class[i] = int(round(num_instances_in_class))
            num_instances_per_class = num_instances_per_class.astype(int)
            num_instances = sum(num_instances_per_class)
            train_generator.x_train = np.empty((0, *train_generator.dimensions))
            train_generator.y_train = np.empty(num_instances)
            y_start_index = 0
            # Update distribution between classes
            for i, _class in enumerate(train_generator.classes):
                class_instances = train_generator.instances_by_class[_class]
                while len(class_instances) < num_instances_per_class[i]:
                    if data_augmenter is None:
                        class_instances = np.vstack((class_instances, class_instances))
                    else:
                        class_instances = np.vstack((class_instances, self.augmented_data(data_augmenter, class_instances)))
                train_generator.x_train = np.vstack(
                    (train_generator.x_train, class_instances[:num_instances_per_class[i]]))
                train_generator.y_train[y_start_index:y_start_index + num_instances_per_class[i]] = _class
                y_start_index = y_start_index + num_instances_per_class[i]
            # Shuffle training indices for randomization during training
            train_indices = np.random.permutation(num_instances)
            train_generator.x_train, train_generator.y_train = train_generator.x_train[train_indices], \
                                                               train_generator.y_train[train_indices]
            train_generator.indices = np.arange(num_instances)
            np.random.shuffle(train_generator.indices)
            train_generator.num_batches = math.ceil(num_instances / train_generator.batch_size)

    def augmented_data(self, data_augmenter, class_instances):
        """Augment data and return augmented class instances

            Args:
                data_augmenter: data augmentation class object
                class_instances: instances of class to augment
                """
        iterator = data_augmenter.flow(class_instances, batch_size=len(class_instances))
        return next(iterator)
