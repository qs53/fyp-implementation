import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from dynamic_sampling_data_generator import DynamicSamplingDataGenerator
from focal_loss import SparseCategoricalFocalLoss
from session_state import SessionState
from model_training_loop import ModelTrainer

st.set_option('deprecation.showPyplotGlobalUse', False)

session_state = SessionState.get(is_model_trained=False, trained_model=None, data_augmenter=None, model_trainer=None)

td = None
tl = None
vd = None
vl = None

st.sidebar.write("""
## Upload Training Data and Labels
""")

# Upload training set
training_data = st.sidebar.file_uploader("Upload Training Data", type=["csv"])
training_data_dims = st.sidebar.text_input('Reshape Training Data Dimensions (e.g. (40000,28,28,1)):')
if training_data_dims != "":
    td = pd.read_csv(training_data).to_numpy()
    td = td.reshape(eval(training_data_dims))

# Upload training labels
training_labels = st.sidebar.file_uploader("Upload Training Labels", type=["csv"])
training_label_dims = st.sidebar.text_input('Reshape Training Label Dimensions (e.g. (40000,)):')
if training_label_dims != "":
    tl = pd.read_csv(training_labels).to_numpy()
    tl = tl.reshape(eval(training_label_dims))

st.sidebar.write("""
## Upload Validation Data and Labels
""")

# Upload validation set
val_data = st.sidebar.file_uploader("Upload Validation Data", type=["csv"])
val_data_dims = st.sidebar.text_input('Reshape Validation Data Dimensions (e.g. (10000,28,28,1)):')
if val_data_dims != "":
    vd = pd.read_csv(val_data).to_numpy()
    vd = vd.reshape(eval(val_data_dims))

# Upload validation labels
val_labels = st.sidebar.file_uploader("Upload Validation Labels", type=["csv"])
val_label_dims = st.sidebar.text_input('Reshape Validation Label Dimensions (e.g. (10000,)):')
if val_label_dims != "":
    vl = pd.read_csv(val_labels).to_numpy()
    vl = vl.reshape(eval(val_label_dims))

placeholder = st.empty()

if session_state.is_model_trained is False:
    with placeholder.beta_container():
        # Display imbalance chart of training set
        st.write("""
                ## Training Dataset Instances By Class
                """)
        if training_label_dims != "":
            c1, c2 = st.beta_columns(2)
            y = LabelEncoder().fit_transform(tl)
            counter = Counter(y)
            counter_sorted = sorted(counter.items())
            classwise_percentage = ""
            for k, v in counter_sorted:
                per = v / len(y) * 100
                classwise_percentage += 'Class %d, Number of Samples=%d (%.2f%%)  \n' % (k, v, per)
            with c1:
                pyplot.bar(counter.keys(), counter.values())
                st.pyplot()
            with c2:
                st.write(classwise_percentage)
        else:
            st.markdown("<p style='color:red'>Training Data, Labels And Dimensions Not Provided</p>",
                        unsafe_allow_html=True)

        # Focal loss parameters
        st.write("""
                    ## Focal Loss Function Parameters
                    """)
        gamma = st.text_input('Gamma (e.g. 2):')
        if gamma != "":
            gamma = float(gamma)
        else:
            gamma = 2.0

        alpha = st.text_input('Alpha (e.g. {0:1, 1:2, 2:1}):')
        if alpha != "":
            alpha = eval(alpha)
        else:
            alpha = None

        from_logits = st.text_input('From Logits (True or False):')
        if from_logits != "":
            from_logits = bool(from_logits)
        else:
            from_logits = False

        focal_kwargs = st.text_input('Keyword Arguments (e.g. name="focal_loss",...):')
        if focal_kwargs != "":
            focal_kwargs = eval(focal_kwargs)
        else:
            focal_kwargs = {}

        loss_fn = SparseCategoricalFocalLoss(gamma=gamma, alpha=alpha, from_logits=from_logits, **focal_kwargs)

        # Restricted dynamic sampling parameters
        st.write("""
                    ## Restricted Dynamic Sampling Parameters
                    """)
        dynamic_sampling_start_epoch = st.text_input('Dynamic Sampling Start Epoch (e.g. 10):')
        if dynamic_sampling_start_epoch != "":
            dynamic_sampling_start_epoch = int(dynamic_sampling_start_epoch)
        else:
            dynamic_sampling_start_epoch = 0

        data_dimensions = st.text_input('Data Dimensions (Dimensions of one sample in dataset):')
        if data_dimensions != "":
            data_dimensions = eval(data_dimensions)

        batch_size = st.text_input('Batch Size (e.g. 64):')
        if batch_size != "":
            batch_size = int(batch_size)
        else:
            batch_size = 64

        data_augmenter = None
        data_augmenter_definintion = st.text_area(
            'Optionally, provide a data augmentation data generator with a flow() function defined. '
            'An example is given below. Please delete the text in the area '
            'below if you do not want to augment data for dynamic sampling.',
            value='import tensorflow\nfrom tensorflow.keras.preprocessing.image import ImageDataGenerator\n\n'
                  'data_augmenter = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, '
                  'shear_range=0.15, zoom_range=0.1, channel_shift_range=10., horizontal_flip=True)', height=53)
        exec(data_augmenter_definintion)
        session_state.data_augmenter = data_augmenter

        # Model definition and training
        st.write("""
                    ## Model Training
                    """)

        model: tf.keras.Model = None
        optimizer = None

        example_model_definition = "import tensorflow as tf\n\nmodel = tf.keras.models.Sequential([ # An example model" \
                                   "\n\ttf.keras.layers.Flatten(input_shape=(28, 28, 1)),\n\ttf.keras.layers.Dense(128, activation='relu')," \
                                   "\n\ttf.keras.layers.Dense(10, activation='softmax')\n])\n\noptimizer = tf.keras.optimizers.Adam() # An example optimizer"

        model_definition_code = st.text_area(
            'Type code here to define model. An example is given in the area below. Specify the optimizer as well.',
            value=example_model_definition, height=53)

        # Get model and optimizer variables from input code
        exec(model_definition_code)

        epochs = st.text_input('Training Epochs (e.g. 20):')
        if epochs != "":
            epochs = int(epochs)
        else:
            epochs = 5

        if st.button('Train Model'):
            classes = np.unique(tl) # All unique classes in the dataset
            train_generator = DynamicSamplingDataGenerator(x_train=td, y_train=tl, batch_size=batch_size,
                                                           dimensions=data_dimensions, classes=classes)
            validation_generator = DynamicSamplingDataGenerator(x_train=td, y_train=tl, batch_size=batch_size,
                                                                dimensions=data_dimensions, classes=classes)
            model_trainer = ModelTrainer()
            model_trainer.train_model(model, epochs, train_generator, validation_generator, loss_fn, optimizer,
                        dynamic_sampling_start_epoch, session_state)
            session_state.model_trainer = model_trainer

            with placeholder.beta_container():
                save_model = st.button('Save Trained Model ')
                st.info(model_trainer.final_status)
                for _ in range(16):
                    st.text("")

# Save trained model locally
if session_state.is_model_trained:
    with placeholder.beta_container():
        c1, c2 = st.beta_columns([1,3])
        with c1:
            save_model = st.button('Save Trained Model')
        st.info(session_state.model_trainer.final_status)
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        if save_model:
            directory = filedialog.askdirectory(master=root)
            if directory != "":
                with c2:
                    with st.spinner("Saving model to selected directory..."):
                        session_state.trained_model.save(os.path.join(directory, "model.h5"))
                        st.success("Model saved successfully to selected directory!")
