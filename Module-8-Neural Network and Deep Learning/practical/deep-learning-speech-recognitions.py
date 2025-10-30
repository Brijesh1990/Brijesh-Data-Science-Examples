import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed

# --- 1. Prepare Dummy Data for Illustration ---
# In a real scenario, X would be a sequence of Mel-Frequency Cepstral Coefficients (MFCCs)
# extracted from audio, and Y would be the corresponding transcript.

# Parameters:
num_samples = 100
sequence_length = 50  # Time steps (e.g., audio frames)
num_features = 40     # Feature dimension (e.g., number of MFCCs)
num_classes = 30      # Output classes (e.g., number of unique characters/phonemes)

# Dummy Input Data (Audio Features)
# Shape: (samples, time_steps, features)
X_train = np.random.rand(num_samples, sequence_length, num_features)

# Dummy Output Data (Target Sequence)
# Shape: (samples, time_steps, classes) - often used for Connectionist Temporal Classification (CTC) loss
Y_train = np.random.randint(0, num_classes, size=(num_samples, sequence_length))
# One-hot encode the output for this simplified example
Y_train_one_hot = tf.keras.utils.to_categorical(Y_train, num_classes=num_classes)


# --- 2. Define the Deep Learning Model (LSTM) ---
def build_speech_model(seq_len, feature_dim, output_classes):
    model = Sequential()

    # LSTM Layer: Processes the sequence of features
    # return_sequences=True is crucial for sequence-to-sequence tasks (like speech)
    # where an output is needed for every input time step.
    model.add(LSTM(
        units=128,                     # Number of LSTM units (hidden state dimension)
        input_shape=(seq_len, feature_dim),
        return_sequences=True,
        name='lstm_layer_1'
    ))

    # Another LSTM layer for deeper feature extraction
    model.add(LSTM(
        units=128,
        return_sequences=True,
        name='lstm_layer_2'
    ))

    # TimeDistributed: Applies the Dense layer independently to every time step
    # Maps the LSTM output (128 units) to the final class probabilities (30 classes)
    model.add(TimeDistributed(
        Dense(output_classes, activation='softmax'),
        name='output_layer'
    ))

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy', # A common loss for multi-class classification
        metrics=['accuracy']
    )
    return model

# --- 3. Instantiate and Summarize the Model ---
model = build_speech_model(sequence_length, num_features, num_classes)
print("--- Model Summary ---")
model.summary()
print("---------------------")

# --- 4. Train the Model (Conceptual Step) ---
# model.fit(X_train, Y_train_one_hot, epochs=10, batch_size=32, validation_split=0.1)

# Note: The actual training requires real data and would take significant time.
# This example is illustrative of the model's structure.