import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os
import joblib

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="glorot_uniform",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[1], 1),
                                 initializer="zeros",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        transformed = K.dot(inputs, self.W)
        e = K.sum(transformed, axis=-1, keepdims=True)
        e = K.tanh(e + self.b)
        
        attention_weights = K.softmax(e, axis=1)
        context_vector = attention_weights * inputs
        context_vector = K.sum(context_vector, axis=1)
        
        return context_vector, attention_weights

    def compute_output_shape(self, input_shape):
        return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1], 1)]

def load_data(file_path, test_size=0.2, val_size=0.2):
    data = pd.read_excel(file_path)
    print(f"Dataset loaded successfully with shape: {data.shape}")

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].values

    le = LabelEncoder()
    y = le.fit_transform(y)
    fault_types = le.classes_
    print(f"Fault types: {fault_types}")
    num_classes = len(np.unique(y))
    print(f"Number of classes: {num_classes}")
    y_cat = to_categorical(y, num_classes=num_classes)

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_cat, test_size=(test_size + val_size), stratify=y_cat, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    return X_train, X_val, X_test, y_train, y_val, y_test, num_classes, le, scaler, fault_types

def create_attention_lstm_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = LayerNormalization()(lstm_out)
    attention_out, att_weights = AttentionLayer()(lstm_out)
    dense1 = Dense(64, activation='relu')(attention_out)
    dropout1 = Dropout(0.3)(dense1)
    dense2 = Dense(32, activation='relu')(dropout1)
    dropout2 = Dropout(0.2)(dense2)
    outputs = Dense(num_classes, activation='softmax')(dropout2)

    model = Model(inputs=inputs, outputs=outputs)
    model.att_model = Model(inputs=inputs, outputs=att_weights)
    return model

def apply_label_smoothing(y, factor=0.1):
    return y * (1 - factor) + (factor / y.shape[1])

def train_model(model, X_train, y_train, X_val, y_val, model_path):
    # Apply label smoothing for better generalization
    y_train_smooth = apply_label_smoothing(y_train)
    y_val_smooth = apply_label_smoothing(y_val)

    os.makedirs(model_path, exist_ok=True)
    checkpoint = ModelCheckpoint(
        os.path.join(model_path, 'best_model.keras'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    )

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Use class weights to handle imbalance
    y_train_labels = np.argmax(y_train, axis=1)
    class_counts = np.bincount(y_train_labels)
    total_samples = len(y_train_labels)
    class_weights = {i: total_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
    
    # Focal loss parameters built into fit via class_weights
    history = model.fit(
        X_train, y_train_smooth,
        validation_data=(X_val, y_val_smooth),
        epochs=10,
        batch_size=32,
        callbacks=[checkpoint, early_stop],
        class_weight=class_weights,
        verbose=1
    )
    
    # Load the best model (saved by checkpoint callback)
    if os.path.exists(os.path.join(model_path, 'best_model.keras')):
        model = load_model(os.path.join(model_path, 'best_model.keras'), 
                          custom_objects={'AttentionLayer': AttentionLayer})
    
    return model, history

def evaluate_model(model, X_test, y_test, le):
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=le.classes_))
    return y_pred_prob, y_true, y_pred

def save_model_artifacts(model, scaler, le, model_path):
    os.makedirs(model_path, exist_ok=True)
    # Save model and preprocessing objects
    model.save(os.path.join(model_path, 'final_model.keras'))
    joblib.dump(scaler, os.path.join(model_path, 'scaler.pkl'))
    joblib.dump(le, os.path.join(model_path, 'label_encoder.pkl'))
    print(f"\nModel and artifacts saved to {model_path}")

def predict_fault(input_data, model, scaler, le):
    """
    Function for making predictions with the model
    
    Parameters:
    input_data: numpy array of input features
    model: trained model
    scaler: fitted StandardScaler
    le: fitted LabelEncoder
    
    Returns:
    predicted_class: The predicted fault type
    probabilities: Probabilities for each class
    """
    # Preprocess input data
    input_data_scaled = scaler.transform(input_data.reshape(1, -1))
    input_data_reshaped = input_data_scaled.reshape(1, 1, input_data_scaled.shape[1])
    
    # Make prediction
    pred_probs = model.predict(input_data_reshaped)[0]
    pred_class_idx = np.argmax(pred_probs)
    predicted_class = le.inverse_transform([pred_class_idx])[0]
    
    # Return prediction and probabilities
    result = {
        'fault_type': predicted_class,
        'probabilities': {le.inverse_transform([i])[0]: float(prob) for i, prob in enumerate(pred_probs)}
    }
    
    return result

def main(file_path=None, model_path='model'):
    if file_path is None:
        # Default file path if not provided
        file_path = r"C:\Users\jovik\OneDrive\Desktop\TLFaultDataset.xlsx"
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, le, scaler, fault_types = load_data(file_path)
    
    # Create model
    model = create_attention_lstm_model((X_train.shape[1], X_train.shape[2]), num_classes)
    model.summary()
    
    # Train
    model, history = train_model(model, X_train, y_train, X_val, y_val, model_path)
    
    # Evaluate
    evaluate_model(model, X_test, y_test, le)
    
    # Save model and artifacts
    save_model_artifacts(model, scaler, le, model_path)
    
    print(f"Available fault types: {fault_types}")
    return model, scaler, le, fault_types

if __name__ == "__main__":
    main()