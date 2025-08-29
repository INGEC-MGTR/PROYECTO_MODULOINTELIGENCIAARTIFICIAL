import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, LSTM, Conv1D, MaxPooling1D, Flatten, 
                                   Dropout, BatchNormalization, Input, 
                                   MultiHeadAttention, LayerNormalization,
                                   GlobalAveragePooling1D, Concatenate)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                       ModelCheckpoint, TensorBoard)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configurar TensorFlow para reproducibilidad
tf.random.set_seed(42)
np.random.seed(42)

class DeepLearningClassifier:
    def __init__(self, input_shape, num_classes, random_state=42):
        """
        Inicializar clasificador de aprendizaje profundo
        
        Args:
            input_shape (tuple): Forma de los datos de entrada
            num_classes (int): Número de clases
            random_state (int): Semilla para reproducibilidad
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.random_state = random_state
        self.models = {}
        self.histories = {}
        self.evaluation_results = {}
        
        # Configurar GPU si está disponible
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print(f"GPU detectada: {tf.config.list_physical_devices('GPU')}")
        else:
            print("Ejecutando en CPU")
    
    def create_cnn_lstm_hybrid(self, conv_filters=[64, 128, 256], lstm_units=[128, 64],
                              dense_units=[512, 256], dropout_rate=0.3, l2_reg=0.001):
        """
        Crear arquitectura CNN-LSTM híbrida para análisis temporal-espacial
        Inspirada en investigación especializada en tráfico de red
        
        Args:
            conv_filters (list): Filtros para capas convolucionales
            lstm_units (list): Unidades para capas LSTM
            dense_units (list): Unidades para capas densas
            dropout_rate (float): Tasa de dropout
            l2_reg (float): Regularización L2
            
        Returns:
            tensorflow.keras.Model: Modelo compilado
        """
        print("=== CREANDO MODELO CNN-LSTM HÍBRIDO ===")
        
        # Entrada principal
        input_layer = Input(shape=self.input_shape, name='main_input')
        
        # Expandir dimensiones para convolución 1D
        x = tf.expand_dims(input_layer, axis=-1)
        
        # Bloque Convolucional para extracción de características espaciales
        for i, filters in enumerate(conv_filters):
            x = Conv1D(filters=filters, 
                      kernel_size=3, 
                      activation='relu',
                      padding='same',
                      kernel_regularizer=l2(l2_reg),
                      name=f'conv1d_{i+1}')(x)
            x = BatchNormalization(name=f'bn_conv_{i+1}')(x)
            x = MaxPooling1D(pool_size=2, name=f'maxpool_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_conv_{i+1}')(x)
        
        # Bloque LSTM para análisis temporal
        for i, units in enumerate(lstm_units):
            return_sequences = i < len(lstm_units) - 1  # Solo la última LSTM no retorna secuencias
            x = LSTM(units=units,
                    return_sequences=return_sequences,
                    dropout=dropout_rate,
                    recurrent_dropout=dropout_rate,
                    kernel_regularizer=l2(l2_reg),
                    name=f'lstm_{i+1}')(x)
            x = BatchNormalization(name=f'bn_lstm_{i+1}')(x)
        
        # Capas densas finales
        for i, units in enumerate(dense_units):
            x = Dense(units=units,
                     activation='relu',
                     kernel_regularizer=l2(l2_reg),
                     name=f'dense_{i+1}')(x)
            x = BatchNormalization(name=f'bn_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_dense_{i+1}')(x)
        
        # Capa de salida
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid', name='binary_output')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        else:
            output = Dense(self.num_classes, activation='softmax', name='multiclass_output')(x)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy', tf.keras.metrics.CategoricalAccuracy()]
        
        # Crear modelo
        model = Model(inputs=input_layer, outputs=output, name='CNN_LSTM_Hybrid')
        
        # Compilar modelo con optimizador adaptativo
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
        
        model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=metrics)
        
        print(f"Modelo creado con {model.count_params():,} parámetros")
        return model
    
    def create_attention_enhanced_model(self, lstm_units=[128, 64], dense_units=[512, 256],
                                       dropout_rate=0.3, l2_reg=0.001):
        """
        Crear modelo con mecanismo de atención para características importantes
        Basado en Zhang et al. (2023) - attention mechanism for network traffic
        
        Args:
            lstm_units (list): Unidades LSTM
            dense_units (list): Unidades densas
            dropout_rate (float): Tasa de dropout
            l2_reg (float): Regularización L2
            
        Returns:
            tensorflow.keras.Model: Modelo con atención
        """
        print("=== CREANDO MODELO CON MECANISMO DE ATENCIÓN ===")
        
        input_layer = Input(shape=self.input_shape, name='attention_input')
        
        # Expandir para análisis secuencial
        x = tf.expand_dims(input_layer, axis=1)  # Añadir dimensión temporal
        
        # Capas LSTM bidireccionales
        x = tf.keras.layers.Bidirectional(
            LSTM(lstm_units[0], return_sequences=True, dropout=dropout_rate),
            name='bidirectional_lstm_1'
        )(x)
        x = BatchNormalization(name='bn_bi_lstm_1')(x)
        
        if len(lstm_units) > 1:
            x = tf.keras.layers.Bidirectional(
                LSTM(lstm_units[1], return_sequences=True, dropout=dropout_rate),
                name='bidirectional_lstm_2'
            )(x)
            x = BatchNormalization(name='bn_bi_lstm_2')(x)
        
        # Mecanismo de atención multi-cabeza
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64,
            dropout=dropout_rate,
            name='multi_head_attention'
        )(x, x)
        
        # Conexión residual y normalización
        x = x + attention_output
        x = LayerNormalization(name='layer_norm_attention')(x)
        
        # Pooling global
        x = GlobalAveragePooling1D(name='global_avg_pooling')(x)
        
        # Capas densas finales
        for i, units in enumerate(dense_units):
            x = Dense(units=units,
                     activation='relu',
                     kernel_regularizer=l2(l2_reg),
                     name=f'attention_dense_{i+1}')(x)
            x = BatchNormalization(name=f'bn_attention_dense_{i+1}')(x)
            x = Dropout(dropout_rate, name=f'dropout_attention_{i+1}')(x)
        
        # Capa de salida
        if self.num_classes == 2:
            output = Dense(1, activation='sigmoid', name='attention_binary_output')(x)
            loss = 'binary_crossentropy'
        else:
            output = Dense(self.num_classes, activation='softmax', name='attention_multiclass_output')(x)
            loss = 'categorical_crossentropy'
        
        model = Model(inputs=input_layer, outputs=output, name='Attention_Enhanced_Model')
        
        optimizer = Adam(learning_rate=0.0005)  # Learning rate más bajo para modelo complejo
        model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=['accuracy'])
        
        print(f"Modelo con atención creado con {model.count_params():,} parámetros")
        return model
    
    def create_simple_deep_model(self, hidden_layers=[512, 256, 128, 64], 
                                dropout_rate=0.4, l2_reg=0.001):
        """
        Crear modelo profundo simple para comparación (MLP profundo)
        
        Args:
            hidden_layers (list): Neuronas en capas ocultas
            dropout_rate (float): Tasa de dropout
            l2_reg (float): Regularización L2
            
        Returns:
            tensorflow.keras.Model: Modelo MLP profundo
        """
        print("=== CREANDO MODELO MLP PROFUNDO ===")
        
        model = Sequential(name='Deep_MLP')
        model.add(Input(shape=self.input_shape, name='mlp_input'))
        
        # Capas ocultas con regularización
        for i, units in enumerate(hidden_layers):
            model.add(Dense(units=units,
                           activation='relu',
                           kernel_regularizer=l2(l2_reg),
                           name=f'mlp_hidden_{i+1}'))
            model.add(BatchNormalization(name=f'bn_mlp_{i+1}'))
            model.add(Dropout(dropout_rate, name=f'dropout_mlp_{i+1}'))
        
        # Capa de salida
        if self.num_classes == 2:
            model.add(Dense(1, activation='sigmoid', name='mlp_binary_output'))
            loss = 'binary_crossentropy'
        else:
            model.add(Dense(self.num_classes, activation='softmax', name='mlp_multiclass_output'))
            loss = 'categorical_crossentropy'
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer,
                     loss=loss,
                     metrics=['accuracy'])
        
        print(f"Modelo MLP profundo creado con {model.count_params():,} parámetros")
        return model
    
    def prepare_callbacks(self, model_name, save_dir='models/deep/'):
        """
        Preparar callbacks para entrenamiento
        
        Args:
            model_name (str): Nombre del modelo
            save_dir (str): Directorio de guardado
            
        Returns:
            list: Lista de callbacks
        """
        os.makedirs(save_dir, exist_ok=True)
        
        callbacks = [
            # Parada temprana para evitar sobreajuste
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1,
                name='early_stopping'
            ),
            
            # Reducir learning rate cuando se estanque
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                name='reduce_lr'
            ),
            
            # Guardar mejor modelo
            ModelCheckpoint(
                filepath=f'{save_dir}/{model_name}_best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
                name='model_checkpoint'
            ),
            
            # TensorBoard para monitoreo
            TensorBoard(
                log_dir=f'{save_dir}/tensorboard_logs/{model_name}_{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                name='tensorboard'
            )
        ]
        
        return callbacks
    
    def train_model(self, model, X_train, y_train, X_val, y_val, model_name,
                   epochs=100, batch_size=256, verbose=1):
        """
        Entrenar modelo de aprendizaje profundo
        
        Args:
            model: Modelo a entrenar
            X_train, y_train: Datos de entrenamiento
            X_val, y_val: Datos de validación
            model_name (str): Nombre del modelo
            epochs (int): Número de épocas
            batch_size (int): Tamaño de lote
            verbose (int): Verbosidad
            
        Returns:
            tensorflow.keras.callbacks.History: Historia de entrenamiento
        """
        print(f"=== ENTRENANDO {model_name.upper()} ===")
        print(f"Forma de entrenamiento: {X_train.shape}")
        print(f"Forma de validación: {X_val.shape}")
        
        # Preparar callbacks
        callbacks = self.prepare_callbacks(model_name)
        
        # Entrenar modelo
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        training_time = time.time() - start_time
        
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        print(f"Mejor val_loss: {min(history.history['val_loss']):.4f}")
        print(f"Mejor val_accuracy: {max(history.history.get('val_accuracy', [0])):.4f}")
        
        # Guardar modelo y historia
        self.models[model_name] = model
        self.histories[model_name] = {
            'history': history.history,
            'training_time': training_time,
            'epochs_trained': len(history.history['loss'])
        }
        
        return history
    
    def evaluate_deep_model(self, model, X_test, y_test, model_name, class_names=None):
        """
        Evaluación completa del modelo profundo
        
        Args:
            model: Modelo entrenado
            X_test, y_test: Datos de prueba
            model_name (str): Nombre del modelo
            class_names (list): Nombres de las clases
            
        Returns:
            dict: Resultados de evaluación
        """
        print(f"=== EVALUANDO {model_name.upper()} ===")
        
        # Predicciones
        start_time = time.time()
        y_pred_proba = model.predict(X_test, verbose=0)
        prediction_time = time.time() - start_time
        
        # Convertir predicciones según el tipo de clasificación
        if self.num_classes == 2:
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            auc_score = roc_auc_score(y_test, y_pred_proba.flatten())
        else:
            y_pred = np.argmax(y_pred_proba, axis=1)
            # Para multiclase, convertir y_test si está en formato categórico
            if len(y_test.shape) > 1:
                y_test_labels = np.argmax(y_test, axis=1)
            else:
                y_test_labels = y_test
            auc_score = roc_auc_score(y_test_labels, y_pred_proba, multi_class='ovr')
        
        # Métricas de evaluación
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        test_labels = y_test_labels if self.num_classes > 2 else y_test
        
        accuracy = accuracy_score(test_labels, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            test_labels, y_pred, average=None, zero_division=0
        )
        
        # Métricas promedio
        precision_macro = precision_recall_fscore_support(test_labels, y_pred, average='macro', zero_division=0)[0]
        recall_macro = precision_recall_fscore_support(test_labels, y_pred, average='macro', zero_division=0)[1]
        f1_macro = precision_recall_fscore_support(test_labels, y_pred, average='macro', zero_division=0)[2]
        
        precision_weighted = precision_recall_fscore_support(test_labels, y_pred, average='weighted', zero_division=0)[0]
        recall_weighted = precision_recall_fscore_support(test_labels, y_pred, average='weighted', zero_division=0)[1]
        f1_weighted = precision_recall_fscore_support(test_labels, y_pred, average='weighted', zero_division=0)[2]
        
        # Matriz de confusión
        cm = confusion_matrix(test_labels, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(
            test_labels, y_pred, 
            target_names=class_names, 
            output_dict=True, 
            zero_division=0
        )
        
        # Métricas específicas para detección de intrusiones
        if self.num_classes == 2:
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
            false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            intrusion_metrics = {
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn),
                'specificity': specificity,
                'false_positive_rate': false_positive_rate,
                'false_negative_rate': false_negative_rate,
                'detection_rate': recall[1] if len(recall) > 1 else recall[0]
            }
        else:
            intrusion_metrics = {}
        
        # Compilar resultados
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'auc_score': auc_score,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'per_class_metrics': {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            },
            'prediction_time': prediction_time,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'intrusion_specific_metrics': intrusion_metrics
        }
        
        # Imprimir resultados principales
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Tiempo de predicción: {prediction_time:.4f} segundos")
        
        if intrusion_metrics:
            print(f"Tasa de Detección (Recall para ataques): {intrusion_metrics['detection_rate']:.4f}")
            print(f"Tasa de Falsos Positivos: {intrusion_metrics['false_positive_rate']:.4f}")
        
        self.evaluation_results[model_name] = results
        return results
    
    def plot_training_history(self, model_name, save_path=None):
        """
        Visualizar historia de entrenamiento
        
        Args:
            model_name (str): Nombre del modelo
            save_path (str): Ruta para guardar figura
        """
        if model_name not in self.histories:
            print(f"Historia no encontrada para {model_name}")
            return
        
        history = self.histories[model_name]['history']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Historia de Entrenamiento - {model_name}', fontsize=16)
        
        # Loss
        axes[0, 0].plot(history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Pérdida (Loss)')
        axes[0, 0].set_xlabel('Época')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Precisión (Accuracy)')
        axes[0, 1].set_xlabel('Época')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning Rate (si está disponible)
        if 'lr' in history:
            axes[1, 0].plot(history['lr'], linewidth=2, color='orange')
            axes[1, 0].set_title('Tasa de Aprendizaje')
            axes[1, 0].set_xlabel('Época')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Learning Rate\nno disponible', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Resumen de métricas finales
        final_train_acc = history['accuracy'][-1]
        final_val_acc = history['val_accuracy'][-1]
        final_train_loss = history['loss'][-1]
        final_val_loss = history['val_loss'][-1]
        
        summary_text = f"""Métricas Finales:
        
Training Accuracy: {final_train_acc:.4f}
Validation Accuracy: {final_val_acc:.4f}
Training Loss: {final_train_loss:.4f}
Validation Loss: {final_val_loss:.4f}

Épocas entrenadas: {len(history['loss'])}
Tiempo total: {self.histories[model_name]['training_time']:.2f}s"""
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_architecture(self, model, model_name, save_path=None):
        """
        Visualizar arquitectura del modelo
        
        Args:
            model: Modelo de Keras
            model_name (str): Nombre del modelo
            save_path (str): Ruta para guardar
        """
        try:
            tf.keras.utils.plot_model(
                model,
                to_file=save_path if save_path else f'{model_name}_architecture.png',
                show_shapes=True,
                show_layer_names=True,
                rankdir='TB',
                expand_nested=True,
                dpi=96
            )
            print(f"Arquitectura del modelo guardada: {model_name}")
        except Exception as e:
            print(f"No se pudo generar diagrama de arquitectura: {str(e)}")
            
        # Resumen del modelo
        print(f"\n=== RESUMEN DE ARQUITECTURA - {model_name.upper()} ===")
        model.summary()
    
    def save_deep_model(self, model_name, save_dir='models/deep/'):
        """
        Guardar modelo profundo completo
        
        Args:
            model_name (str): Nombre del modelo
            save_dir (str): Directorio de guardado
        """
        if model_name not in self.models:
            print(f"Modelo {model_name} no encontrado")
            return
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Guardar modelo completo
        model_path = f'{save_dir}/{model_name}_complete_model.h5'
        self.models[model_name].save(model_path)
        
        # Guardar historia y resultados
        metadata = {
            'history': self.histories.get(model_name, {}),
            'evaluation_results': self.evaluation_results.get(model_name, {}),
            'model_config': self.models[model_name].get_config(),
            'input_shape': self.input_shape,
            'num_classes': self.num_classes
        }
        
        metadata_path = f'{save_dir}/{model_name}_metadata.pkl'
        joblib.dump(metadata, metadata_path)
        
        print(f"Modelo {model_name} guardado en: {model_path}")
        print(f"Metadatos guardados en: {metadata_path}")

# Pipeline completo para entrenamiento y evaluación de modelos profundos
def train_and_evaluate_deep_models(processed_data, save_dir='models/deep/'):
    """
    Pipeline completo para entrenar y evaluar modelos de aprendizaje profundo
    
    Args:
        processed_data (dict): Datos preprocesados
        save_dir (str): Directorio para guardar modelos
        
    Returns:
        tuple: (resultados_evaluacion, clasificador)
    """
    # Extraer y preparar datos
    X_train = processed_data['X_train'].values
    X_test = processed_data['X_test'].values
    y_train = processed_data['y_binary_train'].values
    y_test = processed_data['y_binary_test'].values
    feature_names = processed_data['selected_features']
    
    # División adicional para validación
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Forma de datos:")
    print(f"  Entrenamiento: {X_train_split.shape}")
    print(f"  Validación: {X_val.shape}")
    print(f"  Prueba: {X_test.shape}")
    
    # Inicializar clasificador profundo
    input_shape = (X_train_split.shape[1],)
    num_classes = 2  # Clasificación binaria
    
    deep_classifier = DeepLearningClassifier(
        input_shape=input_shape,
        num_classes=num_classes,
        random_state=42
    )
    
    # Crear y entrenar modelos
    print("\n" + "="*60)
    
    # 1. Modelo CNN-LSTM Híbrido (principal)
    print("CREANDO Y ENTRENANDO MODELO CNN-LSTM HÍBRIDO")
    cnn_lstm_model = deep_classifier.create_cnn_lstm_hybrid(
        conv_filters=[64, 128, 256],
        lstm_units=[128, 64],
        dense_units=[512, 256],
        dropout_rate=0.3,
        l2_reg=0.001
    )
    
    # Entrenar CNN-LSTM
    cnn_lstm_history = deep_classifier.train_model(
        cnn_lstm_model, X_train_split, y_train_split, X_val, y_val,
        'cnn_lstm_hybrid', epochs=50, batch_size=256
    )
    
    # 2. Modelo con Mecanismo de Atención
    print("\n" + "="*60)
    print("CREANDO Y ENTRENANDO MODELO CON ATENCIÓN")
    attention_model = deep_classifier.create_attention_enhanced_model(
        lstm_units=[128, 64],
        dense_units=[512, 256],
        dropout_rate=0.3,
        l2_reg=0.001
    )
    
    # Entrenar modelo con atención
    attention_history = deep_classifier.train_model(
        attention_model, X_train_split, y_train_split, X_val, y_val,
        'attention_enhanced', epochs=50, batch_size=256
    )
    
    # 3. Modelo MLP Profundo (comparación)
    print("\n" + "="*60)
    print("CREANDO Y ENTRENANDO MODELO MLP PROFUNDO")
    mlp_model = deep_classifier.create_simple_deep_model(
        hidden_layers=[512, 256, 128, 64],
        dropout_rate=0.4,
        l2_reg=0.001
    )
    
    # Entrenar MLP profundo
    mlp_history = deep_classifier.train_model(
        mlp_model, X_train_split, y_train_split, X_val, y_val,
        'deep_mlp', epochs=50, batch_size=256
    )
    
    # Evaluación de modelos
    print("\n" + "="*60)
    print("EVALUANDO MODELOS PROFUNDOS")
    
    evaluation_results = {}
    class_names = ['Normal', 'Attack']
    
    # Evaluar CNN-LSTM
    cnn_lstm_results = deep_classifier.evaluate_deep_model(
        cnn_lstm_model, X_test, y_test, 'CNN-LSTM Hybrid', class_names
    )
    evaluation_results['cnn_lstm_hybrid'] = cnn_lstm_results
    
    # Evaluar modelo con atención
    attention_results = deep_classifier.evaluate_deep_model(
        attention_model, X_test, y_test, 'Attention Enhanced', class_names
    )
    evaluation_results['attention_enhanced'] = attention_results
    
    # Evaluar MLP profundo
    mlp_results = deep_classifier.evaluate_deep_model(
        mlp_model, X_test, y_test, 'Deep MLP', class_names
    )
    evaluation_results['deep_mlp'] = mlp_results
    
    # Visualizaciones
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    # Historias de entrenamiento
    deep_classifier.plot_training_history('cnn_lstm_hybrid')
    deep_classifier.plot_training_history('attention_enhanced')
    deep_classifier.plot_training_history('deep_mlp')
    
    # Arquitecturas de modelos
    deep_classifier.plot_model_architecture(cnn_lstm_model, 'cnn_lstm_hybrid')
    
    # Matrices de confusión
    from sklearn.metrics import ConfusionMatrixDisplay
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_results = [
        (cnn_lstm_results, 'CNN-LSTM Hybrid'),
        (attention_results, 'Attention Enhanced'),
        (mlp_results, 'Deep MLP')
    ]
    
    for i, (results, title) in enumerate(models_results):
        cm = results['confusion_matrix']
        ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=axes[i])
        axes[i].set_title(f'Matriz de Confusión\n{title}')
    
    plt.tight_layout()
    plt.show()
    
    # Guardar modelos
    print("\n=== GUARDANDO MODELOS ===")
    os.makedirs(save_dir, exist_ok=True)
    
    deep_classifier.save_deep_model('cnn_lstm_hybrid', save_dir)
    deep_classifier.save_deep_model('attention_enhanced', save_dir)
    deep_classifier.save_deep_model('deep_mlp', save_dir)
    
    # Guardar resultados de evaluación
    joblib.dump(evaluation_results, f'{save_dir}/deep_evaluation_results.pkl')
    
    # Resumen final
    print(f"\n=== RESUMEN FINAL DE MODELOS PROFUNDOS ===")
    print(f"CNN-LSTM Hybrid - Accuracy: {cnn_lstm_results['accuracy']:.4f}, F1: {cnn_lstm_results['f1_macro']:.4f}")
    print(f"Attention Enhanced - Accuracy: {attention_results['accuracy']:.4f}, F1: {attention_results['f1_macro']:.4f}")
    print(f"Deep MLP - Accuracy: {mlp_results['accuracy']:.4f}, F1: {mlp_results['f1_macro']:.4f}")
    
    return evaluation_results, deep_classifier

# Ejemplo de uso principal
if __name__ == "__main__":
    # Cargar datos preprocesados
    processed_data = joblib.load('data/processed/unsw_nb15_processed.pkl')
    
    # Entrenar y evaluar modelos profundos
    deep_evaluation_results, deep_classifier = train_and_evaluate_deep_models(processed_data)
    
    print("Entrenamiento y evaluación de modelos profundos completado.")
