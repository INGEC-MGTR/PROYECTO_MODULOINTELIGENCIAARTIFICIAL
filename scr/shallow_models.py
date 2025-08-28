"""
Modelo de Aprendizaje Superficial para detección de intrusiones en telecomunicaciones
Random Forest con Optimización de Hiperparámetros

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

class ShallowLearningClassifier:
    def __init__(self, random_state=42):
        """
        Inicializar clasificador de aprendizaje superficial
        
        Args:
            random_state (int): Semilla para reproducibilidad
        """
        self.random_state = random_state
        self.models = {}
        self.best_params = {}
        self.training_history = {}
        
    def define_hyperparameter_grids(self):
        """
        Define espacios de búsqueda para optimización de hiperparámetros
        
        Returns:
            dict: Grids de hiperparámetros para cada algoritmo
        """
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', 0.3, 0.5],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            },
            
            'extra_trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', 0.3],
                'bootstrap': [True, False],
                'class_weight': ['balanced', None]
            },
            
            'decision_tree': {
                'max_depth': [10, 20, 30, 50, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4, 8],
                'max_features': ['sqrt', 'log2', None],
                'class_weight': ['balanced', None],
                'criterion': ['gini', 'entropy']
            },
            
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced', None],
                'max_iter': [1000, 2000, 5000]
            }
        }
        
        return param_grids
    
    def train_random_forest(self, X_train, y_train, optimize_hyperparams=True, cv_folds=5):
        """
        Entrenar Random Forest con optimización de hiperparámetros
        
        Args:
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Variable objetivo
            optimize_hyperparams (bool): Si optimizar hiperparámetros
            cv_folds (int): Número de folds para validación cruzada
            
        Returns:
            RandomForestClassifier: Modelo entrenado
        """
        print("=== ENTRENANDO RANDOM FOREST ===")
        start_time = time.time()
        
        if optimize_hyperparams:
            # Usar RandomizedSearchCV para eficiencia
            param_grid = self.define_hyperparameter_grids()['random_forest']
            
            # Distribuciones para RandomizedSearchCV
            param_dist = {
                'n_estimators': randint(100, 500),
                'max_depth': [10, 20, 30, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 8),
                'max_features': ['sqrt', 'log2', uniform(0.1, 0.5)],
                'bootstrap': [True, False],
                'class_weight': ['balanced', 'balanced_subsample', None]
            }
            
            rf_base = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                warm_start=False
            )
            
            # Optimización con RandomizedSearchCV (más eficiente que GridSearchCV)
            rf_search = RandomizedSearchCV(
                estimator=rf_base,
                param_distributions=param_dist,
                n_iter=50,  # Número de combinaciones a probar
                cv=cv_folds,
                scoring='f1_macro',  # Métrica balanceada para clases desbalanceadas
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
            
            rf_search.fit(X_train, y_train)
            best_rf = rf_search.best_estimator_
            self.best_params['random_forest'] = rf_search.best_params_
            
            print(f"Mejores parámetros encontrados:")
            for param, value in rf_search.best_params_.items():
                print(f"  {param}: {value}")
            
        else:
            best_rf = RandomForestClassifier(
                n_estimators=200,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            best_rf.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.training_history['random_forest'] = {
            'training_time': training_time,
            'feature_importance': best_rf.feature_importances_
        }
        
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Evaluar con validación cruzada
        cv_scores = cross_val_score(best_rf, X_train, y_train, 
                                  cv=cv_folds, scoring='f1_macro')
        print(f"F1-Score promedio (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.models['random_forest'] = best_rf
        return best_rf
    
    def train_extra_trees(self, X_train, y_train, cv_folds=5):
        """
        Entrenar Extra Trees como modelo de comparación
        
        Args:
            X_train (pd.DataFrame): Características de entrenamiento
            y_train (pd.Series): Variable objetivo
            cv_folds (int): Número de folds para validación cruzada
            
        Returns:
            ExtraTreesClassifier: Modelo entrenado
        """
        print("=== ENTRENANDO EXTRA TREES ===")
        start_time = time.time()
        
        # Configuración optimizada para detección de intrusiones
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=30,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=False,  # Extra Trees no usa bootstrap por defecto
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        et_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.training_history['extra_trees'] = {
            'training_time': training_time,
            'feature_importance': et_model.feature_importances_
        }
        
        print(f"Entrenamiento completado en {training_time:.2f} segundos")
        
        # Validación cruzada
        cv_scores = cross_val_score(et_model, X_train, y_train, 
                                  cv=cv_folds, scoring='f1_macro')
        print(f"F1-Score promedio (CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        self.models['extra_trees'] = et_model
        return et_model
    
    def evaluate_model(self, model, X_test, y_test, model_name, class_names=None):
        """
        Evaluación completa del modelo con métricas especializadas
        
        Args:
            model: Modelo entrenado
            X_test (pd.DataFrame): Características de prueba
            y_test (pd.Series): Variable objetivo de prueba
            model_name (str): Nombre del modelo
            class_names (list): Nombres de las clases
            
        Returns:
            dict: Diccionario con todas las métricas
        """
        print(f"=== EVALUANDO {model_name.upper()} ===")
        
        # Predicciones
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Predicciones probabilísticas para ROC-AUC
        try:
            y_pred_proba = model.predict_proba(X_test)
            if len(np.unique(y_test)) == 2:  # Clasificación binaria
                auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:  # Clasificación multiclase
                auc_score = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc_score = None
        
        # Métricas básicas
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Métricas promedio
        precision_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[0]
        recall_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[1]
        f1_macro = precision_recall_fscore_support(y_test, y_pred, average='macro')[2]
        
        precision_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[0]
        recall_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[1]
        f1_weighted = precision_recall_fscore_support(y_test, y_pred, average='weighted')[2]
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Reporte de clasificación
        class_report = classification_report(y_test, y_pred, 
                                           target_names=class_names,
                                           output_dict=True)
        
        # Métricas específicas para detección de intrusiones
        if len(np.unique(y_test)) == 2:  # Clasificación binaria
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
                'detection_rate': recall[1] if len(recall) > 1 else recall[0]  # TPR para ataques
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
            'intrusion_specific_metrics': intrusion_metrics
        }
        
        # Imprimir resultados principales
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        if auc_score:
            print(f"AUC-ROC: {auc_score:.4f}")
        print(f"Tiempo de predicción: {prediction_time:.4f} segundos")
        
        if intrusion_metrics:
            print(f"Tasa de Detección (Recall para ataques): {intrusion_metrics['detection_rate']:.4f}")
            print(f"Tasa de Falsos Positivos: {intrusion_metrics['false_positive_rate']:.4f}")
        
        return results
    
    def plot_confusion_matrix(self, cm, class_names, model_name, save_path=None):
        """
        Visualizar matriz de confusión
        
        Args:
            cm (np.array): Matriz de confusión
            class_names (list): Nombres de las clases
            model_name (str): Nombre del modelo
            save_path (str): Ruta para guardar la figura
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, model_name, 
                               top_k=20, save_path=None):
        """
        Visualizar importancia de características
        
        Args:
            model: Modelo con feature_importances_
            feature_names (list): Nombres de las características
            model_name (str): Nombre del modelo
            top_k (int): Número de características principales a mostrar
            save_path (str): Ruta para guardar la figura
        """
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_k]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Importancia de Características - {model_name}')
            plt.barh(range(top_k), importances[indices])
            plt.yticks(range(top_k), [feature_names[i] for i in indices])
            plt.xlabel('Importancia')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            # Retornar top características
            return [(feature_names[i], importances[i]) for i in indices]
        else:
            print(f"El modelo {model_name} no tiene importancia de características")
            return None
    
    def plot_roc_curve(self, model, X_test, y_test, model_name, save_path=None):
        """
        Plotear curva ROC para clasificación binaria
        
        Args:
            model: Modelo entrenado
            X_test (pd.DataFrame): Características de prueba
            y_test (pd.Series): Variable objetivo de prueba
            model_name (str): Nombre del modelo
            save_path (str): Ruta para guardar la figura
        """
        if len(np.unique(y_test)) == 2:  # Solo para clasificación binaria
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, linewidth=2, label=f'{model_name} (AUC = {auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('Tasa de Falsos Positivos')
                plt.ylabel('Tasa de Verdaderos Positivos')
                plt.title(f'Curva ROC - {model_name}')
                plt.legend(loc="lower right")
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"Error al generar curva ROC: {str(e)}")
    
    def save_model(self, model_name, save_path):
        """
        Guardar modelo entrenado
        
        Args:
            model_name (str): Nombre del modelo a guardar
            save_path (str): Ruta de guardado
        """
        if model_name in self.models:
            model_data = {
                'model': self.models[model_name],
                'best_params': self.best_params.get(model_name, {}),
                'training_history': self.training_history.get(model_name, {})
            }
            joblib.dump(model_data, save_path)
            print(f"Modelo {model_name} guardado en: {save_path}")
        else:
            print(f"Modelo {model_name} no encontrado")
    
    def load_model(self, model_name, load_path):
        """
        Cargar modelo guardado
        
        Args:
            model_name (str): Nombre del modelo
            load_path (str): Ruta de carga
        """
        try:
            model_data = joblib.load(load_path)
            self.models[model_name] = model_data['model']
            self.best_params[model_name] = model_data.get('best_params', {})
            self.training_history[model_name] = model_data.get('training_history', {})
            print(f"Modelo {model_name} cargado desde: {load_path}")
        except Exception as e:
            print(f"Error al cargar modelo: {str(e)}")

# Pipeline de entrenamiento y evaluación completo
def train_and_evaluate_shallow_models(processed_data, save_dir='models/shallow/'):
    """
    Pipeline completo para entrenar y evaluar modelos superficiales
    
    Args:
        processed_data (dict): Datos preprocesados
        save_dir (str): Directorio para guardar modelos
        
    Returns:
        dict: Resultados de evaluación de todos los modelos
    """
    # Extraer datos
    X_train = processed_data['X_train']
    X_test = processed_data['X_test']
    y_train = processed_data['y_binary_train']
    y_test = processed_data['y_binary_test']
    feature_names = processed_data['selected_features']
    
    # Inicializar clasificador
    shallow_classifier = ShallowLearningClassifier(random_state=42)
    
    # Entrenar Random Forest (modelo principal)
    rf_model = shallow_classifier.train_random_forest(
        X_train, y_train, optimize_hyperparams=True, cv_folds=5
    )
    
    # Entrenar Extra Trees (modelo de comparación)
    et_model = shallow_classifier.train_extra_trees(X_train, y_train, cv_folds=5)
    
    # Evaluar modelos
    results = {}
    
    # Evaluación Random Forest
    print("\n" + "="*50)
    rf_results = shallow_classifier.evaluate_model(
        rf_model, X_test, y_test, 'Random Forest', 
        class_names=['Normal', 'Attack']
    )
    results['random_forest'] = rf_results
    
    # Evaluación Extra Trees
    print("\n" + "="*50)
    et_results = shallow_classifier.evaluate_model(
        et_model, X_test, y_test, 'Extra Trees',
        class_names=['Normal', 'Attack']
    )
    results['extra_trees'] = et_results
    
    # Visualizaciones
    print("\n=== GENERANDO VISUALIZACIONES ===")
    
    # Matrices de confusión
    shallow_classifier.plot_confusion_matrix(
        rf_results['confusion_matrix'], ['Normal', 'Attack'], 'Random Forest'
    )
    
    # Importancia de características
    rf_importance = shallow_classifier.plot_feature_importance(
        rf_model, feature_names, 'Random Forest', top_k=15
    )
    
    # Curvas ROC
    shallow_classifier.plot_roc_curve(rf_model, X_test, y_test, 'Random Forest')
    
    # Guardar modelos
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    shallow_classifier.save_model('random_forest', 
                                f'{save_dir}/random_forest_model.pkl')
    shallow_classifier.save_model('extra_trees', 
                                f'{save_dir}/extra_trees_model.pkl')
    
    # Guardar resultados de evaluación
    joblib.dump(results, f'{save_dir}/shallow_evaluation_results.pkl')
    
    print(f"\n=== RESUMEN FINAL ===")
    print(f"Random Forest - Accuracy: {rf_results['accuracy']:.4f}, F1: {rf_results['f1_macro']:.4f}")
    print(f"Extra Trees - Accuracy: {et_results['accuracy']:.4f}, F1: {et_results['f1_macro']:.4f}")
    
    return results, shallow_classifier

# Ejemplo de uso principal
if __name__ == "__main__":
    # Cargar datos preprocesados
    import joblib
    processed_data = joblib.load('data/processed/unsw_nb15_processed.pkl')
    
    # Entrenar y evaluar modelos superficiales
    evaluation_results, classifier = train_and_evaluate_shallow_models(processed_data)
    
    print("Entrenamiento y evaluación de modelos superficiales completado.")
