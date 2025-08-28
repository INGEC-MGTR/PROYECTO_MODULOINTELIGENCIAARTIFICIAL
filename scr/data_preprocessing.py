
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, chi2
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class UNSWDataPreprocessor:
    def __init__(self, dataset_path):
        """
        Inicializa el preprocesador para UNSW-NB15
        
        Args:
            dataset_path (str): Ruta al archivo CSV del dataset
        """
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        
        # Nombres de las características del UNSW-NB15 según la documentación oficial
        self.unsw_features = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
            'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
            'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
            'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime',
            'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
            'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd',
            'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm',
            'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
    
    def load_data(self):
        """
        Carga el dataset UNSW-NB15
        
        Returns:
            pd.DataFrame: Dataset cargado
        """
        try:
            # Intentar cargar con nombres de columnas predefinidos
            df = pd.read_csv(self.dataset_path, low_memory=False, na_values=['', ' ', 'nan', 'NaN'])
            print(f"Dataset cargado exitosamente: {df.shape}")
            return df
        except Exception as e:
            print(f"Error cargando dataset: {str(e)}")
            # Si falla, crear un dataset de muestra
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self):
        """Crear dataset sintético si no se puede cargar el original"""
        print("Creando dataset sintético para demostración...")
        np.random.seed(42)
        n_samples = 10000
        
        data = {
            'dur': np.random.exponential(10, n_samples),
            'sbytes': np.random.exponential(1000, n_samples),
            'dbytes': np.random.exponential(500, n_samples),
            'sttl': np.random.randint(1, 255, n_samples),
            'dttl': np.random.randint(1, 255, n_samples),
            'sloss': np.random.poisson(0.1, n_samples),
            'dloss': np.random.poisson(0.1, n_samples),
            'sload': np.random.exponential(1000, n_samples),
            'dload': np.random.exponential(500, n_samples),
            'spkts': np.random.poisson(10, n_samples),
            'dpkts': np.random.poisson(8, n_samples),
            'swin': np.random.randint(0, 65536, n_samples),
            'dwin': np.random.randint(0, 65536, n_samples),
            'stcpb': np.random.randint(0, 1000000, n_samples),
            'dtcpb': np.random.randint(0, 1000000, n_samples),
            'smeansz': np.random.exponential(100, n_samples),
            'dmeansz': np.random.exponential(80, n_samples),
            'trans_depth': np.random.randint(0, 10, n_samples),
            'res_bdy_len': np.random.exponential(500, n_samples),
            'sjit': np.random.exponential(1, n_samples),
            'djit': np.random.exponential(1, n_samples),
            'stime': np.random.uniform(0, 1000000, n_samples),
            'ltime': np.random.uniform(0, 1000000, n_samples),
            'sintpkt': np.random.exponential(0.1, n_samples),
            'dintpkt': np.random.exponential(0.1, n_samples),
            'tcprtt': np.random.exponential(0.01, n_samples),
            'synack': np.random.exponential(0.01, n_samples),
            'ackdat': np.random.exponential(0.01, n_samples),
            'is_sm_ips_ports': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'ct_state_ttl': np.random.randint(0, 100, n_samples),
            'ct_flw_http_mthd': np.random.randint(0, 10, n_samples),
            'is_ftp_login': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
            'ct_ftp_cmd': np.random.randint(0, 5, n_samples),
            'ct_srv_src': np.random.randint(0, 50, n_samples),
            'ct_srv_dst': np.random.randint(0, 50, n_samples),
            'ct_dst_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_ltm': np.random.randint(0, 100, n_samples),
            'ct_src_dport_ltm': np.random.randint(0, 50, n_samples),
            'ct_dst_sport_ltm': np.random.randint(0, 50, n_samples),
            'ct_dst_src_ltm': np.random.randint(0, 100, n_samples),
        }
        
        # Generar etiquetas
        labels = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        data['label'] = labels
        
        attack_categories = ['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 
                           'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms']
        attack_cats = []
        for label in labels:
            if label == 0:
                attack_cats.append('Normal')
            else:
                attack_cats.append(np.random.choice(attack_categories[1:]))
        data['attack_cat'] = attack_cats
        
        return pd.DataFrame(data)
    
    def explore_data(self, df):
        """
        Análisis exploratorio inicial del dataset
        
        Args:
            df (pd.DataFrame): Dataset a analizar
            
        Returns:
            dict: Resumen estadístico
        """
        exploration_results = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        # Análisis de la variable objetivo
        if 'label' in df.columns:
            exploration_results['label_distribution'] = df['label'].value_counts().to_dict()
        elif 'attack_cat' in df.columns:
            exploration_results['attack_distribution'] = df['attack_cat'].value_counts().to_dict()
        
        # Estadísticas descriptivas para columnas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exploration_results['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        print("=== ANÁLISIS EXPLORATORIO COMPLETADO ===")
        print(f"Forma del dataset: {exploration_results['shape']}")
        print(f"Valores faltantes totales: {sum(exploration_results['missing_values'].values())}")
        print(f"Duplicados: {exploration_results['duplicates']}")
        print(f"Uso de memoria: {exploration_results['memory_usage']:.2f} MB")
        
        return exploration_results
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Manejo de valores faltantes
        
        Args:
            df (pd.DataFrame): Dataset con valores faltantes
            strategy (str): Estrategia de imputación ('mean', 'median', 'mode')
            
        Returns:
            pd.DataFrame: Dataset con valores imputados
        """
        df_clean = df.copy()
        
        # Separar columnas numéricas y categóricas
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        
        # Imputación para columnas numéricas
        if len(numeric_cols) > 0:
            print(f"Imputando {len(numeric_cols)} columnas numéricas...")
            imputer_num = SimpleImputer(strategy=strategy)
            
            # Uso de loc para asignación segura
            numeric_data_imputed = imputer_num.fit_transform(df_clean[numeric_cols])
            df_clean.loc[:, numeric_cols] = numeric_data_imputed
        
        # Imputación para columnas categóricas
        if len(categorical_cols) > 0:
            print(f"Imputando {len(categorical_cols)} columnas categóricas...")
            imputer_cat = SimpleImputer(strategy='most_frequent')
            
            # CORRECCIÓN: Usar loc para asignación segura
            categorical_data_imputed = imputer_cat.fit_transform(df_clean[categorical_cols])
            df_clean.loc[:, categorical_cols] = categorical_data_imputed
        
        print(f"Valores faltantes después de imputación: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def encode_categorical_features(self, df):
        """
        Codificación de variables categóricas
        
        Args:
            df (pd.DataFrame): Dataset con variables categóricas
            
        Returns:
            pd.DataFrame: Dataset con variables codificadas
        """
        df_encoded = df.copy()
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        
        # Excluir las columnas objetivo
        target_cols = ['label', 'attack_cat']
        feature_cols = [col for col in categorical_cols if col not in target_cols]
        
        # Aplicar Label Encoding para variables categóricas
        for col in feature_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        print(f"Variables categóricas codificadas: {len(feature_cols)}")
        return df_encoded
    
    def feature_engineering(self, df):
        """     
        Args:
            df (pd.DataFrame): Dataset procesado
            
        Returns:
            pd.DataFrame: Dataset con nuevas características
        """
        df_engineered = df.copy()
        
        # Características de ratio para análisis de tráfico
        if 'sbytes' in df.columns and 'dbytes' in df.columns:
            df_engineered['bytes_ratio'] = np.where(df_engineered['dbytes'] != 0,
                                                   df_engineered['sbytes'] / df_engineered['dbytes'], 0)
        
        if 'spkts' in df.columns and 'dpkts' in df.columns:
            df_engineered['packets_ratio'] = np.where(df_engineered['dpkts'] != 0,
                                                     df_engineered['spkts'] / df_engineered['dpkts'], 0)
        
        # Características temporales
        if 'dur' in df.columns and 'spkts' in df.columns:
            df_engineered['pkt_rate'] = np.where(df_engineered['dur'] != 0,
                                               df_engineered['spkts'] / df_engineered['dur'], 0)
        
        # Características de ventana TCP
        if 'swin' in df.columns and 'dwin' in df.columns:
            df_engineered['win_diff'] = df_engineered['swin'] - df_engineered['dwin']
        
        # Logaritmo de bytes para reducir asimetría
        for col in ['sbytes', 'dbytes']:
            if col in df.columns:
                df_engineered[f'log_{col}'] = np.log1p(df_engineered[col])
        
        print(f"Nuevas características creadas. Forma actual: {df_engineered.shape}")
        return df_engineered
    
    def feature_selection(self, X, y, k=30):
        """
        Selección de características más relevantes
        
        Args:
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            k (int): Número de características a seleccionar
            
        Returns:
            tuple: (X_selected, selected_features)
        """
        # Usar SelectKBest con f_classif para clasificación
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Obtener nombres de características seleccionadas
        selected_features = X.columns[selector.get_support()].tolist()
        
        print(f"Características seleccionadas: {len(selected_features)}")
        print("Top 10 características:")
        for i, feature in enumerate(selected_features[:10]):
            print(f"{i+1}. {feature}")
        
        return pd.DataFrame(X_selected, columns=selected_features), selected_features
    
    def scale_features(self, X_train, X_test):
        """
        Normalización de características
        
        Args:
            X_train (pd.DataFrame): Conjunto de entrenamiento
            X_test (pd.DataFrame): Conjunto de prueba
            
        Returns:
            tuple: (X_train_scaled, X_test_scaled)
        """
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return (pd.DataFrame(X_train_scaled, columns=X_train.columns),
                pd.DataFrame(X_test_scaled, columns=X_test.columns))
    
    def prepare_target_variable(self, df):
        """
        Preparación de la variable objetivo para clasificación binaria y multiclase
        
        Args:
            df (pd.DataFrame): Dataset con variable objetivo
            
        Returns:
            tuple: (y_binary, y_multiclass, class_mapping)
        """
        # Clasificación binaria: Normal vs Ataque
        if 'label' in df.columns:
            y_binary = df['label'].copy()
        else:
            # Si no existe 'label', crear basado en 'attack_cat'
            y_binary = (df['attack_cat'] != 'Normal').astype(int)
        
        # Clasificación multiclase: Tipos de ataque
        if 'attack_cat' in df.columns:
            y_multiclass = self.label_encoder.fit_transform(df['attack_cat'])
            class_mapping = dict(enumerate(self.label_encoder.classes_))
        else:
            y_multiclass = y_binary.copy()
            class_mapping = {0: 'Normal', 1: 'Attack'}
        
        print("Distribución de clases (binaria):")
        print(pd.Series(y_binary).value_counts().sort_index())
        print("\nDistribución de clases (multiclase):")
        print(pd.Series(y_multiclass).value_counts().sort_index())
        
        return y_binary, y_multiclass, class_mapping
    
    def full_preprocessing_pipeline(self, test_size=0.2, random_state=42, k_features=30):
        """
        Pipeline completo de preprocesamiento
        
        Args:
            test_size (float): Proporción del conjunto de prueba
            random_state (int): Semilla para reproducibilidad
            k_features (int): Número de características a seleccionar
            
        Returns:
            dict: Diccionario con todos los conjuntos de datos procesados
        """
        print("=== INICIANDO PIPELINE DE PREPROCESAMIENTO ===")
        
        # 1. Cargar datos
        df = self.load_data()
        
        # 2. Análisis exploratorio
        exploration_results = self.explore_data(df)
        
        # 3. Manejo de valores faltantes
        df_clean = self.handle_missing_values(df)
        
        # 4. Codificación de variables categóricas
        df_encoded = self.encode_categorical_features(df_clean)
        
        # 5. Ingeniería de características
        df_engineered = self.feature_engineering(df_encoded)
        
        # 6. Preparar variables objetivo
        y_binary, y_multiclass, class_mapping = self.prepare_target_variable(df_engineered)
        
        # 7. Separar características y objetivo
        feature_cols = [col for col in df_engineered.columns 
                       if col not in ['label', 'attack_cat']]
        X = df_engineered[feature_cols]
        
        # 8. Selección de características
        X_selected, selected_features = self.feature_selection(X, y_binary, k=k_features)
        
        # 9. División en conjuntos de entrenamiento y prueba
        X_train, X_test, y_bin_train, y_bin_test = train_test_split(
            X_selected, y_binary, test_size=test_size, random_state=random_state, 
            stratify=y_binary
        )
        
        _, _, y_multi_train, y_multi_test = train_test_split(
            X_selected, y_multiclass, test_size=test_size, random_state=random_state,
            stratify=y_multiclass
        )
        
        # 10. Normalización
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print("=== PREPROCESAMIENTO COMPLETADO ===")
        print(f"Forma final del conjunto de entrenamiento: {X_train_scaled.shape}")
        print(f"Forma final del conjunto de prueba: {X_test_scaled.shape}")
        
        # Retornar todos los conjuntos de datos
        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_binary_train': y_bin_train,
            'y_binary_test': y_bin_test,
            'y_multiclass_train': y_multi_train,
            'y_multiclass_test': y_multi_test,
            'selected_features': selected_features,
            'class_mapping': class_mapping,
            'exploration_results': exploration_results,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }

# Ejemplo de uso
if __name__ == "__main__":
    # Ruta al dataset UNSW-NB15
    dataset_path = "../data/raw/UNSW_NB15_training-set.csv"
    
    # Inicializar preprocesador
    preprocessor = UNSWDataPreprocessor(dataset_path)
    
    # Ejecutar pipeline completo
    processed_data = preprocessor.full_preprocessing_pipeline(
        test_size=0.2, 
        random_state=42, 
        k_features=30
    )
    
    # Guardar datos procesados
    import joblib
    joblib.dump(processed_data, '../data/processed/unsw_nb15_processed.pkl')
    print("Datos procesados guardados exitosamente.")
