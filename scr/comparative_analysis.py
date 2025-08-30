import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, ttest_rel
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score, accuracy_score, precision_score, recall_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class StatisticalComparator:
    def __init__(self, alpha=0.05):
        """
        Inicializar comparador estadístico
        
        Args:
            alpha (float): Nivel de significancia
        """
        self.alpha = alpha
        self.results = {}
        self.statistical_tests = {}
        
    def perform_cross_validation_comparison(self, models_dict, X, y, cv_folds=10):
        """
        Realizar validación cruzada para comparación estadística robusta
        
        Args:
            models_dict (dict): Diccionario de modelos {nombre: modelo}
            X (pd.DataFrame): Características
            y (pd.Series): Variable objetivo
            cv_folds (int): Número de folds para CV
            
        Returns:
            pd.DataFrame: Resultados de validación cruzada
        """
        print("=== REALIZANDO VALIDACIÓN CRUZADA PARA COMPARACIÓN ===")
        
        # Definir métricas de evaluación
        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'f1_macro': make_scorer(f1_score, average='macro'),
            'f1_weighted': make_scorer(f1_score, average='weighted'),
            'precision_macro': make_scorer(precision_score, average='macro'),
            'recall_macro': make_scorer(recall_score, average='macro')
        }
        
        # Configurar validación cruzada estratificada
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        cv_results = {}
        
        for model_name, model in models_dict.items():
            print(f"Evaluando {model_name}...")
            model_results = {}
            
            for metric_name, scorer in scoring_metrics.items():
                scores = cross_val_score(
                    model, X, y, cv=cv_strategy, scoring=scorer, n_jobs=-1
                )
                model_results[metric_name] = {
                    'scores': scores,
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'median': np.median(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
                
                print(f"  {metric_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
            
            cv_results[model_name] = model_results
        
        self.cv_results = cv_results
        return cv_results
    
    def create_comparison_dataframe(self):
        """
        Crear DataFrame para análisis estadístico
        
        Returns:
            pd.DataFrame: Datos organizados para tests estadísticos
        """
        if not hasattr(self, 'cv_results'):
            raise ValueError("Debe ejecutar perform_cross_validation_comparison primero")
        
        comparison_data = {}
        
        for model_name, results in self.cv_results.items():
            for metric_name, metric_data in results.items():
                col_name = f"{model_name}_{metric_name}"
                comparison_data[col_name] = metric_data['scores']
        
        df_comparison = pd.DataFrame(comparison_data)
        return df_comparison
    
    def friedman_test(self, metric='f1_macro'):
        """
        Realizar Test de Friedman para comparación múltiple no paramétrica
        
        Args:
            metric (str): Métrica a analizar
            
        Returns:
            dict: Resultados del test de Friedman
        """
        print(f"=== TEST DE FRIEDMAN PARA {metric.upper()} ===")
        
        # Extraer datos para la métrica específica
        metric_data = []
        model_names = []
        
        for model_name, results in self.cv_results.items():
            if metric in results:
                metric_data.append(results[metric]['scores'])
                model_names.append(model_name)
        
        if len(metric_data) < 3:
            print("Se requieren al menos 3 modelos para el test de Friedman")
            return None
        
        # Realizar test de Friedman
        statistic, p_value = friedmanchisquare(*metric_data)
        
        # Calcular rankings promedio
        combined_data = np.array(metric_data).T  # Transponer para tener filas=folds, cols=modelos
        rankings = np.argsort(np.argsort(-combined_data, axis=1), axis=1) + 1  # Rankings (1=mejor)
        average_rankings = np.mean(rankings, axis=0)
        
        results = {
            'metric': metric,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': self.alpha,
            'significant': p_value < self.alpha,
            'model_names': model_names,
            'average_rankings': average_rankings,
            'interpretation': self._interpret_friedman_results(p_value, model_names, average_rankings)
        }
        
        print(f"Estadístico de Friedman: {statistic:.4f}")
        print(f"p-valor: {p_value:.6f}")
        print(f"¿Diferencia significativa? {'Sí' if results['significant'] else 'No'} (α = {self.alpha})")
        print("\nRankings promedio (1 = mejor):")
        for model, rank in zip(model_names, average_rankings):
            print(f"  {model}: {rank:.2f}")
        
        self.statistical_tests[f'friedman_{metric}'] = results
        return results
    
    def wilcoxon_pairwise_tests(self, metric='f1_macro', correction='bonferroni'):
        """
        Realizar tests de Wilcoxon pareados con corrección por comparaciones múltiples
        
        Args:
            metric (str): Métrica a analizar
            correction (str): Método de corrección ('bonferroni', 'holm', 'none')
            
        Returns:
            dict: Resultados de comparaciones pareadas
        """
        print(f"=== TESTS DE WILCOXON PAREADOS PARA {metric.upper()} ===")
        
        # Extraer datos
        models_data = {}
        for model_name, results in self.cv_results.items():
            if metric in results:
                models_data[model_name] = results[metric]['scores']
        
        model_names = list(models_data.keys())
        n_models = len(model_names)
        
        if n_models < 2:
            print("Se requieren al menos 2 modelos para comparaciones pareadas")
            return None
        
        # Realizar todas las comparaciones pareadas
        pairwise_results = []
        p_values = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                model1, model2 = model_names[i], model_names[j]
                data1, data2 = models_data[model1], models_data[model2]
                
                # Test de Wilcoxon
                statistic, p_value = wilcoxon(data1, data2, alternative='two-sided')
                
                # Tamaño del efecto (Cohen's d)
                cohens_d = self._calculate_cohens_d(data1, data2)
                
                # Diferencia de medias
                mean_diff = np.mean(data1) - np.mean(data2)
                
                pairwise_results.append({
                    'model1': model1,
                    'model2': model2,
                    'comparison': f"{model1} vs {model2}",
                    'statistic': statistic,
                    'p_value': p_value,
                    'mean_diff': mean_diff,
                    'cohens_d': cohens_d,
                    'effect_size': self._interpret_effect_size(cohens_d)
                })
                p_values.append(p_value)
        
        # Aplicar corrección por comparaciones múltiples
        if correction == 'bonferroni':
            corrected_alpha = self.alpha / len(p_values)
            correction_factor = len(p_values)
        elif correction == 'holm':
            # Método de Holm-Bonferroni (más potente que Bonferroni)
            sorted_indices = np.argsort(p_values)
            corrected_alpha = self.alpha / (len(p_values) - np.arange(len(p_values)))
            correction_factor = None
        else:  # 'none'
            corrected_alpha = self.alpha
            correction_factor = 1
        
        # Determinar significancia con corrección
        for i, result in enumerate(pairwise_results):
            if correction == 'holm':
                # Para Holm, necesitamos ordenar los p-valores
                sorted_idx = np.where(sorted_indices == i)[0][0]
                result['corrected_alpha'] = corrected_alpha[sorted_idx]
                result['significant'] = result['p_value'] < corrected_alpha[sorted_idx]
            else:
                result['corrected_alpha'] = corrected_alpha
                result['significant'] = result['p_value'] < corrected_alpha
        
        # Crear DataFrame de resultados
        df_results = pd.DataFrame(pairwise_results)
        
        # Mostrar resultados
        print(f"Corrección aplicada: {correction}")
        if correction_factor:
            print(f"Alpha corregido: {corrected_alpha:.6f} (factor: {correction_factor})")
        
        print("\nResultados de comparaciones pareadas:")
        for result in pairwise_results:
            significance = "***" if result['significant'] else "n.s."
            print(f"{result['comparison']:25} | p={result['p_value']:.6f} {significance} | "
                  f"Δ={result['mean_diff']:+.4f} | d={result['cohens_d']:+.3f} ({result['effect_size']})")
        
        results_dict = {
            'metric': metric,
            'correction_method': correction,
            'pairwise_results': pairwise_results,
            'results_dataframe': df_results,
            'significant_comparisons': [r for r in pairwise_results if r['significant']]
        }
        
        self.statistical_tests[f'wilcoxon_{metric}'] = results_dict
        return results_dict
    
    def bootstrap_confidence_intervals(self, metric='f1_macro', n_bootstrap=1000, confidence=0.95):
        """
        Calcular intervalos de confianza bootstrap para métricas
        
        Args:
            metric (str): Métrica a analizar
            n_bootstrap (int): Número de muestras bootstrap
            confidence (float): Nivel de confianza
            
        Returns:
            dict: Intervalos de confianza bootstrap
        """
        print(f"=== INTERVALOS DE CONFIANZA BOOTSTRAP PARA {metric.upper()} ===")
        
        bootstrap_results = {}
        alpha_ci = 1 - confidence
        
        for model_name, results in self.cv_results.items():
            if metric in results:
                scores = results[metric]['scores']
                
                # Generar muestras bootstrap
                bootstrap_means = []
                np.random.seed(42)  # Para reproducibilidad
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
                    bootstrap_means.append(np.mean(bootstrap_sample))
                
                bootstrap_means = np.array(bootstrap_means)
                
                # Calcular intervalos de confianza
                lower_percentile = (alpha_ci / 2) * 100
                upper_percentile = (1 - alpha_ci / 2) * 100
                
                ci_lower = np.percentile(bootstrap_means, lower_percentile)
                ci_upper = np.percentile(bootstrap_means, upper_percentile)
                
                bootstrap_results[model_name] = {
                    'original_mean': np.mean(scores),
                    'bootstrap_mean': np.mean(bootstrap_means),
                    'bootstrap_std': np.std(bootstrap_means),
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower,
                    'confidence_level': confidence
                }
                
                print(f"{model_name:20} | Media: {np.mean(scores):.4f} | "
                      f"IC {confidence*100:.0f}%: [{ci_lower:.4f}, {ci_upper:.4f}] | "
                      f"Ancho: {ci_upper - ci_lower:.4f}")
        
        self.statistical_tests[f'bootstrap_{metric}'] = bootstrap_results
        return bootstrap_results
    
    def _calculate_cohens_d(self, data1, data2):
        """Calcular Cohen's d para tamaño del efecto"""
        n1, n2 = len(data1), len(data2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(data1, ddof=1) + 
                             (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0
        
        return (np.mean(data1) - np.mean(data2)) / pooled_std
    
    def _interpret_effect_size(self, cohens_d):
        """Interpretar el tamaño del efecto según Cohen's d"""
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "trivial"
        elif abs_d < 0.5:
            return "pequeño"
        elif abs_d < 0.8:
            return "mediano"
        else:
            return "grande"
    
    def _interpret_friedman_results(self, p_value, model_names, rankings):
        """Interpretar resultados del test de Friedman"""
        if p_value >= self.alpha:
            return "No hay evidencia de diferencias significativas entre los modelos."
        
        best_model_idx = np.argmin(rankings)
        best_model = model_names[best_model_idx]
        
        return f"Diferencias significativas detectadas. Modelo con mejor ranking: {best_model}"
    
    def create_comparison_report(self, metrics=['accuracy', 'f1_macro', 'f1_weighted']):
        """
        Generar reporte completo de comparación estadística
        
        Args:
            metrics (list): Métricas a incluir en el reporte
            
        Returns:
            dict: Reporte completo
        """
        print("=== GENERANDO REPORTE COMPARATIVO COMPLETO ===")
        
        report = {
            'summary': {},
            'statistical_tests': {},
            'recommendations': []
        }
        
        # Resumen de métricas por modelo
        summary_data = []
        for model_name, results in self.cv_results.items():
            model_summary = {'model': model_name}
            for metric in metrics:
                if metric in results:
                    model_summary[f'{metric}_mean'] = results[metric]['mean']
                    model_summary[f'{metric}_std'] = results[metric]['std']
            summary_data.append(model_summary)
        
        report['summary']['metrics_by_model'] = pd.DataFrame(summary_data)
        
        # Realizar tests estadísticos para cada métrica
        for metric in metrics:
            print(f"\n--- Análisis para {metric} ---")
            
            # Test de Friedman
            friedman_results = self.friedman_test(metric)
            if friedman_results:
                report['statistical_tests'][f'friedman_{metric}'] = friedman_results
                
                # Si hay diferencias significativas, hacer comparaciones pareadas
                if friedman_results['significant']:
                    wilcoxon_results = self.wilcoxon_pairwise_tests(metric, correction='holm')
                    report['statistical_tests'][f'wilcoxon_{metric}'] = wilcoxon_results
            
            # Intervalos de confianza bootstrap
            bootstrap_results = self.bootstrap_confidence_intervals(metric)
            report['statistical_tests'][f'bootstrap_{metric}'] = bootstrap_results
        
        # Generar recomendaciones basadas en análisis
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        return report
    
    def _generate_recommendations(self, report):
        """Generar recomendaciones basadas en análisis estadístico"""
        recommendations = []
        
        # Analizar métricas principales
        primary_metric = 'f1_macro'
        
        if f'friedman_{primary_metric}' in report['statistical_tests']:
            friedman_result = report['statistical_tests'][f'friedman_{primary_metric}']
            
            if friedman_result['significant']:
                best_model_idx = np.argmin(friedman_result['average_rankings'])
                best_model = friedman_result['model_names'][best_model_idx]
                
                recommendations.append({
                    'type': 'model_selection',
                    'recommendation': f"Modelo recomendado: {best_model}",
                    'justification': f"Mejor ranking promedio en test de Friedman para {primary_metric}",
                    'confidence': 'alta' if friedman_result['p_value'] < 0.01 else 'media'
                })
                
                # Verificar si hay diferencias significativas en comparaciones pareadas
                if f'wilcoxon_{primary_metric}' in report['statistical_tests']:
                    wilcoxon_result = report['statistical_tests'][f'wilcoxon_{primary_metric}']
                    significant_comparisons = wilcoxon_result['significant_comparisons']
                    
                    if significant_comparisons:
                        for comp in significant_comparisons:
                            if comp['mean_diff'] > 0:  # model1 > model2
                                recommendations.append({
                                    'type': 'pairwise_comparison',
                                    'recommendation': f"{comp['model1']} supera significativamente a {comp['model2']}",
                                    'justification': f"Diferencia de medias: {comp['mean_diff']:+.4f}, tamaño del efecto: {comp['effect_size']}",
                                    'p_value': comp['p_value']
                                })
            else:
                recommendations.append({
                    'type': 'no_difference',
                    'recommendation': "No se detectaron diferencias estadísticamente significativas entre los modelos",
                    'justification': f"Test de Friedman p-valor: {friedman_result['p_value']:.4f} > α = {self.alpha}",
                    'implication': "La selección puede basarse en otros criterios (complejidad, interpretabilidad, tiempo de ejecución)"
                })
        
        return recommendations
    
    def plot_comparison_visualizations(self, save_dir=None):
        """
        Crear visualizaciones comprehensivas de la comparación
        
        Args:
            save_dir (str): Directorio para guardar figuras
        """
        print("=== GENERANDO VISUALIZACIONES COMPARATIVAS ===")
        
        # 1. Boxplot de métricas por modelo
        self._plot_metrics_boxplot(save_dir)
        
        # 2. Heatmap de rankings
        self._plot_rankings_heatmap(save_dir)
        
        # 3. Intervalos de confianza
        self._plot_confidence_intervals(save_dir)
        
        # 4. Matriz de comparaciones pareadas
        self._plot_pairwise_matrix(save_dir)
    
    def _plot_metrics_boxplot(self, save_dir=None):
        """Boxplot de distribución de métricas"""
        metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        available_metrics = []
        
        # Verificar qué métricas están disponibles
        for metric in metrics:
            if any(metric in results for results in self.cv_results.values()):
                available_metrics.append(metric)
        
        if not available_metrics:
            print("No hay métricas disponibles para visualización")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics[:6]):  # Máximo 6 métricas
            ax = axes[i]
            
            # Preparar datos para boxplot
            data_for_plot = []
            labels = []
            
            for model_name, results in self.cv_results.items():
                if metric in results:
                    data_for_plot.append(results[metric]['scores'])
                    labels.append(model_name)
            
            if data_for_plot:
                bp = ax.boxplot(data_for_plot, labels=labels, patch_artist=True)
                
                # Colorear cajas
                colors = plt.cm.Set3(np.linspace(0, 1, len(data_for_plot)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # Rotar etiquetas si son muy largas
                if any(len(label) > 10 for label in labels):
                    ax.tick_params(axis='x', rotation=45)
        
        # Remover subplots vacíos
        for j in range(len(available_metrics), len(axes)):
            fig.delaxes(axes[j])
        
        plt.suptitle('Distribución de Métricas por Modelo (Validación Cruzada)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/metrics_boxplot_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_confidence_intervals(self, save_dir=None):
        """Plot de intervalos de confianza bootstrap"""
        metrics = ['f1_macro', 'accuracy', 'f1_weighted']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if f'bootstrap_{metric}' in self.statistical_tests:
                ax = axes[i]
                bootstrap_results = self.statistical_tests[f'bootstrap_{metric}']
                
                models = list(bootstrap_results.keys())
                means = [bootstrap_results[model]['original_mean'] for model in models]
                ci_lowers = [bootstrap_results[model]['ci_lower'] for model in models]
                ci_uppers = [bootstrap_results[model]['ci_upper'] for model in models]
                
                # Calcular errores para errorbar
                lower_errors = [means[j] - ci_lowers[j] for j in range(len(means))]
                upper_errors = [ci_uppers[j] - means[j] for j in range(len(means))]
                
                y_pos = np.arange(len(models))
                
                ax.errorbar(means, y_pos, xerr=[lower_errors, upper_errors], 
                           fmt='o', capsize=5, capthick=2, markersize=8)
                
                ax.set_yticks(y_pos)
                ax.set_yticklabels(models)
                ax.set_xlabel(f'{metric.replace("_", " ").title()} Score')
                ax.set_title(f'Intervalos de Confianza 95%\n{metric.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Intervalos de Confianza Bootstrap para Métricas Principales', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/confidence_intervals.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_rankings_heatmap(self, save_dir=None):
        """Heatmap de rankings de modelos por métrica"""
        metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        models = list(self.cv_results.keys())
        
        # Crear matriz de rankings
        rankings_matrix = []
        available_metrics = []
        
        for metric in metrics:
            if all(metric in results for results in self.cv_results.values()):
                available_metrics.append(metric)
                metric_scores = [self.cv_results[model][metric]['mean'] for model in models]
                rankings = np.argsort(np.argsort(-np.array(metric_scores))) + 1
                rankings_matrix.append(rankings)
        
        if not rankings_matrix:
            print("No hay suficientes métricas comunes para heatmap de rankings")
            return
        
        rankings_df = pd.DataFrame(rankings_matrix, 
                                 index=[m.replace('_', ' ').title() for m in available_metrics],
                                 columns=models)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(rankings_df, annot=True, cmap='RdYlGn_r', fmt='d',
                   cbar_kws={'label': 'Ranking (1 = mejor)'})
        plt.title('Ranking de Modelos por Métrica', fontsize=14, fontweight='bold')
        plt.ylabel('Métricas')
        plt.xlabel('Modelos')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/rankings_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_pairwise_matrix(self, save_dir=None):
        """Matriz de comparaciones pareadas"""
        metric = 'f1_macro'  # Usar métrica principal
        
        if f'wilcoxon_{metric}' not in self.statistical_tests:
            print(f"No hay resultados de comparaciones pareadas para {metric}")
            return
        
        wilcoxon_results = self.statistical_tests[f'wilcoxon_{metric}']
        pairwise_results = wilcoxon_results['pairwise_results']
        
        # Extraer nombres de modelos únicos
        models = list(set([r['model1'] for r in pairwise_results] + 
                         [r['model2'] for r in pairwise_results]))
        models.sort()
        
        n_models = len(models)
        p_matrix = np.ones((n_models, n_models))
        effect_matrix = np.zeros((n_models, n_models))
        
        # Llenar matriz
        for result in pairwise_results:
            i = models.index(result['model1'])
            j = models.index(result['model2'])
            
            p_matrix[i, j] = result['p_value']
            p_matrix[j, i] = result['p_value']
            
            effect_matrix[i, j] = result['cohens_d']
            effect_matrix[j, i] = -result['cohens_d']
        
        # Crear subplot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Matriz de p-valores
        mask_p = np.eye(n_models, dtype=bool)
        sns.heatmap(p_matrix, mask=mask_p, annot=True, fmt='.4f', 
                   xticklabels=models, yticklabels=models,
                   cmap='RdYlBu_r', cbar_kws={'label': 'p-valor'}, ax=axes[0])
        axes[0].set_title('Matriz de p-valores\n(Comparaciones Pareadas)')
        
        # Matriz de tamaños de efecto
        sns.heatmap(effect_matrix, mask=mask_p, annot=True, fmt='.3f',
                   xticklabels=models, yticklabels=models,
                   cmap='RdBu_r', center=0, cbar_kws={'label': "Cohen's d"}, ax=axes[1])
        axes[1].set_title('Matriz de Tamaños de Efecto\n(Cohen\'s d)')
        
        plt.suptitle(f'Análisis de Comparaciones Pareadas - {metric.replace("_", " ").title()}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(f'{save_dir}/pairwise_comparison_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def export_results(self, save_path):
        """
        Exportar todos los resultados a archivo
        
        Args:
            save_path (str): Ruta para guardar resultados
        """
        results_export = {
            'cv_results': self.cv_results,
            'statistical_tests': self.statistical_tests,
            'alpha': self.alpha,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        joblib.dump(results_export, save_path)
        print(f"Resultados exportados a: {save_path}")

# Pipeline principal para análisis comparativo completo
def perform_complete_comparative_analysis(shallow_results, deep_results, processed_data, 
                                        save_dir='results/comparative_analysis/'):
    """
    Pipeline completo para análisis estadístico comparativo
    
    Args:
        shallow_results (dict): Resultados de modelos superficiales
        deep_results (dict): Resultados de modelos profundos  
        processed_data (dict): Datos preprocesados
        save_dir (str): Directorio para guardar resultados
        
    Returns:
        dict: Reporte completo del análisis comparativo
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    print("=== INICIANDO ANÁLISIS COMPARATIVO COMPLETO ===")
    
    # Combinar modelos para comparación
    # Nota: Para el análisis estadístico necesitamos recrear los modelos
    # ya que necesitamos hacer validación cruzada
    
    # Cargar modelos guardados (esto sería el proceso real)
    # Por simplicidad, aquí mostramos la estructura del análisis
    
    # Preparar datos para análisis
    X = processed_data['X_train']
    y = processed_data['y_binary_train']
    
    # Inicializar comparador estadístico
    comparator = StatisticalComparator(alpha=0.05)
    
    # Modelos para hacer CV
    # Por ejemplo:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    
    models_for_comparison = {
        'Random_Forest': RandomForestClassifier(n_estimators=200, max_depth=30, 
                                              class_weight='balanced', random_state=42),
        'Extra_Trees': ExtraTreesClassifier(n_estimators=200, max_depth=30,
                                          class_weight='balanced', random_state=42)
    }
    
    # Para modelos profundos, sería más complejo debido a la naturaleza de los datos
    # Se podría usar KerasClassifier de sklearn para integrar modelos de TensorFlow
    
    # Realizar validación cruzada comparativa
    cv_results = comparator.perform_cross_validation_comparison(
        models_for_comparison, X, y, cv_folds=10
    )
    
    # Generar reporte completo
    report = comparator.create_comparison_report(
        metrics=['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    )
    
    # Crear visualizaciones
    comparator.plot_comparison_visualizations(save_dir)
    
    # Exportar resultados
    comparator.export_results(f'{save_dir}/statistical_comparison_results.pkl')
    
    # Generar reporte final en texto
    _generate_text_report(report, comparator, f'{save_dir}/comparative_analysis_report.txt')
    
    print(f"\nAnálisis comparativo completado. Resultados guardados en: {save_dir}")
    
    return report, comparator

def _generate_text_report(report, comparator, save_path):
    """Generar reporte en formato texto"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE ANÁLISIS COMPARATIVO\n")
        f.write("Modelos de Aprendizaje Automático en Telecomunicaciones\n")
        f.write("="*60 + "\n\n")
        
        f.write("RESUMEN EJECUTIVO\n")
        f.write("-"*20 + "\n")
        
        # Escribir recomendaciones
        for rec in report['recommendations']:
            f.write(f"• {rec['recommendation']}\n")
            f.write(f"  Justificación: {rec['justification']}\n\n")
        
        f.write("\nRESULTADOS ESTADÍSTICOS DETALLADOS\n")
        f.write("-"*35 + "\n")
        
        # Detalles de tests estadísticos
        for test_name, results in report['statistical_tests'].items():
            f.write(f"\n{test_name.upper()}:\n")
            if 'p_value' in results:
                f.write(f"  p-valor: {results['p_value']:.6f}\n")
                f.write(f"  Significativo: {'Sí' if results.get('significant', False) else 'No'}\n")
    
    print(f"Reporte de texto generado: {save_path}")

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar resultados previos
    processed_data = joblib.load('data/processed/unsw_nb15_processed.pkl')
    shallow_results = joblib.load('models/shallow/shallow_evaluation_results.pkl')
    deep_results = joblib.load('models/deep/deep_evaluation_results.pkl')
    
    # Realizar análisis comparativo completo
    report, comparator = perform_complete_comparative_analysis(
        shallow_results, deep_results, processed_data
    )
    
    print("Análisis estadístico comparativo completado.")
