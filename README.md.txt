# Análisis Comparativo: Modelos de Aprendizaje Automático en Telecomunicaciones

## Descripción
Análisis comparativo entre modelos de aprendizaje automático superficial y profundo aplicados a detección de intrusiones en redes de telecomunicaciones utilizando el dataset UNSW-NB15.

## Dataset
- **Nombre**: UNSW-NB15
- **Fuente**: Australian Centre for Cyber Security
- **Tamaño**: 2,540,044 registros
- **Características**: 49 features
- **Clases**: Normal + 9 tipos de ataques

## Modelos implementados
- **Superficial**: Random Forest con optimización de hiperparámetros
- **Profundo**: CNN-LSTM híbrido para análisis temporal-espacial

## Métricas de evaluación
- Accuracy, precision, recall, F1-Score
- ROC-AUC, Matriz de confusión
- Análisis estadístico comparativo

## Instalación
pip install -r requirements.txt