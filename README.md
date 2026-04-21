# Proyecto de Verificación de Edad para Good Seed

## Descripción del Proyecto
Este proyecto tiene como objetivo desarrollar y evaluar un modelo de aprendizaje profundo para estimar la edad de personas a partir de fotografías faciales. La aplicación principal es ayudar a la cadena de supermercados Good Seed a cumplir con las leyes de venta de alcohol, asegurando que no se venda alcohol a menores de edad.

## Problema
Good Seed busca utilizar métodos de visión artificial para verificar la edad de los clientes en las cajas al comprar alcohol, reduciendo la incidencia de ventas a menores. La tarea es construir un modelo preciso que pueda estimar la edad a partir de una imagen.

## Dataset
El dataset utilizado contiene fotografías faciales con sus respectivas edades reales. Se encuentra en la ruta `(https://www.kaggle.com/c/imagenet-object-localization-challenge/data)` y consta de:
- La carpeta `final_files` con 7600 fotos.
- El archivo `labels.csv` con dos columnas: `file_name` y `real_age`.

**Nota:** Los archivos de imágenes no se incluyen en este repositorio debido a su tamaño. Se espera que el dataset esté disponible en la ruta especificada en el entorno de ejecución.

## Análisis Exploratorio de Datos (EDA)

El análisis exploratorio reveló las siguientes características del dataset:
- **Tamaño del dataset:** 7591 imágenes.
- **Rango de edades:** Las edades varían desde 1 hasta 100 años.
- **Distribución de edades:** La distribución no es uniforme, mostrando una concentración de personas entre los 20 y 40 años (edad promedio: 31.2 años, mediana: 29.0 años).
- **Picos de edad:** Se observaron picos en edades redondas (ej. 25, 30, 40 años), lo que sugiere posibles sesgos en la recolección o etiquetado de datos.
- Las frecuencias en los extremos (niños muy pequeños y adultos mayores de 70 años) son significativamente menores.

## Arquitectura del Modelo
El modelo utiliza una arquitectura basada en `ResNet50` como *backbone* para la extracción de características, pre-entrenado con el dataset `imagenet`. Sobre este, se añade una capa de `GlobalAveragePooling2D` y una capa `Dense` de salida con activación `relu` para la regresión de edad.

- **Base Model:** ResNet50 (sin la capa `include_top`)
- **Capas añadidas:** GlobalAveragePooling2D, Dense (1 neurona, activación 'relu')
- **Optimizador:** Adam (learning rate = 0.0001)
- **Función de pérdida:** Mean Squared Error (MSE)
- **Métrica:** Mean Absolute Error (MAE)

## Entrenamiento
El modelo fue entrenado durante 20 épocas utilizando `ImageDataGenerator` para cargar las imágenes y sus etiquetas, con un tamaño de lote de 32 y un escalado de píxeles a 1/255.

## Resultados
Tras el entrenamiento, el modelo alcanzó un **Error Absoluto Medio (MAE) de 3.239 años** en el conjunto de prueba.

## Conclusiones e Implicaciones
El modelo desarrollado demuestra un rendimiento prometedor en la estimación de edad con un MAE de 3.239 años. Esto significa que, en promedio, las predicciones del modelo se desvían en aproximadamente 3.2 años de la edad real.

**Implicaciones para Good Seed:**
*   **Herramienta de Apoyo:** El modelo puede ser una herramienta eficaz para el personal de caja, ayudando a verificar la edad, especialmente cuando no es obvia.
*   **Cumplimiento Normativo:** Un MAE bajo sugiere que el sistema puede contribuir significativamente a reducir las ventas accidentales de alcohol a menores, apoyando el cumplimiento de las leyes.
*   **Eficiencia Operativa:** Podría agilizar el proceso de verificación de edad.

**Consideraciones para el futuro:**
*   El sesgo demográfico del dataset podría afectar el rendimiento en grupos de edad subrepresentados. Sería beneficioso analizar el rendimiento en casos extremos (menores de 18 y mayores de 70).
*   La robustez del modelo en condiciones reales de iluminación, ángulos y accesorios (gafas, sombreros) necesitaría ser evaluada y, si fuera necesario, mejorada con datos más variados.
*   Para una aplicación práctica, se debe definir un umbral de decisión claro para la verificación manual y evaluar el equilibrio entre falsos positivos y falsos negativos.

## Cómo Ejecutar el Modelo

1.  **Requisitos:**
    - Python 3.x
    - TensorFlow (con soporte GPU recomendado)
    - Pandas
    - Keras (parte de TensorFlow)
    - `tensorflow.keras.preprocessing.image.ImageDataGenerator`
    
    Puedes instalar las dependencias con un `requirements.txt` (no incluido, pero puedes generarlo con `pip freeze > requirements.txt` si lo ejecutas en tu entorno):
    ```bash
    pip install tensorflow pandas
    ```

2.  **Preparar el Dataset:**
    Asegúrate de que el dataset (`labels.csv` y la carpeta `final_files`) esté disponible en la ruta `/content/` o la ruta que se configure en el script.

3.  **Ejecutar el Script de Entrenamiento:**
    El archivo `run_model_on_gpu.py` contiene todo el código necesario para cargar los datos, crear y entrenar el modelo, y evaluarlo.
    ```bash
    python run_model_on_gpu.py
    ```
    Este script generará la salida del entrenamiento y la evaluación final del MAE en la consola.

-----

## 👩‍💻 Autora

*Jenifer Gonzalez*

Data Science | QA Engineer | Scrum Master 

[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?style=flat&logo=linkedin)](www.linkedin.com/in/jenifer-paola-gonzalez-peñuela)
[![GitHub](https://img.shields.io/badge/GitHub-black?style=flat&logo=github)](https://github.com/Jenifer-Gonzalez-DS-QA/jenifergon91))

-----
    
