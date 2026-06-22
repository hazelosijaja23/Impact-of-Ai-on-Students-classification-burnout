# Predicción del Nivel de Burnout en Estudiantes mediante Machine Learning

## Descripción del Proyecto

Este proyecto analiza el impacto del uso de herramientas de Inteligencia Artificial Generativa (GenAI) en estudiantes y estudia su relación con el nivel de **Burnout** (agotamiento académico). Se construyeron modelos de clasificación supervisada con el objetivo de predecir el nivel de riesgo de Burnout utilizando variables académicas, hábitos de estudio y factores psicológicos.

Se compararon diferentes algoritmos de Machine Learning y se realizó un análisis estadístico para evaluar si la dificultad del problema se debía al modelo utilizado o a la propia estructura de los datos.

---

## Objetivo

Desarrollar un sistema capaz de clasificar el nivel de Burnout de los estudiantes en tres categorías:

* Low
* Medium
* High

y analizar cuáles son las variables que tienen mayor influencia sobre dicha clasificación.

---

## Dataset

Se utilizó el conjunto de datos:

**AI Impact on Students**

El dataset contiene información relacionada con:

* Horas de uso semanal de IA generativa.
* Horas de estudio tradicional.
* Dependencia percibida de la IA.
* Nivel de ansiedad durante exámenes.
* GPA antes y después del semestre.
* Diversidad de herramientas utilizadas.
* Nivel de habilidad en Prompt Engineering.
* Políticas institucionales.
* Riesgo de Burnout.

---

# Flujo de Trabajo

## 1. Ingeniería de Características

Se construyeron nuevas variables con el objetivo de representar relaciones que no estaban explícitas en los datos originales.

### Ratio de Dependencia Tecnológica

Se calculó:

```
AI_Study_Ratio =
Weekly_GenAI_Hours /
(Traditional_Study_Hours + 1)
```

Esta variable permite medir cuánto depende un estudiante de herramientas de IA en comparación con las horas de estudio tradicionales.

---

### Variación del Rendimiento Académico

```
GPA_Delta =
Post_Semester_GPA - Pre_Semester_GPA
```

Permite observar si el rendimiento mejoró o empeoró después del uso de IA.

---

### Carga Combinada de Estrés

```
Stress_AI_Load =
Anxiety_Level_During_Exams × Perceived_AI_Dependency
```

Esta variable intenta capturar la interacción entre ansiedad y dependencia tecnológica.

---

## 2. Análisis Exploratorio de Datos

Se verificó la existencia de valores faltantes.

Resultado:

* No se encontraron datos faltantes.
* No fue necesario aplicar técnicas de imputación.

Posteriormente se analizaron las variables categóricas y sus distribuciones de frecuencia para comprender la composición del conjunto de datos.

---

## 3. Análisis de Correlación

Se construyó una matriz de correlación entre todas las variables numéricas.

El objetivo fue:

* Identificar relaciones entre variables.
* Detectar posibles redundancias.
* Evaluar qué variables podrían tener mayor relación con el nivel de Burnout.

---

## 4. Preprocesamiento

Se utilizó un `ColumnTransformer` para procesar automáticamente cada tipo de variable.

### Variables numéricas

Se aplicó:

* StandardScaler

para normalizar los datos y evitar que las diferencias de escala afectaran el aprendizaje.

### Variables categóricas

Se aplicó:

* OneHotEncoder

para convertir las categorías en variables binarias.

---

## 5. División del Dataset

Se realizó una partición:

* 80 % entrenamiento
* 20 % prueba

utilizando:

```
train_test_split()
```

con estratificación para conservar la proporción original de clases.

---

## 6. Modelo 1: Regresión Logística

Se entrenó una Regresión Logística optimizando hiperparámetros mediante:

* GridSearchCV
* Validación cruzada de 5 folds

Mejores parámetros:

```
C = 0.01
solver = saga
```

### Resultados

Accuracy:

```
51.27 %
```

F1-Score:

```
0.4988
```

---

## 7. Modelo 2: Random Forest

Se construyó un Random Forest y se optimizaron los hiperparámetros mediante GridSearchCV.

Mejores parámetros:

```
max_depth = 20
min_samples_leaf = 1
min_samples_split = 5
n_estimators = 300
```

### Resultados

Accuracy:

```
52.62 %
```

F1-Score:

```
0.5274
```

El modelo Random Forest presentó una ligera mejora respecto a la Regresión Logística.

---

## 8. Evaluación de los Modelos

Se analizaron:

### Reportes de clasificación

* Precisión
* Recall
* F1-score

### Matrices de confusión

Permiten visualizar los errores cometidos entre las clases:

* Low
* Medium
* High

### Curvas ROC Multiclase

Se calcularon las probabilidades para cada clase y se evaluó su capacidad discriminativa.

---

## 9. Importancia de Variables

A partir del Random Forest se calcularon las variables más relevantes.

Las características con mayor influencia fueron:

* Anxiety_Level_During_Exams
* Perceived_AI_Dependency
* AI_Study_Ratio
* Traditional_Study_Hours
* GPA_Delta

Estas variables fueron las que más contribuyeron a la predicción del Burnout.

---

## 10. Detección de Sobreajuste

Se comparó el rendimiento entre:

* Entrenamiento.
* Validación.
* Prueba.

Se observó que Random Forest obtenía resultados considerablemente superiores sobre entrenamiento, indicando cierto grado de sobreajuste.

---

## 11. Balanceo de Clases mediante SMOTE

Aunque las clases no estaban extremadamente desbalanceadas, se aplicó:

```
SMOTE
```

para generar ejemplos sintéticos y equilibrar las tres categorías.

Distribución original:

```
High    9990
Medium 16915
Low    13095
```

Distribución después de SMOTE:

```
High    16915
Medium 16915
Low     16915
```

---

## 12. Random Forest Regularizado

Se construyó un nuevo Random Forest restringiendo su complejidad para reducir el sobreajuste.

Parámetros óptimos:

```
max_depth = 12
max_features = 0.5
min_samples_leaf = 10
n_estimators = 250
```

---

## 13. Análisis Estadístico

Se realizó un ANOVA para determinar si las variables realmente diferenciaban las clases.

Variables evaluadas:

* Anxiety_Level_During_Exams
* Traditional_Study_Hours
* GPA_Delta

Todas resultaron estadísticamente significativas.

Sin embargo, los tamaños de efecto (η²) fueron pequeños:

| Variable                   | Varianza explicada |
| -------------------------- | ------------------ |
| Anxiety_Level_During_Exams | 3.01 %             |
| Traditional_Study_Hours    | 1.91 %             |
| GPA_Delta                  | 0.10 %             |

Esto indica que ninguna variable separa claramente las clases.

---

## 14. Análisis Geométrico del Problema

Se calculó el índice de Davies-Bouldin.

Resultado:

```
Davies-Bouldin = 8.6813
```

Valores tan altos indican una fuerte superposición entre las clases.

Esto significa que los estudiantes con niveles de Burnout Low, Medium y High poseen características muy similares y ocupan regiones mezcladas del espacio de características.

---

# Conclusiones

Se implementaron y compararon distintos modelos de Machine Learning para predecir el nivel de Burnout estudiantil.

Random Forest obtuvo el mejor rendimiento, alcanzando aproximadamente un 53 % de precisión.

Sin embargo, los análisis estadísticos y geométricos mostraron que el principal problema no radica en los modelos utilizados, sino en la naturaleza del conjunto de datos:

* Las clases presentan una gran superposición.
* Las variables poseen baja capacidad discriminativa.
* Incluso con ingeniería de características y balanceo mediante SMOTE, la separación entre clases sigue siendo limitada.

Por ello, mejorar significativamente el rendimiento requeriría incorporar nuevas variables que describan mejor el estado emocional y académico de los estudiantes.

---

## Tecnologías Utilizadas

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Imbalanced-Learn
* SciPy
* Google Colab

---

## Algoritmos Utilizados

* Logistic Regression
* Random Forest
* GridSearchCV
* Cross Validation
* SMOTE
* ANOVA
* Davies-Bouldin Index
* ROC Multiclase
* Feature Importance Analysis
* One-Hot Encoding
* Standard Scaling
