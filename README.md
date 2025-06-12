# Predicción de Ganadores de Partidos de Tenis 🏆🎾  

## Descripción 📖  
Este proyecto tiene como objetivo desarrollar y evaluar modelos de clasificación para predecir el ganador de partidos de tenis utilizando datos históricos de partidos oficiales de la ATP. Se exploraron diferentes algoritmos de machine learning, se realizó un análisis exploratorio de datos (EDA) y se optimizaron los modelos para mejorar su rendimiento.  

## Dataset 📊  
El dataset utilizado contiene información de partidos de tenis desde el año 2000 hasta 2025. Algunas de las variables principales incluyen:  

- **Rank_1, Rank_2**: Ranking de los jugadores.  
- **Pts_1, Pts_2**: Puntos ATP de los jugadores.  
- **Odd_1, Odd_2**: Probabilidades de apuestas para cada jugador.  
- **Winner_binary**: Variable objetivo (1 si gana el jugador 1, 0 si gana el jugador 2).  
- **Gano_el_que_tenia_mas_puntos** y **Gano_el_que_tenia_menor_odd**: Variables derivadas a partir del dataset original.  

## Modelos Utilizados 🤖  
Se probaron dos algoritmos de clasificación:  

1. **SGDClassifier** (Stochastic Gradient Descent): Modelo lineal basado en descenso de gradiente estocástico.  
2. **KNeighborsClassifier** (K-NN): Modelo basado en vecinos más cercanos.  

## Metodología 🛠️  
- **Análisis exploratorio de datos (EDA)** con histogramas, gráficos de correlación y visualizaciones.  
- **Preprocesamiento**: Escalado de variables con `StandardScaler` , Onehot encoding para tratar las variables categoricas y Feature Engineering para nuevas variables predictoras.  
- **Balanceo de clases**: Uso de **SMOTE** para mejorar el rendimiento en clases desbalanceadas.    

## Resultados 📈  
| Modelo               | Accuracy | Precisión | Recall | F1-Score |  
|----------------------|---------|-----------|--------|----------|  
| **SGDClassifier**    | 76%     | 0.81      | 0.76   | 0.75     |  
| **KNeighborsClassifier** | 98% | 0.98      | 0.98   | 0.98     |  

📌 **Ambos modelos superaron el baseline de predicción unicamente tomando en cuenta si el ganador tenia el mejor ranking (65% de accuracy).**  

## Conclusiones 🏁   
✔️ **KNeighborsClassifier** tuvo un rendimiento alto y fue el mejor modelo.  
✔️ Las variables derivadas (feature engineering) ayudaron a mejorar la precisión del modelo.
✔️ Ambos modelos mejoraron mediante Cross Validation

