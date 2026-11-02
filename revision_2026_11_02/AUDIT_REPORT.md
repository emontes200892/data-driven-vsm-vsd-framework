# Auditoría técnica y de reproducibilidad

Manuscrito: *A Closed-Loop Data-Driven Framework for Value Stream Mapping and Future-State Design*.

## Resultado principal

Los cinco porcentajes operativos del manuscrito se reprodujeron ejecutando el script 04 original sobre los parámetros entregados, con 30 réplicas y 500 órdenes por escenario. Su reproducción no valida la atribución causal de las mejoras al framework: los escenarios ya reciben multiplicadores progresivamente favorables.

El F1 sintético de retrabajo también se reprodujo. La auditoría encontró una explicación verificable: `rework_flag` coincide exactamente con `activity == "Rework"` en las 8,532 filas. Una regla que aprende la etiqueta mayoritaria de cada actividad solo en entrenamiento logra F1 = 1.0000 en el test, al igual que una red construida exclusivamente en entrenamiento. La auditoría se amplió después a los tres CSV preparados. BPI2017 también reproduce F1 = 1.0000 con la red de entrenamiento; una regla por actividad alcanza el mismo resultado porque `rework_flag` identifica exactamente `O_Returned`.

## Evidencia ejecutada

| Comprobación | Resultado |
|---|---|
| Dataset entregado | 8,532 eventos, 1,200 casos, 8 actividades |
| Retrabajo positivo en test | 21 de 1,280 eventos |
| `rework_flag == (activity == "Rework")` | Coincide en todos los eventos |
| `delay_flag == (waiting_time_min > 15)` | Coincide en todos los eventos |
| Regla mayoritaria por actividad, aprendida solo en entrenamiento | F1 = 1.0000 |
| Random Forest con actividad codificada | F1 = 1.0000 |
| Random Forest con centralidades de entrenamiento únicamente | F1 = 1.0000 |
| Random Forest con variables operativas únicamente | F1 = 0.2500 |
| Random Forest con variables operativas y centralidades de entrenamiento | F1 = 1.0000 |
| Matriz de confusión de las condiciones perfectas | TN = 1,259; FP = 0; FN = 0; TP = 21 |
| Centralidades de entrenamiento distintas al usar el grafo completo, en este dataset | 0 de 8 actividades para cada centralidad |
| Firmas de centralidad que identifican una sola actividad | 8 de 8 |
| Casos compartidos entrenamiento–validación | 2 |
| Casos compartidos validación–test | 3 |
| Casos compartidos entrenamiento–test | 0 |
| Eventos anteriores al test cuyo `end_ts` alcanza o supera el inicio del test | 2 |

Los conteos temporales corresponden al ordenamiento por inicio de evento y a la partición 70/15/15. No demuestran contaminación por sí solos, pero muestran que no es una partición de casos completos y que la disponibilidad de resultados requiere control de tiempos de finalización.

## Reproducción de los porcentajes

| Indicador del script | Cambio reproducido (%) | Redondeo del artículo (%) | IC 95% nuevo por bootstrap de bloques (%) |
|---|---:|---:|---:|
| Lead time | 32.398868 de reducción | 32.40 | 32.257447 a 32.541441 |
| WIP del modelo | 45.093769 de reducción | 45.09 | 44.655717 a 45.530380 |
| Defect rate | 27.628697 de reducción | 27.63 | 25.451496 a 29.826476 |
| Rework rate | 45.524476 de reducción | 45.52 | 40.719820 a 50.000000 |
| Throughput del modelo | 71.727047 de aumento | 71.73 | 71.717569 a 71.736743 |

Los intervalos nuevos se calcularon con 10,000 remuestreos de las 30 réplicas por bloques, semilla 20261024. Se remuestrean pares de escenarios para calcular una razón de medias. Caracterizan variabilidad condicional al simulador; no incluyen incertidumbre de calibración de una planta ni del diseño de los factores.

Las comparaciones G5–G0 con Wilcoxon bilateral y Holm sobre las cinco comparaciones contra el baseline dentro de cada indicador arrojan p ajustadas de 9.313226e-09 para lead time, throughput y WIP; 8.628429e-06 para defect rate; y 8.644742e-06 para rework rate. No se aplicó una corrección global entre familias de indicadores. Los archivos CSV contienen las pruebas globales de Friedman, todas las comparaciones y sus intervalos.

## 1. Construcción de las redes

En `09_real_dataset_ml.py`, la partición temporal ocurre antes de `build_training_network_features(train_raw)`. Las centralidades de entrenamiento se unen a validación y test por actividad, y una actividad desconocida recibe ceros. Este control está implementado; no corresponde afirmar que el script 09 usa necesariamente el grafo completo para entrenar.

En cambio, `02_network_features.py` calcula el grafo del dataset sintético completo y `03_ml_experiments.py` lee ese archivo antes de separar las filas. Esto es una violación del protocolo de características exclusivamente de entrenamiento. Sin embargo, en el dataset sintético recibido las cuatro centralidades resultan numéricamente iguales al reconstruirlas solo con entrenamiento. Por eso no es correcto atribuir el F1 perfecto sintético a una diferencia observada entre esas dos redes. La explicación comprobada es que las centralidades funcionan como firmas de actividad y la etiqueta es determinista por actividad.

## 2. Objetivos y variables disponibles

El generador define el retraso a partir de la espera actual (`waiting_time > 15`). El script sintético 03 incluye esa misma espera en sus predictores. Su F1 cercano a uno reconstruye una regla disponible entre las entradas; no prueba anticipación de retrasos.

En 09, la exclusión de retraso es `waiting_time_min`, no `cycle_time_min`. El bottleneck proxy se obtiene del percentil 90 de medias por actividad de ciclo o espera y excluye ambos tiempos de sus predictores. No es una etiqueta definida por betweenness. La versión editorial anterior describía esas exclusiones de manera incorrecta y debe corregirse según el código.

La etiqueta real de retrabajo se lee de `rework_flag`; 09 no la crea a partir del futuro. Para conocer qué representa realmente hay que revisar la preparación específica de cada dataset.

El script 07 proporcionado crea `cycle_time_min` como la diferencia con el siguiente timestamp y `end_ts` como ese siguiente timestamp. Esa variable utiliza un evento posterior al inicio actual; no está disponible para un pronóstico en el inicio del evento. La diferencia entre eventos consecutivos no identifica, sin información de lifecycle, un tiempo de servicio de máquina. La espera se crea como la diferencia desde el evento anterior, por lo que las dos series pueden duplicar la interpretación de intervalos entre eventos. Además, cola, WIP y defectos se rellenan con cero; eso representa ausencia de medición en la transformación, no evidencia de ausencia física de cola, inventario o defectos.

07 también recorta con `head(250000)` antes de ordenar y puede dejar trazas incompletas. Su fallback de un solo caso si falta el identificador no debe utilizarse para afirmar validez de procesos multicaso. Los resultados sobre los CSV preparados y la sensibilidad al lector están documentados en la sección 5.

## 3. Simulación y alcance de la atribución

El script 04 usa NumPy para muestrear órdenes independientes. No ejecuta SimPy, un calendario de eventos, recursos compartidos, servidores ni una cola con competencia entre órdenes. Es un análisis Monte Carlo de escenarios con factores de tiempo, espera y calidad fijados antes del cálculo.

| Escenario original | Factor ciclo | Factor espera | Factor defectos | Ganancia multiplicativa adicional |
|---|---:|---:|---:|---:|
| Baseline | 1.00 | 1.00 | 1.00 | 0.00 |
| Manual VSM | 0.98 | 0.92 | 0.97 | 0.00 |
| PM-VSM | 0.96 | 0.85 | 0.95 | 0.00 |
| PM-VSM-XGB | 0.92 | 0.78 | 0.88 | 0.00 |
| PM-VSM-XGB-DT | 0.88 | 0.68 | 0.82 | 0.02 |
| Full Framework | 0.84 | 0.58 | 0.72 | 0.06 |

Las diferencias son reales dentro de ese cálculo reproducible. No son una estimación de cuánto mejora un proceso al incorporar PM, ML o CNA: 04 no lee salidas de los modelos, SHAP o la red para determinar los factores. El término `network_gain` es un supuesto del escenario, no una ganancia estimada de la red.

`wip_mean` se genera a partir de una normal centrada en la espera dividida entre ocho, truncada en cero y multiplicada por el factor adicional. `throughput_per_hour` usa un calendario construido con 7.5 minutos por orden multiplicados por el factor de espera más el lead time medio. No son conteos de una simulación de colas con recursos. En el manuscrito deben describirse como proxies del modelo, con sus convenciones explícitas.

Las réplicas de distintos escenarios usan las mismas semillas `1000 + replication`. Las trayectorias de números aleatorios pueden divergir por ramas que dependen del escenario, pero las etiquetas de réplica siguen definiendo bloques de semilla compartidos. El análisis adicional por bloques evita tratarlas simplemente como seis conjuntos independientes. Ni los p-valores ni los intervalos convierten los supuestos de factores en efectos causales de módulos.

## 4. Capacidades descritas frente a lo ejecutado

Los scripts entregados construyen una red de actividades y un DFG mediante NetworkX. No se encontraron ejecuciones de redes de recursos o flujos de valor, una capa SHAP, conformance checking ni un simulador SimPy. En 07, PM4Py se usa para leer XES y convertirlos en dataframe. Estas capacidades pueden formar parte de la arquitectura propuesta, pero no deben presentarse como componentes empíricamente ejecutados por estos scripts sin materiales adicionales.

La porción de validación de 09 se reserva y no se usa para selección ni ajuste. Eso debe describirse así, en lugar de afirmar que se seleccionaron hiperparámetros con ella. Los modelos tienen parámetros fijos. El generador usa siete estaciones de ruta y puede insertar una octava actividad “Rework”; el manuscrito no debe confundir número de estaciones y actividades observadas.

## 5. Auditoría ejecutada de BPI2017, BPI2019 y DQI

Se recuperaron y ejecutaron los tres CSV preparados exactos, con hashes en `input_manifest.json`. La preparación original de los dos BPI no está implementada en el script 07 entregado; esta limitación de procedencia se declara, sin impedir reproducir los experimentos sobre los CSV proporcionados.

- BPI2017: 250,000 eventos, 31,469 casos y 26 actividades. La ejecución original 09 conserva 249,533 filas y reproduce RF de 0.2243 a 1.0000; XGBoost también alcanza 1.0000 con la red de entrenamiento.
- BPI2019: 250,000 eventos, 155,771 casos y 39 actividades. La ejecución original reproduce RF de 0.4658 a 0.7451 y XGBoost de 0.0062 a 0.7453.
- DQI: 8,607 eventos, 271 casos y 21 actividades. El lector original conserva 8,601 filas. Varias cifras RF de la nueva ejecución del original difieren del redondeo publicado; el informe no atribuye esa diferencia a una causa no comprobada.

### Corrección de lectura y ordenamiento

El lector original usa inferencia de un único formato de fechas. En BPI2017 rechaza 248 cadenas de inicio y 249 de fin, con 467 filas distintas eliminadas. Con formato mixto todas esas fechas son válidas; ocurre una pérdida análoga de seis filas válidas en DQI. Los scripts corregidos 08 y 09 usan `format="mixed", utc=True` y ordenamiento temporal estable de filas para explicitar los empates. Eso conserva 250,000 filas BPI2017 y 8,607 DQI. BPI2019 conserva 250,000 filas; el cambio de orden de empates puede modificar la representación y el ajuste del modelo, por lo que sus nuevos resultados se presentan como una ejecución corregida, no como una reproducción numéricamente idéntica.

La tabla del manuscrito y sus figuras se actualizan desde los resultados corregidos. La comparación completa original–corregido está en `results/original_vs_corrected_real_F1.csv`. El F1 perfecto de BPI2017 se mantiene. En el RF corregido, BPI2017 pasa de 0.2233 a 1.0000; BPI2019 de 0.4665 a 0.9608; DQI rework de 0.9352 a 0.9463. Las otras tareas y XGBoost constan en los CSV.

### Control por identidad de actividad

| dataset | test_events | test_positives | operational_RF | operational_plus_network_RF | activity_majority | activity_one_hot_RF |
| --- | --- | --- | --- | --- | --- | --- |
| Synthetic | 1280 | 21 | 0.2500 | 1.0000 | 1.0000 | 1.0000 |
| BPI2017 | 37500 | 846 | 0.2233 | 1.0000 | 1.0000 | 1.0000 |
| BPI2019 | 37500 | 1276 | 0.4665 | 0.9608 | 1.0000 | 1.0000 |
| DQI | 1292 | 948 | 0.9352 | 0.9463 | 0.8464 | 0.4817 |


En BPI2017, todos los positivos y únicamente ellos pertenecen a `O_Returned`. En BPI2019, todos los positivos corresponden a un conjunto de actividades de cambios y cada ocurrencia de esas actividades es positiva. Las líneas base por actividad se aprenden únicamente en entrenamiento, y ambas logran F1 = 1.0000 en los dos BPI. Esto explica la separabilidad del objetivo y evita atribuirla a anticipación de retrabajo.

En DQI, el flag coincide con la repetición de una actividad ya observada dentro del mismo caso. Las líneas base y el clasificador se refieren a etiquetas retrospectivas actuales, no a un resultado futuro independiente.

### Disponibilidad temporal y calidad de datos

El grafo predictivo de 09 se estima únicamente con entrenamiento. Los CSV de auditoría guardan la red, las firmas de actividad, los conteos por clase, las predicciones y las matrices de confusión. El split corregido comparte siete casos BPI2017 y 2,424 casos BPI2019 entre entrenamiento y prueba; DQI comparte cero. No se afirma evaluación de casos completamente inéditos. El hecho de que una actividad sea conocida en el evento actual no permite decir que se predijo el retrabajo antes de que ocurriera.

El rango de inicio BPI2019 incluye 1948-01-26 a 2019-01-18. El span de 622,190.53 horas se conserva como metadato observado, no como horizonte industrial validado para interpretar throughput físico. No se borran esos registros sin una regla de limpieza justificada por su procedencia.

## Cambios necesarios en el artículo

1. Corregir la definición de bottleneck y las exclusiones según 09.
2. Explicar el F1 sintético con la línea base por actividad y distinguirlo de BPI2017.
3. Describir los cinco porcentajes como respuestas de escenarios Monte Carlo con factores predefinidos; conservar sus valores, que se reproducen.
4. Mostrar los factores y explicitar los proxies de WIP y throughput.
5. Actualizar las estadísticas con el análisis por bloques y declarar su alcance condicional.
6. Distinguir capacidades propuestas de componentes ejecutados: SHAP, redes adicionales, SimPy, conformance y sincronización no están demostrados por los scripts recibidos.
7. Los resultados reales ya fueron recalculados y contrastados con líneas base por actividad. La revisión describe clasificación retrospectiva; la predicción de resultados futuros permanece fuera de lo demostrado.

## Conclusión de preparación para envío

La reproducción de BPI2017 ya no está pendiente. Los cinco porcentajes sintéticos se reproducen, y el F1 perfecto se conserva con una explicación verificable por actividad. El manuscrito y la respuesta se actualizan con los controles ejecutados, las correcciones de fechas, las estadísticas por bloques y el alcance real de los escenarios.

La evidencia valida los cálculos y la clasificación de etiquetas preparadas. No demuestra una predicción prospectiva, beneficios causales de módulos analíticos, ni mejoras implementadas en una planta. La versión revisada declara esos límites y presenta las capacidades adicionales como arquitectura propuesta. El informe no garantiza aceptación editorial.

## Corrección de consistencia G4

La ejecución de 04 produce un aumento de throughput G4 de aproximadamente 46.58%, que redondea a 46.6%. La tabla original mostraba 46.4%, mientras la figura original mostraba 46.6%. El artículo y las nuevas figuras usan ahora el valor ejecutado. Los cinco porcentajes headline de G5 no cambian.
