# Cambios de código

- 03: red solo de entrenamiento, exclusión de variables que definen delay/bottleneck, etiquetas presentadas como diagnóstico retrospectivo, predicciones guardadas.
- 04: documentación explícita de Monte Carlo y factores supuestos; no se alteraron sus parámetros ni sus resultados.
- 05: estadísticas por bloques de semilla, Wilcoxon/Friedman, Holm e intervalos bootstrap; originales conservados.
- 07: falla si no hay case_id y admite fechas mixtas. No se revalidó la conversión raw ni se usa para afirmar la procedencia BPI.
- 08/09: lectura UTC de formatos mixtos y orden estable; 09 falla si falta el objetivo o una variable operativa, y guarda metadatos.
- 14: nueva auditoría con controles por actividad y redes de entrenamiento.
- 15: nuevo análisis de escenarios que respeta bloques y no atribuye efectos a módulos.

Se conserva la carpeta original_scripts para reproducir el comportamiento anterior.

- 06/12/13 delegan las figuras al nuevo script 16, que lee resultados ejecutados. La figura publicada se alimenta desde un JSON derivado y con hashes, no de cifras escritas a mano.
- 11 exporta el texto auditado y revisado; no genera conclusiones automáticamente a partir de F1.
- G4 throughput se corrige a 46.6% desde la salida ejecutada de aproximadamente 46.58%.
