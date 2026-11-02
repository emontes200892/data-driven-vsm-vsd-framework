# IJAMT — paquete auditado de código y manuscrito

Se reprodujeron los resultados sintéticos y los tres datasets preparados reales. El F1 perfecto de BPI2017 se reproduce y se explica con una línea base por actividad. El informe completo está en `AUDIT_REPORT.md`.

## Contenido

- `code/original_scripts/`: los 14 scripts entregados, con nombres normalizados y contenido conservado.
- `code/scripts/`: implementación activa; se corrigen 03, 05, 07, 08 y 09, y se documenta el alcance de 04. Los scripts 14 y 15 añaden las auditorías y las estadísticas por bloques.
- `code/data/`: CSV sintéticos originales. El generador vuelve a producir exactamente el CSV entregado.
- `code/datasets/prepared/`: los tres CSV preparados utilizados en la auditoría real.
- `code/results/`: resultados activos, con fechas de formato mixto y orden temporal estable.
- `results/original_reproduction/` y `results/real_reproduction/`: ejecuciones originales conservadas por separado.
- `results/*_reported_audit/`: líneas base que siguen la lectura original.
- `results/*_corrected_audit/`: líneas base con la lectura corregida; incluyen predicciones y matrices de confusión.
- `results/paired_scenario_analysis/`: 30 bloques de semilla, intervalos bootstrap y pruebas ajustadas.
- `manuscript/`: LaTeX, bibliografía, figuras y respuesta al revisor actualizados a la evidencia ejecutada.
- `input_manifest.json` y `source_hashes.json`: hashes de entradas y scripts originales.

## Ejecutar

Desde `code/`, instale `requirements-tested.txt` en un entorno aislado. Las versiones documentan el entorno de la auditoría, no el entorno histórico de los autores.

```bash
python -m pip install -r requirements-tested.txt
python run_audit.py
```

El runner ejecuta los cálculos sintéticos, los tres datasets preparados, las estadísticas corregidas y las auditorías por actividad. No descarga fuentes nuevas ni sobrescribe los CSV de entrada. La regeneración opcional de los datos sintéticos se hace ejecutando `scripts/00_generate_synthetic_dataset.py` y sí sobrescribe `data/`.

La conversión de XES del script 07 requiere además PM4Py y los archivos raw. No se revalidó esa conversión en esta auditoría ni se presenta como procedencia de los dos BPI. Los CSV preparados incluidos permiten reproducir los análisis realizados.

Para compilar el artículo, entre a `manuscript/` y ejecute pdfLaTeX, BibTeX y dos pasadas de pdfLaTeX. La respuesta compila aparte con pdfLaTeX. Los PDF de lectura se entregan fuera del ZIP de fuentes para respetar la solicitud editorial de archivos editables.

## Interpretación

Los cinco porcentajes operativos son respuestas de escenarios con factores predefinidos. La clasificación perfecta de BPI identifica etiquetas ligadas a la actividad. El paquete no demuestra predicción de resultados futuros ni implementación en una planta física. El texto revisado conserva esa distinción.
