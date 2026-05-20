# 💳 credit-approval-ml-pipeline

> 📊 End-to-end credit approval analysis | ETL · EDA · ML models (LR, RF, XGBoost) | 91% accuracy · AUC 0.97

Pipeline completo de ciencia de datos sobre el dataset **Credit Approval (UCI)** para predecir la aprobación o rechazo de solicitudes de crédito a partir de variables sociodemográficas y financieras.

---

## 📁 Estructura del proyecto

```
credit-approval-ml-pipeline/
│
├── credit_pipeline.py          # Pipeline principal ejecutable
├── credit_data_clean.csv       # Dataset limpio generado por el pipeline
└── README.md
```

---

## ⚙️ ETL — Preparación de datos

El dataset original contiene **690 registros × 16 columnas** con variables anonimizadas (A1–A15 + Class).

Transformaciones realizadas:

- **Renombrado de columnas** a nombres descriptivos en español
- **Reinterpretación de variables**: `anios_empleo` e `ingreso_mensual` identificadas como binarias; `consultas_buro` y `saldo_cuenta` corregidas desde nombres erróneos
- **Mapeo de categóricas** a texto legible (`genero`, `estado_civil`, `nivel_educativo`, `tiene_propiedad`, `tipo_cuenta_bancaria`, `anios_empleo`, `ingreso_mensual`)
- **Transformación de edad**: aplicación de `floor` para convertir decimal a entero
- **Eliminación de outliers**: 35 registros de menores de 18 años eliminados por invalidez legal e inconsistencias internas

**Dataset final limpio: 655 filas × 16 columnas — sin nulos ni duplicados**

---

## 🔍 EDA — Análisis Exploratorio

### Perfil del dataset
| Variable | Dato destacado |
|---|---|
| Género | 68.4% hombres |
| Estado civil | 75.7% casados |
| Nivel educativo | 58.2% Bachillerato |
| Empleo | 54.2% empleados |
| Tipo de cuenta | 90.7% cuenta de ahorro |
| Variable objetivo | 54.2% rechazados / 45.8% aprobados ✅ balanceado |

### Hallazgos clave — Aprobado vs Rechazado
| Variable | Tasa aprobación |
|---|---|
| Empleados | 78.6% |
| No empleados | 7.0% |
| Con propiedad | 73.2% |
| Sin propiedad | 25.3% |
| Cuenta corriente | 57.1% |
| Sin cuenta | 27.8% |

### Correlaciones más relevantes con `aprobado`
| Variable | Correlación | Dirección |
|---|---|---|
| `anios_empleo` | -0.72 | No empleo → Rechazo |
| `tiene_propiedad` | +0.48 | Propiedad → Aprobación |
| `consultas_buro` | +0.41 | Más consultas → Aprobación |
| `num_dependientes` | +0.38 | Más dependientes → Aprobación |
| `historial_crediticio` | +0.32 | Más historial → Aprobación |

---

## 🤖 Modelado — Machine Learning

Se entrenaron y compararon tres modelos de clasificación binaria:

| Modelo | Accuracy | ROC-AUC | F1 Rechazado | F1 Aprobado | Falsos Positivos |
|---|---|---|---|---|---|
| Regresión Logística | 89% | 0.9688 | 0.90 | 0.89 | 8 |
| ✅ **Random Forest** | **91%** | 0.9663 | **0.91** | **0.90** | **7** |
| XGBoost | 86% | 0.9568 | 0.87 | 0.85 | 10 |

### 🏆 Modelo ganador: Random Forest
- **91% de accuracy** y **AUC 0.9663**
- Minimiza los falsos positivos (7) — el error más costoso en contexto crediticio
- Mejor balance entre precisión y recall en ambas clases

### Variables más importantes (consenso 3 modelos)
1. 🥇 `anios_empleo_No empleado` — predictor dominante de rechazo
2. 🥈 `tiene_propiedad_Sí` — mayor predictor de aprobación
3. 🥉 `consultas_buro` — perfil financieramente activo
4. `antiguedad_cuenta` — historial de relación bancaria
5. `num_dependientes` — factor positivo

---

## 🚀 Cómo ejecutar

### 1. Instalar dependencias
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2. Configurar rutas en `credit_pipeline.py`
```python
INPUT_FILE  = r"ruta\a\tu\Credit_Card_Applications.csv"
OUTPUT_FILE = r"ruta\donde\guardar\credit_data_clean.csv"
```

### 3. Ejecutar
```bash
python credit_pipeline.py
```

El pipeline ejecuta automáticamente ETL → EDA → Modelado y genera todos los gráficos y métricas.

---

## 🛠️ Tecnologías utilizadas

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-2.0-lightblue)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-1.8-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.10-green)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)

---

## 📌 Próximos pasos

- [ ] Tuning de hiperparámetros con `GridSearchCV`
- [ ] Cross-validation para validar robustez
- [ ] SHAP values para explicabilidad de predicciones individuales
- [ ] Mejora de calidad en variables con alta proporción de valores desconocidos
