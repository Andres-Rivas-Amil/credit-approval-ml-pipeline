"""
=============================================================================
PIPELINE — ANÁLISIS CREDITICIO
=============================================================================
Descripción : Pipeline completo de ETL, EDA y Modelado ML sobre datos
              crediticios del dataset Credit Approval (UCI).
Autor       : -
Fecha       : 2026
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from xgboost import XGBClassifier

# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
RANDOM_STATE = 42
TEST_SIZE    = 0.2
INPUT_FILE  = r"C:\Users\andy_\Downloads\Credit_Card_Applications.csv"
OUTPUT_FILE = r"C:\Users\andy_\Documents\credit_data_clean.csv"


# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
def cargar_datos(filepath: str) -> pd.DataFrame:
    """Carga el dataset desde un CSV."""
    df = pd.read_csv(filepath)
    print(f"✅ Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


# =============================================================================
# 2. ETL — TRANSFORMACIÓN DE DATOS
# =============================================================================
def etl(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de ETL:
    - Renombrado de columnas
    - Reinterpretación de variables
    - Mapeo de categóricas a texto
    - Transformaciones numéricas
    - Eliminación de outliers
    """
    print("\n── ETL ──────────────────────────────────────────────────────────")

    # 2.1 Renombrar columnas
    df = df.rename(columns={
        'A1':    'genero',
        'A2':    'edad',
        'A3':    'deuda',
        'A4':    'estado_civil',
        'A5':    'num_dependientes',
        'A6':    'nivel_educativo',
        'A7':    'historial_crediticio',
        'A8':    'anios_empleo',
        'A9':    'tiene_propiedad',
        'A10':   'consultas_buro',
        'A11':   'ingreso_mensual',
        'A12':   'tipo_cuenta_bancaria',
        'A13':   'tipo_cuenta',
        'A14':   'saldo_cuenta',
        'A15':   'antiguedad_cuenta',
        'Class': 'aprobado'
    })
    print("✅ Columnas renombradas")

    # 2.2 Renombrar columnas reinterpretadas
    df = df.rename(columns={
        'tipo_cuenta': 'saldo_cuenta',
        'saldo_cuenta': 'antiguedad_cuenta',
        'antiguedad_cuenta': 'saldo_cuenta_orig'
    })

    # 2.3 Mapeo de variables categóricas
    df['genero'] = df['genero'].map({0: 'Femenino', 1: 'Masculino'})

    df['estado_civil'] = df['estado_civil'].map({
        1: 'Soltero/a',
        2: 'Casado/a',
        3: 'Otro'
    })

    df['nivel_educativo'] = df['nivel_educativo'].map({
        1: 'Sin estudios',
        2: 'Primaria',
        3: 'Secundaria',
        4: 'Bachillerato',
        5: 'Técnico',
        7: 'Posgrado',
        8: 'Otro',
        9: 'Desconocido'
    })

    df['tiene_propiedad'] = df['tiene_propiedad'].map({0: 'No', 1: 'Sí'})

    df['tipo_cuenta_bancaria'] = df['tipo_cuenta_bancaria'].map({
        1: 'Sin cuenta',
        2: 'Ahorro',
        3: 'Corriente'
    })

    df['anios_empleo']    = df['anios_empleo'].map({0: 'No empleado', 1: 'Empleado'})
    df['ingreso_mensual'] = df['ingreso_mensual'].map({0: 'Sin ingreso', 1: 'Con ingreso'})
    print("✅ Variables categóricas mapeadas")

    # 2.4 Transformaciones numéricas
    df['edad'] = np.floor(df['edad']).astype(int)
    print("✅ Edad truncada con floor")

    # 2.5 Eliminar menores de 18
    n_antes = len(df)
    df = df[df['edad'] >= 18].reset_index(drop=True)
    print(f"✅ Menores de 18 eliminados: {n_antes - len(df)} filas")
    print(f"✅ Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")

    return df


# =============================================================================
# 3. EDA — ANÁLISIS EXPLORATORIO
# =============================================================================
def eda(df: pd.DataFrame) -> None:
    """
    Análisis exploratorio completo:
    - Estadísticas descriptivas
    - Distribución de variables numéricas y categóricas
    - Análisis aprobado vs rechazado
    - Matriz de correlación
    """
    print("\n── EDA ──────────────────────────────────────────────────────────")

    # 3.1 Estadísticas descriptivas
    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df.describe())

    print("\n=== DISTRIBUCIÓN APROBADO ===")
    print(df['aprobado'].value_counts())
    print(df['aprobado'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%')

    # 3.2 Distribución variables numéricas
    numericas = ['edad', 'deuda', 'consultas_buro', 'saldo_cuenta',
                 'antiguedad_cuenta', 'historial_crediticio']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        axes[i].hist(df[col], bins=30, color='#3498db', edgecolor='black', alpha=0.8)
        axes[i].set_title(f'Distribución: {col}', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frecuencia')
    plt.suptitle('Distribución de Variables Numéricas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3.3 Distribución variables categóricas
    categoricas = ['genero', 'estado_civil', 'nivel_educativo', 'tiene_propiedad',
                   'tipo_cuenta_bancaria', 'anios_empleo', 'ingreso_mensual', 'aprobado']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(categoricas):
        orden = df[col].value_counts().index
        sns.countplot(data=df, x=col, ax=axes[i], order=orden,
                      palette='Blues_r', edgecolor='black')
        axes[i].set_title(f'Distribución: {col}', fontsize=12)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Frecuencia')
        axes[i].tick_params(axis='x', rotation=15)
        total = len(df)
        for p in axes[i].patches:
            pct = f'{100 * p.get_height() / total:.1f}%'
            axes[i].annotate(pct, (p.get_x() + p.get_width() / 2,
                             p.get_height() + 3), ha='center', fontsize=9)
    axes[-1].set_visible(False)
    plt.suptitle('Distribución de Variables Categóricas', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3.4 Tasa de aprobación por variable categórica
    cat_analisis = ['genero', 'estado_civil', 'nivel_educativo', 'tiene_propiedad',
                    'tipo_cuenta_bancaria', 'anios_empleo', 'ingreso_mensual']

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for i, col in enumerate(cat_analisis):
        tasa = df.groupby(col)['aprobado'].mean().mul(100).round(1).sort_values(ascending=False)
        bars = axes[i].bar(tasa.index, tasa.values, color='#2ecc71', edgecolor='black', alpha=0.85)
        axes[i].set_title(f'% Aprobación por {col}', fontsize=12)
        axes[i].set_ylabel('% Aprobados')
        axes[i].set_ylim(0, 100)
        axes[i].axhline(y=df['aprobado'].mean() * 100, color='red',
                        linestyle='--', linewidth=1.2, label='Media global')
        axes[i].legend(fontsize=8)
        axes[i].tick_params(axis='x', rotation=15)
        for bar, val in zip(bars, tasa.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2,
                         val + 1.5, f'{val}%', ha='center', fontsize=9)
    axes[-1].set_visible(False)
    axes[-2].set_visible(False)
    plt.suptitle('Tasa de Aprobación por Variable Categórica', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3.5 Boxplots numéricos vs aprobado
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        sns.boxplot(data=df,
                    x=df['aprobado'].map({0: 'Rechazado', 1: 'Aprobado'}),
                    y=col, ax=axes[i],
                    palette={'Rechazado': '#e74c3c', 'Aprobado': '#2ecc71'},
                    order=['Rechazado', 'Aprobado'], width=0.5)
        axes[i].set_title(f'{col} por Aprobado/Rechazado', fontsize=12)
        axes[i].set_xlabel('')
    plt.suptitle('Variables Numéricas vs Aprobado', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 3.6 Matriz de correlación
    df_corr = df.copy()
    cols_cat = ['genero', 'estado_civil', 'nivel_educativo', 'tiene_propiedad',
                'tipo_cuenta_bancaria', 'anios_empleo', 'ingreso_mensual']
    for col in cols_cat:
        df_corr[col] = df_corr[col].astype('category').cat.codes
    df_corr = df_corr.drop(columns=['CustomerID'])

    fig, ax = plt.subplots(figsize=(14, 10))
    mask = np.triu(np.ones_like(df_corr.corr(), dtype=bool))
    sns.heatmap(df_corr.corr(), annot=True, fmt='.2f', cmap='RdYlGn',
                mask=mask, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title('Matriz de Correlación', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("✅ EDA completado")


# =============================================================================
# 4. PREPARACIÓN PARA MODELADO
# =============================================================================
def preparar_datos(df: pd.DataFrame):
    """
    Prepara X e y, aplica One-Hot Encoding y StandardScaler.
    Retorna X_train, X_test, y_train, y_test y el scaler.
    """
    print("\n── PREPARACIÓN PARA MODELADO ────────────────────────────────────")

    cols_cat = ['genero', 'estado_civil', 'nivel_educativo', 'tiene_propiedad',
                'tipo_cuenta_bancaria', 'anios_empleo', 'ingreso_mensual']

    df_model = pd.get_dummies(df, columns=cols_cat, drop_first=True)
    df_model = df_model.drop(columns=['CustomerID'])

    X = df_model.drop(columns=['aprobado'])
    y = df_model['aprobado']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    cols_num = ['edad', 'deuda', 'consultas_buro', 'saldo_cuenta',
                'antiguedad_cuenta', 'historial_crediticio', 'num_dependientes']

    scaler = StandardScaler()
    X_train[cols_num] = scaler.fit_transform(X_train[cols_num])
    X_test[cols_num]  = scaler.transform(X_test[cols_num])

    print(f"✅ Train: {X_train.shape} | Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, scaler


# =============================================================================
# 5. MODELADO
# =============================================================================
def entrenar_evaluar(nombre, modelo, X_train, X_test, y_train, y_test,
                     color, resultados_roc):
    """Entrena un modelo, muestra métricas y guarda curva ROC."""
    print(f"\n=== {nombre.upper()} ===")
    modelo.fit(X_train, y_train)

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred,
                                 target_names=['Rechazado', 'Aprobado']))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # Matriz de confusión
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Rechazado', 'Aprobado'],
                yticklabels=['Rechazado', 'Aprobado'])
    axes[0].set_title(f'Matriz de Confusión — {nombre}', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Real')
    axes[0].set_xlabel('Predicho')

    # Importancia de variables
    if hasattr(modelo, 'coef_'):
        coefs = pd.DataFrame({'variable': X_train.columns,
                               'valor': modelo.coef_[0]})
        coefs = coefs.reindex(coefs['valor'].abs().sort_values().index)
        colors_bar = ['#e74c3c' if v < 0 else '#2ecc71' for v in coefs['valor']]
        axes[1].barh(coefs['variable'], coefs['valor'],
                     color=colors_bar, edgecolor='black', alpha=0.85)
        axes[1].axvline(x=0, color='black', linewidth=0.8, linestyle='--')
        axes[1].set_title(f'Coeficientes — {nombre}', fontsize=13, fontweight='bold')
    else:
        imp = pd.DataFrame({'variable': X_train.columns,
                             'importancia': modelo.feature_importances_})
        imp = imp.sort_values('importancia', ascending=True)
        axes[1].barh(imp['variable'], imp['importancia'],
                     color=color, edgecolor='black', alpha=0.85)
        axes[1].set_title(f'Importancia Variables — {nombre}', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    resultados_roc.append((nombre, fpr, tpr, auc, color))

    return modelo


def comparar_roc(resultados_roc):
    """Gráfica comparativa de curvas ROC de todos los modelos."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for nombre, fpr, tpr, auc, color in resultados_roc:
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{nombre} AUC = {auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title('Curva ROC — Comparativa Modelos', fontsize=13, fontweight='bold')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# 6. PIPELINE PRINCIPAL
# =============================================================================
def main():
    # ── Carga ──────────────────────────────────────────────────────────────
    df = cargar_datos(INPUT_FILE)

    # ── ETL ────────────────────────────────────────────────────────────────
    df = etl(df)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Dataset limpio guardado en '{OUTPUT_FILE}'")

    # ── EDA ────────────────────────────────────────────────────────────────
    eda(df)

    # ── Modelado ───────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler = preparar_datos(df)

    resultados_roc = []

    modelos = [
        ("Regresión Logística",
         LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
         '#3498db'),
        ("Random Forest",
         RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
         '#2ecc71'),
        ("XGBoost",
         XGBClassifier(n_estimators=100, random_state=RANDOM_STATE,
                       eval_metric='logloss', verbosity=0),
         '#e67e22'),
    ]

    modelos_entrenados = {}
    for nombre, modelo, color in modelos:
        modelos_entrenados[nombre] = entrenar_evaluar(
            nombre, modelo, X_train, X_test, y_train, y_test,
            color, resultados_roc
        )

    comparar_roc(resultados_roc)

    print("\n✅ Pipeline completado")
    return df, modelos_entrenados, scaler


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    df_clean, modelos, scaler = main()