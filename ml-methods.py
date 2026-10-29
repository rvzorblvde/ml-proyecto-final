"""
Segundo Reporte - Avance del proyecto
Prediccion de demanda de inventario en sistemas de Punto de Venta
utilizando metodos de regresion.

Guillermo Jair Munoz Amaro - Aprendizaje Automatico - UASLP

Dataset: Online Retail (UCI)
    https://archive.ics.uci.edu/dataset/352/online+retail
    Se asume que "online-retail-dataset.xlsx" esta en la misma carpeta que este
    script. Si no existe o tiene errores, se descarga automaticamente con
    ucimlrepo (id=352) y, si eso falla, desde el enlace directo de UCI.
    (Tambien puedes pasar otra ruta:  python avance_proyecto.py ruta/archivo.xlsx)

Tarea: regresion sobre serie temporal tabular.
Observacion = un producto en una semana. Objetivo = unidades vendidas esa semana.

Dependencias: pip install pandas numpy scikit-learn matplotlib openpyxl ucimlrepo
Salidas: carpeta figuras/ (PNG) y carpeta resultados/ (CSV).
"""
import io
import os
import sys
import urllib.request
import warnings
import zipfile

import matplotlib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

# ----------------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------------
SEED = 42
RUTA_DATOS = sys.argv[1] if len(sys.argv) > 1 else "online-retail-dataset.xlsx"  # .xlsx o .csv
TOP_N = 50              # numero de productos a modelar
EVALUAR_TEST = False    # mantener False hasta elegir el modelo final
MOSTRAR_GRAFICAS = False  # True = abre cada ventana (hay que cerrarla para continuar)

if not MOSTRAR_GRAFICAS:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (debe ir despues de elegir el backend)

warnings.filterwarnings("ignore")
np.random.seed(SEED)
os.makedirs("figuras", exist_ok=True)
os.makedirs("resultados", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def titulo(texto):
    print("\n" + "=" * 78 + f"\n{texto}\n" + "=" * 78)


def guardar(nombre):
    plt.tight_layout()
    plt.savefig(f"figuras/{nombre}.png")
    if MOSTRAR_GRAFICAS:
        plt.show()
    plt.close()


# ----------------------------------------------------------------------------
# 3.3 DATASET UTILIZADO Y ESTADO ACTUAL DE LOS DATOS
# ----------------------------------------------------------------------------
titulo("3.3 Dataset utilizado")
COLUMNAS = ["InvoiceNo", "StockCode", "Description", "Quantity",
            "InvoiceDate", "UnitPrice", "CustomerID", "Country"]
URL_UCI = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"


def validar(d):
    """Verifica que el DataFrame tenga las columnas esperadas y fechas validas."""
    faltan = [c for c in COLUMNAS if c not in d.columns]
    if faltan:
        raise ValueError(f"Faltan columnas: {faltan}")
    d = d[COLUMNAS].copy()
    d["InvoiceDate"] = pd.to_datetime(d["InvoiceDate"], errors="raise")
    if len(d) == 0:
        raise ValueError("El dataset esta vacio")
    return d


def desde_archivo(ruta):
    if ruta.lower().endswith(".csv"):
        return validar(pd.read_csv(ruta, encoding="latin-1"))
    return validar(pd.read_excel(ruta))


def desde_ucimlrepo():
    from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo

    online_retail = fetch_ucirepo(id=352)
    X = online_retail.data.features
    y = online_retail.data.targets
    print(online_retail.metadata)
    print(online_retail.variables)
    # Por si el repositorio entrega features y targets por separado
    d = X if y is None else pd.concat([X, y], axis=1)
    return validar(d)


def desde_enlace():
    with urllib.request.urlopen(URL_UCI, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    nombre = next(n for n in z.namelist() if n.lower().endswith((".xlsx", ".csv")))
    with z.open(nombre) as f:
        datos = io.BytesIO(f.read())
    return validar(pd.read_csv(datos, encoding="latin-1") if nombre.lower().endswith(".csv")
                   else pd.read_excel(datos))


def cargar_dataset():
    fuentes = [
        (f"archivo local '{RUTA_DATOS}'", lambda: desde_archivo(RUTA_DATOS), os.path.exists(RUTA_DATOS)),
        ("paquete ucimlrepo (id=352)", desde_ucimlrepo, True),
        (f"enlace directo {URL_UCI}", desde_enlace, True),
    ]
    for nombre, cargar, disponible in fuentes:
        if not disponible:
            print(f"[aviso] No existe {nombre}; se intenta la siguiente fuente.")
            continue
        try:
            d = cargar()
            print(f"[ok] Dataset cargado desde {nombre}")
            return d
        except Exception as e:  # noqa: BLE001
            print(f"[aviso] Fallo la carga desde {nombre}: {type(e).__name__}: {e}")
    sys.exit("No fue posible cargar el dataset. Descargalo manualmente de "
             "https://archive.ics.uci.edu/dataset/352/online+retail")


df = cargar_dataset()

print(f"Observaciones (lineas de factura): {len(df):,} | Variables: {df.shape[1]}")
print(f"Periodo: {df.InvoiceDate.min().date()} a {df.InvoiceDate.max().date()}")
print("\nMuestra de los datos:")
print(df.head().to_string())

tabla_vars = pd.DataFrame(
    [
        ["InvoiceNo", "Categorica", "Num. de factura (prefijo 'C' = cancelacion)", "Identificador"],
        ["StockCode", "Categorica", "Codigo del producto", "Entrada (producto) / Identificador"],
        ["Description", "Texto", "Nombre del producto", "Exploracion"],
        ["Quantity", "Numerica (entero)", "Unidades por linea de factura", "Objetivo (agregado por semana)"],
        ["InvoiceDate", "Fecha/hora", "Fecha y hora de la transaccion", "Entrada (estacionalidad)"],
        ["UnitPrice", "Numerica (float)", "Precio unitario en libras (GBP)", "Entrada"],
        ["CustomerID", "Categorica", "Identificador de cliente", "No utilizada"],
        ["Country", "Categorica", "Pais del cliente", "No utilizada"],
    ],
    columns=["Variable", "Tipo de dato", "Descripcion / unidad", "Funcion"],
)
print("\nTabla de variables:")
print(tabla_vars.to_string(index=False))
tabla_vars.to_csv("resultados/tabla_variables.csv", index=False)

# ----------------------------------------------------------------------------
# 3.4 PREPARACION DE LOS DATOS
# ----------------------------------------------------------------------------
titulo("3.4 Preparacion de los datos")
bitacora = []


def registrar(situacion, tratamiento, afectados, antes, despues):
    bitacora.append([situacion, tratamiento, f"{afectados:,} registros", f"{len(antes):,} -> {len(despues):,}"])


d = df.copy()

# 1) Duplicados exactos
a = d.drop_duplicates()
registrar("Filas duplicadas", "Eliminacion", len(d) - len(a), d, a)
d = a
# 2) Cancelaciones / devoluciones (factura con prefijo C)
m = d.InvoiceNo.astype(str).str.startswith("C")
a = d[~m]
registrar("Cancelaciones (InvoiceNo 'C...')", "Eliminacion", int(m.sum()), d, a)
d = a
# 3) Cantidades no positivas
m = d.Quantity <= 0
a = d[~m]
registrar("Quantity <= 0", "Eliminacion", int(m.sum()), d, a)
d = a
# 4) Precios no positivos
m = d.UnitPrice <= 0
a = d[~m]
registrar("UnitPrice <= 0 (ajustes/regalos)", "Eliminacion", int(m.sum()), d, a)
d = a
# 5) Codigos que no son productos (POST, DOT, M, BANK CHARGES, etc.)
m = ~d.StockCode.astype(str).str.match(r"^\d{5}[A-Za-z]*$")
a = d[~m]
registrar("StockCode que no es producto", "Eliminacion", int(m.sum()), d, a)
d = a
# 6) Informativo: CustomerID nulo (no se usa como variable)
registrar("CustomerID nulo (informativo)", "Variable descartada", int(d.CustomerID.isna().sum()), d, d)

d = d.copy()
d["StockCode"] = d["StockCode"].astype(str)
d["semana"] = d.InvoiceDate.dt.to_period("W-SUN").dt.start_time  # semanas lunes-domingo

# 7) Semanas incompletas en los extremos del periodo
semanas_all = sorted(d.semana.unique())
a = d[~d.semana.isin([semanas_all[0], semanas_all[-1]])]
registrar("Primera y ultima semana incompletas", "Eliminacion", len(d) - len(a), d, a)
d = a

tabla_prep = pd.DataFrame(
    bitacora,
    columns=["Situacion detectada", "Tratamiento realizado", "Resultado (afectados)", "Filas antes -> despues"],
)
print(tabla_prep.to_string(index=False))
tabla_prep.to_csv("resultados/tabla_preparacion.csv", index=False)

# --- Seleccion de productos: los que se venden en mas semanas (series mas continuas)
semanas = sorted(d.semana.unique())
act = d.groupby("StockCode").agg(semanas_activas=("semana", "nunique"), unidades=("Quantity", "sum"))
top = act.sort_values(["semanas_activas", "unidades"], ascending=False).head(TOP_N).index.tolist()
cobertura = d[d.StockCode.isin(top)].Quantity.sum() / d.Quantity.sum()
print(f"\n{len(semanas)} semanas completas | {TOP_N} productos seleccionados "
      f"({cobertura:.1%} de las unidades totales)")

# --- Agregacion a panel producto-semana (semanas sin ventas = demanda 0)
idx = pd.MultiIndex.from_product([top, semanas], names=["StockCode", "semana"])
panel = (
    d[d.StockCode.isin(top)]
    .groupby(["StockCode", "semana"])
    .agg(unidades=("Quantity", "sum"), precio_medio=("UnitPrice", "mean"))
    .reindex(idx)
)
panel["unidades"] = panel.unidades.fillna(0)
panel["precio_medio"] = panel.groupby(level=0).precio_medio.transform(lambda s: s.ffill().bfill())
panel = panel.reset_index()

# --- Dias festivos (UK bank holidays del periodo) -> num. de festivos en la semana
festivos = pd.to_datetime([
    "2010-12-27", "2010-12-28", "2011-01-03", "2011-04-22", "2011-04-25",
    "2011-04-29", "2011-05-02", "2011-05-30", "2011-08-29", "2011-12-26", "2011-12-27",
])
panel["dia_festivo"] = panel.semana.apply(
    lambda s: int(((festivos >= s) & (festivos < s + pd.Timedelta(days=7))).sum())
)

# --- Variables temporales
iso = panel.semana.dt.isocalendar()
panel["semana_anio"] = iso.week.astype(int)
panel["mes"] = panel.semana.dt.month
panel["sin_sem"] = np.sin(2 * np.pi * panel.semana_anio / 52)
panel["cos_sem"] = np.cos(2 * np.pi * panel.semana_anio / 52)

# --- Rezagos (solo informacion PASADA, disponible al momento de predecir)
panel = panel.sort_values(["StockCode", "semana"])
g = panel.groupby("StockCode").unidades
panel["lag_1"] = g.shift(1)
panel["lag_2"] = g.shift(2)
panel["lag_4"] = g.shift(4)
panel["media_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
antes = len(panel)
panel = panel.dropna(subset=["lag_1", "lag_2", "lag_4", "media_4"])
print(f"Filas sin historial suficiente para los rezagos: {antes - len(panel)} (se eliminan)")
print("Panel final:", panel.shape)
print(panel.head().to_string())
panel.to_csv("resultados/panel_producto_semana.csv", index=False)

# ----------------------------------------------------------------------------
# 3.5 ANALISIS EXPLORATORIO Y HALLAZGOS
# ----------------------------------------------------------------------------
titulo("3.5 Analisis exploratorio")

# Frontera temporal de la separacion (60% / 20% / 20% de las semanas)
sem = sorted(panel.semana.unique())
i1, i2 = int(len(sem) * 0.6), int(len(sem) * 0.8)
f_val, f_test = sem[i1], sem[i2]

# Figura 1: distribucion del objetivo
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(panel.unidades, bins=60, color="#4c72b0")
ax[0].set_title("Distribucion de unidades por producto-semana")
ax[0].set_xlabel("Unidades vendidas")
ax[0].set_ylabel("Frecuencia")
ax[1].hist(np.log1p(panel.unidades), bins=40, color="#55a868")
ax[1].set_title("Misma variable en escala log(1+x)")
ax[1].set_xlabel("log(1 + unidades)")
guardar("eda_1_distribucion")
print("Estadisticas de la variable objetivo:")
print(panel.unidades.describe().round(1).to_string())
print(f"% de ceros: {(panel.unidades == 0).mean():.1%}")

# Figura 2: serie semanal total
total = panel.groupby("semana").unidades.sum()
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(total.index, total.values, marker="o", ms=3, color="#c44e52")
ax.axvline(f_val, ls="--", c="k")
ax.axvline(f_test, ls="--", c="k")
ax.text(f_val, total.max(), " val", va="top")
ax.text(f_test, total.max(), " test", va="top")
for f in festivos:
    ax.axvline(f, color="orange", alpha=0.25)
ax.set_title(f"Unidades semanales totales ({TOP_N} productos). Naranja = festivos UK")
ax.set_xlabel("Semana")
ax.set_ylabel("Unidades")
guardar("eda_2_serie_semanal")

# Figura 3: ventas por dia de la semana
dias = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"]
dd = d[d.StockCode.isin(top)].copy()
dd["dow"] = dd.InvoiceDate.dt.dayofweek
por_dia = dd.groupby("dow").Quantity.sum().reindex(range(7), fill_value=0)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.bar([dias[i] for i in por_dia.index], por_dia.values, color="#8172b2")
ax.set_title("Unidades vendidas por dia de la semana")
ax.set_ylabel("Unidades")
guardar("eda_3_dia_semana")

# Figura 4: correlaciones
cols_num = ["unidades", "precio_medio", "dia_festivo", "semana_anio", "lag_1", "lag_2", "lag_4", "media_4"]
corr = panel[cols_num].corr()
fig, ax = plt.subplots(figsize=(6.5, 5.2))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.grid(False)
ax.set_xticks(range(len(cols_num)))
ax.set_xticklabels(cols_num, rotation=45, ha="right")
ax.set_yticks(range(len(cols_num)))
ax.set_yticklabels(cols_num)
for i in range(len(cols_num)):
    for j in range(len(cols_num)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
plt.colorbar(im)
ax.set_title("Correlacion entre variables numericas")
guardar("eda_4_correlacion")
print("\nCorrelacion de cada variable con 'unidades':")
print(corr["unidades"].drop("unidades").sort_values(key=abs, ascending=False).round(3).to_string())

# ----------------------------------------------------------------------------
# 3.6 DISENO EXPERIMENTAL Y PRIMERA IMPLEMENTACION
# ----------------------------------------------------------------------------
titulo("3.6 Diseno experimental")
FEATS_NUM = ["precio_medio", "dia_festivo", "sin_sem", "cos_sem", "lag_1", "lag_2", "lag_4", "media_4"]
FEATS_CAT = ["StockCode"]
OBJ = "unidades"
FEATS = FEATS_NUM + FEATS_CAT

train = panel[panel.semana < f_val]
val = panel[(panel.semana >= f_val) & (panel.semana < f_test)]
test = panel[panel.semana >= f_test]  # RESERVADO: no se usa para elegir modelo
for n, s in [("Entrenamiento", train), ("Validacion", val), ("Prueba (reservada)", test)]:
    print(f"{n:20s} {len(s):5d} filas | {s.semana.min().date()} -> {s.semana.max().date()}")


def prep(poly=False):
    num = [("sc", StandardScaler())]
    if poly:
        num.append(("poly", PolynomialFeatures(2, include_bias=False)))
    return ColumnTransformer([
        ("num", Pipeline(num), FEATS_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATS_CAT),
    ])
# El escalador y el codificador se ajustan SOLO con entrenamiento (viven dentro del Pipeline)


modelos = {
    "Regresion lineal": Pipeline([("p", prep()), ("m", LinearRegression())]),
    "Regresion polinomial (g=2, Ridge)": Pipeline([("p", prep(True)), ("m", Ridge(alpha=10, random_state=SEED))]),
    "SVR kernel RBF": Pipeline([
        ("p", prep()),
        ("m", TransformedTargetRegressor(
            SVR(kernel="rbf", C=10, epsilon=0.1, gamma="scale"), transformer=StandardScaler())),
    ]),
}

# ----------------------------------------------------------------------------
# 3.7 RESULTADOS PRELIMINARES (conjunto de validacion)
# ----------------------------------------------------------------------------
titulo("3.7 Resultados preliminares (validacion)")


def metricas(nombre, conjunto, y, yp):
    mse = mean_squared_error(y, yp)
    return {
        "Modelo / configuracion": nombre, "Conjunto": conjunto,
        "MAE": mean_absolute_error(y, yp), "MSE": mse, "RMSE": np.sqrt(mse), "R2": r2_score(y, yp),
    }


preds = {}
# Referencias
base_media = DummyRegressor(strategy="mean").fit(train[FEATS], train[OBJ])
preds["Referencia: media"] = base_media.predict(val[FEATS])
preds["Referencia: persistencia (lag_1)"] = val["lag_1"].values
# Modelos
for nombre, pipe in modelos.items():
    pipe.fit(train[FEATS], train[OBJ])
    preds[nombre] = pipe.predict(val[FEATS])

tabla_res = pd.DataFrame([metricas(k, "Validacion", val[OBJ], p) for k, p in preds.items()]).round(3)
print(tabla_res.to_string(index=False))
tabla_res.to_csv("resultados/resultados_validacion.csv", index=False)

# Figura 5: comparacion de modelos y real vs predicho
t = tabla_res.set_index("Modelo / configuracion")
mejor = t.drop([i for i in t.index if i.startswith("Referencia")])["RMSE"].idxmin()
print(f"\nMejor modelo (menor RMSE en validacion): {mejor}")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
t["RMSE"].plot.barh(ax=ax[0], color="#4c72b0")
ax[0].set_title("RMSE en validacion (menor = mejor)")
ax[0].set_xlabel("Unidades")
ax[1].scatter(val[OBJ], preds[mejor], s=8, alpha=0.5)
lim = max(val[OBJ].max(), preds[mejor].max())
ax[1].plot([0, lim], [0, lim], "r--")
ax[1].set_title(f"Real vs. predicho - {mejor}")
ax[1].set_xlabel("Unidades reales")
ax[1].set_ylabel("Unidades predichas")
guardar("res_1_comparacion")

# Figura 6: ejemplo por producto (el de mayor venta en validacion)
prod = val.groupby("StockCode").unidades.sum().idxmax()
mask = val.StockCode.values == prod
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(val.semana[mask], val.unidades[mask], marker="o", label="Real")
ax.plot(val.semana[mask], np.array(preds[mejor])[mask], marker="s", label=mejor)
ax.set_title(f"Producto {prod}: demanda semanal en validacion")
ax.set_ylabel("Unidades")
ax.legend()
guardar("res_2_producto")

# --- Prueba final: SOLO cuando el modelo y la configuracion ya esten definidos ---
if EVALUAR_TEST:
    final = modelos[mejor]
    trval = pd.concat([train, val])
    final.fit(trval[FEATS], trval[OBJ])
    res_test = metricas(mejor, "Prueba", test[OBJ], final.predict(test[FEATS]))
    print("\nResultado en PRUEBA:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res_test.items()})

print("\nListo. Figuras en ./figuras  |  tablas en ./resultados")
