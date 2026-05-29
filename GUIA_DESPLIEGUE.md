# Guía de despliegue — Dashboard Aeropuertos Colombia

## Estructura de carpetas que debes tener

```
dashboard_aeropuertos/
├── app.py
├── requirements.txt
└── data/
    ├── metrics_validation_all_models.csv
    ├── metrics_test_final_model.csv
    ├── confusion_matrix_test_final.csv
    ├── metrics_cv_summary.csv
    ├── feature_importance_final_model.csv
    ├── test_error_summary.csv
    ├── final_model_metadata.json
    └── resumen_final.json
```

---

## Paso 1 — Probar localmente (opcional pero recomendado)

```bash
# En la terminal, desde la carpeta dashboard_aeropuertos/
pip install -r requirements.txt
streamlit run app.py
```
Abre http://localhost:8501 en el navegador. Si se ve bien, sigue al paso 2.

---

## Paso 2 — Subir a GitHub

1. Ve a https://github.com y crea una cuenta si no tienes.
2. Crea un repositorio nuevo:
   - Haz clic en **New** (botón verde, arriba a la derecha).
   - Nombre: `dashboard-aeropuertos` (o el que quieras).
   - Visibilidad: **Public** (obligatorio para Streamlit gratis).
   - Haz clic en **Create repository**.

3. Sube los archivos. Desde la terminal:

```bash
# Dentro de la carpeta dashboard_aeropuertos/
git init
git add .
git commit -m "Dashboard inicial aeropuertos Colombia"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/dashboard-aeropuertos.git
git push -u origin main
```

> Reemplaza `TU_USUARIO` con tu usuario de GitHub.

---

## Paso 3 — Desplegar en Streamlit Community Cloud

1. Ve a https://share.streamlit.io
2. Inicia sesión con tu cuenta de GitHub.
3. Haz clic en **New app**.
4. Rellena:
   - **Repository**: `TU_USUARIO/dashboard-aeropuertos`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Haz clic en **Deploy!**

En 2–3 minutos tendrás una URL pública del tipo:
```
https://dashboard-aeropuertos-XXXX.streamlit.app
```

---

## Notas importantes

- Si al desplegar aparece error de módulo, verifica que `requirements.txt` esté en la raíz.
- La carpeta `data/` con los CSV y JSON debe estar en el repositorio.
- Si GitHub te pide token para hacer push, ve a Settings → Developer Settings → Personal Access Tokens → Generate new token (classic), marca `repo`, y úsalo como contraseña.
