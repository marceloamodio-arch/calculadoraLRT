#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULADORA INDEMNIZACIONES LEY 24.557
Sistema de cálculo de indemnizaciones laborales
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import base64

# Configuración de la página
st.set_page_config(
    page_title="Calculadora Indemnizaciones LRT",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para replicar el diseño original
st.markdown("""
<style>
    /* Colores principales */
    :root {
        --primary: #2E86AB;
        --secondary: #A23B72;
        --success: #F18F01;
        --info: #C73E1D;
        --light: #F8F9FA;
        --dark: #343A40;
        --highlight-ripte: #E8F5E8;
        --highlight-tasa: #E8F5E8;
    }
    
    /* Ocultar Deploy y menú de 3 puntos */
    button[kind="header"] {
        display: none;
    }
    
    /* Ocultar los 3 puntos verticales */
    [data-testid="stHeader"] svg[viewBox="0 0 16 16"] {
        display: none;
    }
    
    /* Ocultar footer */
    footer {
        display: none;
    }
    
    /* Header personalizado */
    .main-header {
        background-color: #2E86AB;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: bold;
    }
    
    .main-header h2 {
        margin: 5px 0 0 0;
        font-size: 18px;
        font-weight: normal;
    }
    
    /* Tarjetas de resultados */
    .result-card {
        background-color: #F8F9FA;
        border-left: 4px solid #2E86AB;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    .result-card.highlight-ripte {
        background-color: #E8F5E8;
        border-left-color: #28a745;
    }
    
    .result-card.highlight-tasa {
        background-color: #E8F5E8;
        border-left-color: #28a745;
    }
    
    .result-card h3 {
        color: #2E86AB;
        font-size: 16px;
        margin-bottom: 10px;
    }
    
    .result-amount {
        font-size: 32px;
        font-weight: bold;
        color: #343A40;
        margin: 10px 0;
    }
    
    .result-detail {
        font-size: 14px;
        color: #666;
        margin-top: 10px;
    }
    
    /* Alertas */
    .alert-box {
        background-color: #C73E1D;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .alert-box h4 {
        margin-top: 0;
    }
    
    /* Fórmula */
    .formula-box {
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        padding: 15px;
        border-radius: 8px;
        font-family: monospace;
        margin: 20px 0;
    }
    
    /* Botones personalizados */
    .stButton>button {
        background-color: #2E86AB;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 10px 25px;
        border: none;
    }
    
    .stButton>button:hover {
        background-color: #1a5f7a;
    }
    
    /* Tablas */
    .dataframe {
        font-size: 14px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
</style>
""", unsafe_allow_html=True)

# Alineación vertical corregida sin modificar ancho
st.markdown("""
<style>
    /* Mantener columnas proporcionales */
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start !important;
    }

    /* Tarjetas con alturas coherentes */
    .result-card {
        width: 100% !important;
        min-height: 230px;   /* altura mínima homogénea */
        margin-bottom: 18px; /* separación equilibrada entre tarjetas */
    }

    /* Ajuste solo para la última tarjeta (Últimos Datos Disponibles) */
    .result-card:last-child {
        margin-top: 32px; /* compensa visualmente la altura menor de la derecha */
    }
</style>
""", unsafe_allow_html=True)

# Password por defecto
DEFAULT_PASSWORD = "todosjuntos"

# Paths de datasets
DATASET_DIR = os.path.abspath(os.path.dirname(__file__))
PATH_RIPTE = os.path.join(DATASET_DIR, "dataset_ripte.csv")
PATH_TASA = os.path.join(DATASET_DIR, "dataset_tasa.csv")
PATH_IPC = os.path.join(DATASET_DIR, "dataset_ipc.csv")
PATH_PISOS = os.path.join(DATASET_DIR, "dataset_pisos.csv")

def safe_parse_date(s) -> Optional[date]:
    """Función corregida de parseo de fechas"""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return None
    if isinstance(s, (datetime, date)):
        return s.date() if isinstance(s, datetime) else s
    s = str(s).strip()
    if not s:
        return None
    
    fmts = [
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%Y", "%Y/%m/%d", "%Y-%m",
        "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S", "%B %Y", "%b %Y", 
        "%Y/%m", "%m-%Y",
    ]
    
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if f in ("%m/%Y", "%Y-%m", "%Y/%m", "%m-%Y", "%B %Y", "%b %Y"):
                return date(dt.year, dt.month, 1)
            return dt.date()
        except Exception:
            continue
    
    if "/" in s or "-" in s:
        parts = s.replace("/", "-").split("-")
        if len(parts) == 2:
            try:
                year, month = int(parts[0]), int(parts[1])
                if 1900 <= year <= 2100 and 1 <= month <= 12:
                    return date(year, month, 1)
            except ValueError:
                pass
    
    try:
        dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
        if pd.isna(dt):
            return None
        if isinstance(dt, pd.Timestamp):
            return dt.date()
        return None
    except Exception:
        return None

def days_in_month(d: date) -> int:
    """Días en el mes"""
    if d.month == 12:
        nxt = date(d.year + 1, 1, 1)
    else:
        nxt = date(d.year, d.month + 1, 1)
    return (nxt - date(d.year, d.month, 1)).days

def numero_a_letras(numero):
    """Convierte un número a su representación en letras (pesos argentinos)"""
    unidades = ['', 'UN', 'DOS', 'TRES', 'CUATRO', 'CINCO', 'SEIS', 'SIETE', 'OCHO', 'NUEVE']
    decenas = ['', '', 'VEINTE', 'TREINTA', 'CUARENTA', 'CINCUENTA', 'SESENTA', 'SETENTA', 'OCHENTA', 'NOVENTA']
    especiales = ['DIEZ', 'ONCE', 'DOCE', 'TRECE', 'CATORCE', 'QUINCE', 'DIECISÉIS', 'DIECISIETE', 'DIECIOCHO', 'DIECINUEVE']
    centenas = ['', 'CIENTO', 'DOSCIENTOS', 'TRESCIENTOS', 'CUATROCIENTOS', 'QUINIENTOS', 'SEISCIENTOS', 'SETECIENTOS', 'OCHOCIENTOS', 'NOVECIENTOS']
    
    def convertir_grupo(n):
        if n == 0:
            return ''
        elif n == 100:
            return 'CIEN'
        elif n < 10:
            return unidades[n]
        elif n < 20:
            return especiales[n - 10]
        elif n < 100:
            dec = n // 10
            uni = n % 10
            if uni == 0:
                return decenas[dec]
            else:
                return decenas[dec] + (' Y ' if dec > 2 else 'I') + unidades[uni]
        else:
            cen = n // 100
            resto = n % 100
            if resto == 0:
                return centenas[cen]
            else:
                return centenas[cen] + ' ' + convertir_grupo(resto)
    
    if numero == 0:
        return 'CERO PESOS'
    
    entero = int(numero)
    decimal = int(round((numero - entero) * 100))
    
    if entero >= 1000000000:
        miles_millon = entero // 1000000000
        resto = entero % 1000000000
        texto = convertir_grupo(miles_millon) + ' MIL'
        if resto >= 1000000:
            millones = resto // 1000000
            resto = resto % 1000000
            texto += ' ' + (convertir_grupo(millones) if millones > 1 else 'UN') + ' MILLÓN' + ('ES' if millones > 1 else '')
        if resto > 0:
            if resto >= 1000:
                miles = resto // 1000
                resto = resto % 1000
                texto += ' ' + convertir_grupo(miles) + ' MIL'
            if resto > 0:
                texto += ' ' + convertir_grupo(resto)
    elif entero >= 1000000:
        millones = entero // 1000000
        resto = entero % 1000000
        texto = (convertir_grupo(millones) if millones > 1 else 'UN') + ' MILLÓN' + ('ES' if millones > 1 else '')
        if resto > 0:
            if resto >= 1000:
                miles = resto // 1000
                resto = resto % 1000
                texto += ' ' + convertir_grupo(miles) + ' MIL'
            if resto > 0:
                texto += ' ' + convertir_grupo(resto)
    elif entero >= 1000:
        miles = entero // 1000
        resto = entero % 1000
        texto = convertir_grupo(miles) + ' MIL'
        if resto > 0:
            texto += ' ' + convertir_grupo(resto)
    else:
        texto = convertir_grupo(entero)
    
    return f'PESOS {texto} CON {decimal:02d}/100'

def get_mes_nombre(mes):
    """Retorna el nombre del mes en español"""
    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
    return meses[mes - 1]

@dataclass
class InputData:
    """Estructura para los datos de entrada"""
    pmi_date: date
    final_date: date
    ibm: float
    edad: int
    incapacidad_pct: float
    incluir_20_pct: bool

@dataclass
class Results:
    """Estructura para los resultados de cálculo"""
    capital_formula: float
    capital_base: float
    piso_aplicado: bool
    piso_info: str
    piso_monto: float
    piso_proporcional: float
    piso_norma: str
    adicional_20_pct: float
    
    ripte_coef: float
    ripte_pmi: float
    ripte_final: float
    ripte_actualizado: float
    interes_puro_3_pct: float
    total_ripte_3: float
    
    tasa_activa_pct: float
    total_tasa_activa: float
    
    inflacion_acum_pct: float

class DataManager:
    """Gestor de datasets CSV"""
    
    def __init__(self):
        self.ipc_data = None
        self.pisos_data = None
        self.ripte_data = None
        self.tasa_data = None
        self.load_all_datasets()
    
    def _load_csv(self, path):
        """Carga CSV con múltiples separadores"""
        if not os.path.exists(path):
            st.error(f"No se encontró el dataset: {path}")
            return pd.DataFrame()
        
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(path, sep=sep)
                if df.shape[1] >= 1:
                    return df
            except Exception:
                continue
        
        try:
            return pd.read_csv(path, sep=",", encoding="latin-1")
        except Exception as e:
            st.error(f"No se pudo leer el dataset {path}.\n{e}")
            return pd.DataFrame()
    
    def load_all_datasets(self):
        """Carga todos los datasets"""
        try:
            self.ripte_data = self._load_csv(PATH_RIPTE)
            self.tasa_data = self._load_csv(PATH_TASA)  
            self.ipc_data = self._load_csv(PATH_IPC)
            self.pisos_data = self._load_csv(PATH_PISOS)
            
            self._norm_ripte()
            self._norm_tasa()
            self._norm_ipc()
            self._norm_pisos()
                
        except Exception as e:
            st.error(f"Error cargando datasets: {str(e)}")
    
    def _norm_ripte(self):
        """Normalización RIPTE"""
        if self.ripte_data.empty: 
            return
        cols = [c.lower() for c in self.ripte_data.columns]
        self.ripte_data.columns = cols
        
        if 'año' in cols and 'mes' in cols:
            meses_dict = {
                'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
                'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12,
                'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
                'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12,
                'ene': 1, 'abr': 4, 'ago': 8, 'set': 9, 'dic': 12
            }
            
            def convertir_mes(valor):
                if pd.isna(valor):
                    return None
                valor_str = str(valor).strip().lower()
                
                try:
                    return int(float(valor_str))
                except ValueError:
                    pass
                
                if valor_str in meses_dict:
                    return meses_dict[valor_str]
                
                for mes_nombre, mes_num in meses_dict.items():
                    if mes_nombre.startswith(valor_str[:3]) or valor_str.startswith(mes_nombre[:3]):
                        return mes_num
                
                return None
            
            def crear_fecha_combined(row):
                try:
                    año = int(row['año'])
                    mes_num = convertir_mes(row['mes'])
                    if mes_num is None:
                        return None
                    return f"{año}-{mes_num:02d}-01"
                except (ValueError, TypeError):
                    return None
            
            self.ripte_data['fecha_combined'] = self.ripte_data.apply(crear_fecha_combined, axis=1)
            fecha_col = 'fecha_combined'
        else:
            fecha_col = None
            for c in cols:
                if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                    fecha_col = c
                    break
            if fecha_col is None:
                fecha_col = cols[0]
        
        val_col = None
        if 'indice_ripte' in cols:
            val_col = 'indice_ripte'
        else:
            for c in cols:
                if ("ripte" in c) or ("valor" in c) or ("indice" in c):
                    val_col = c
                    break
            if val_col is None:
                num_cols = self.ripte_data.select_dtypes(include="number").columns.tolist()
                val_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        self.ripte_data["fecha"] = self.ripte_data[fecha_col].apply(safe_parse_date)
        self.ripte_data["ripte"] = pd.to_numeric(self.ripte_data[val_col], errors="coerce")
        self.ripte_data = self.ripte_data.dropna(subset=["fecha", "ripte"]).sort_values("fecha").reset_index(drop=True)

    def _norm_tasa(self):
        """Normalización TASA"""
        if self.tasa_data.empty:
            return

        # normaliza nombres de columnas (y limpia BOM si existiera)
        cols = [str(c).strip().lower().replace("\ufeff", "") for c in self.tasa_data.columns]
        self.tasa_data.columns = cols

        # parseo de fechas
        if "desde" in self.tasa_data.columns:
            self.tasa_data["desde"] = self.tasa_data["desde"].apply(safe_parse_date)
        if "hasta" in self.tasa_data.columns:
            self.tasa_data["hasta"] = self.tasa_data["hasta"].apply(safe_parse_date)
        else:
            if "desde" in self.tasa_data.columns:
                self.tasa_data["hasta"] = self.tasa_data["desde"]

        if "desde" in self.tasa_data.columns:
            self.tasa_data["fecha"] = self.tasa_data["desde"]

        # columna base para la tasa
        base_col = None
        for cand in ("valor", "porcentaje", "tasa"):
            if cand in self.tasa_data.columns:
                base_col = cand
                break

        if base_col is not None:
            # convierte "3,982" o "3.982" → 3.982 (sin eliminar punto decimal)
            self.tasa_data["tasa"] = (
                self.tasa_data[base_col]
                .astype(str)
                .str.replace(",", ".", regex=False)             
            )
            self.tasa_data["tasa"] = pd.to_numeric(self.tasa_data["tasa"], errors="coerce")

        # columnas útiles y orden
        keep_cols = [c for c in ("fecha", "tasa", "desde", "hasta") if c in self.tasa_data.columns]
        if "fecha" in self.tasa_data.columns and "tasa" in self.tasa_data.columns:
            self.tasa_data = (
                self.tasa_data.dropna(subset=["fecha", "tasa"])
                .sort_values("fecha")
                .reset_index(drop=True)
            )[keep_cols]

    def _norm_ipc(self):
        """Normalización IPC"""
        if self.ipc_data.empty: 
            return
        cols = [c.lower() for c in self.ipc_data.columns]
        self.ipc_data.columns = cols
        
        fecha_col = None
        if 'periodo' in cols:
            fecha_col = 'periodo'
        else:
            for c in cols:
                if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                    fecha_col = c
                    break
            if fecha_col is None:
                fecha_col = cols[0]
        
        val_col = None
        if 'variacion_mensual' in cols:
            val_col = 'variacion_mensual'
        else:
            for c in cols:
                if ("variacion" in c) or ("inflacion" in c) or ("ipc" in c) or ("porcentaje" in c) or ("mensual" in c) or ("indice" in c):
                    val_col = c
                    break
            if val_col is None:
                num_cols = self.ipc_data.select_dtypes(include="number").columns.tolist()
                val_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        self.ipc_data["fecha"] = self.ipc_data[fecha_col].apply(safe_parse_date)
        self.ipc_data["ipc"] = pd.to_numeric(self.ipc_data[val_col], errors="coerce")
        self.ipc_data = self.ipc_data.dropna(subset=["fecha", "ipc"]).sort_values("fecha").reset_index(drop=True)

    def _norm_pisos(self):
        """Normalización PISOS"""
        if self.pisos_data.empty: 
            return
        cols = [c.lower() for c in self.pisos_data.columns]
        self.pisos_data.columns = cols
        
        # El dataset tiene: fecha_inicio, fecha_fin, norma, monto_minimo, enlace
        self.pisos_data["desde"] = self.pisos_data["fecha_inicio"].apply(safe_parse_date)
        self.pisos_data["hasta"] = self.pisos_data["fecha_fin"].apply(safe_parse_date)
        self.pisos_data["piso"] = pd.to_numeric(self.pisos_data["monto_minimo"], errors="coerce")
        self.pisos_data["resol"] = self.pisos_data["norma"].astype(str)
        self.pisos_data["enlace"] = self.pisos_data["enlace"].astype(str).replace('nan', '')
        
        self.pisos_data = self.pisos_data.dropna(subset=["desde", "piso"]).sort_values("desde").reset_index(drop=True)
    
    def get_piso_minimo(self, fecha_pmi: date) -> Tuple[Optional[float], str]:
        """Obtiene piso mínimo"""
        if self.pisos_data.empty:
            return (None, "")
            
        candidate = None
        for _, r in self.pisos_data.iterrows():
            d0 = r["desde"]
            d1 = r["hasta"] if not pd.isna(r["hasta"]) else None
            if d1 is None:
                if fecha_pmi >= d0:
                    candidate = (float(r["piso"]), r.get("resol", ""))
            else:
                if d0 <= fecha_pmi <= d1:
                    return (float(r["piso"]), r.get("resol", ""))
        return candidate if candidate else (None, "")
    
    def get_ripte_coeficiente(self, fecha_pmi: date, fecha_final: date) -> Tuple[float, float, float]:
        """Cálculo RIPTE"""
        if self.ripte_data.empty:
            return 1.0, 0.0, 0.0
        
        ripte_pmi_data = self.ripte_data[self.ripte_data['fecha'] <= fecha_pmi]
        if ripte_pmi_data.empty:
            ripte_pmi = float(self.ripte_data.iloc[0]['ripte'])
        else:
            ripte_pmi = float(ripte_pmi_data.iloc[-1]['ripte'])
        
        ripte_final = float(self.ripte_data.iloc[-1]['ripte'])
        
        coeficiente = ripte_final / ripte_pmi if ripte_pmi > 0 else 1.0
        
        return coeficiente, ripte_pmi, ripte_final
    
    def calcular_tasa_activa(self, fecha_pmi: date, fecha_final: date, capital_base: float) -> Tuple[float, float]:
        """Cálculo de tasa activa"""
        if self.tasa_data.empty:
            return 0.0, capital_base
            
        total_aporte_pct = 0.0
        
        for _, row in self.tasa_data.iterrows():
            if "desde" in self.tasa_data.columns and not pd.isna(row.get("desde")):
                fecha_desde = row["desde"]
            else:
                fecha_desde = row["fecha"]
                
            if "hasta" in self.tasa_data.columns and not pd.isna(row.get("hasta")):
                fecha_hasta = row["hasta"]
            else:
                fecha_hasta = date(fecha_desde.year, fecha_desde.month, days_in_month(fecha_desde))
            
            if isinstance(fecha_desde, pd.Timestamp):
                fecha_desde = fecha_desde.date()
            if isinstance(fecha_hasta, pd.Timestamp):
                fecha_hasta = fecha_hasta.date()
            
            inicio_interseccion = max(fecha_pmi, fecha_desde)
            fin_interseccion = min(fecha_final, fecha_hasta)
            
            if inicio_interseccion <= fin_interseccion:
                dias_interseccion = (fin_interseccion - inicio_interseccion).days + 1
                
                if "tasa" in self.tasa_data.columns and not pd.isna(row.get("tasa")):
                    valor_mensual_pct = float(row["tasa"])
                elif "valor" in self.tasa_data.columns and not pd.isna(row.get("valor")):
                    valor_mensual_pct = float(row["valor"])
                else:
                    continue
                
                aporte_pct = valor_mensual_pct * (dias_interseccion / 30.0)
                total_aporte_pct += aporte_pct
        
        total_actualizado = capital_base * (1.0 + total_aporte_pct / 100.0)
        
        return total_aporte_pct, total_actualizado
    
    def calcular_inflacion(self, fecha_pmi: date, fecha_final: date) -> float:
        """Cálculo de inflación"""
        if self.ipc_data.empty:
            return 0.0
            
        fecha_inicio_mes = pd.Timestamp(fecha_pmi.replace(day=1))
        fecha_final_mes = pd.Timestamp(fecha_final.replace(day=1))
        
        ipc_periodo = self.ipc_data[
            (pd.to_datetime(self.ipc_data['fecha']) >= fecha_inicio_mes) &
            (pd.to_datetime(self.ipc_data['fecha']) <= fecha_final_mes)
        ]
        
        if ipc_periodo.empty:
            return 0.0
        
        factor_acumulado = 1.0
        for _, row in ipc_periodo.iterrows():
            variacion = row['ipc']
            if not pd.isna(variacion):
                factor_acumulado *= (1 + variacion / 100)
        
        inflacion_acumulada = (factor_acumulado - 1) * 100
        return inflacion_acumulada

class Calculator:
    """Motor de cálculos"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def calcular_indemnizacion(self, input_data: InputData) -> Results:
        """Realiza todos los cálculos"""
        
        capital_formula = self._calcular_capital_formula(input_data)
        
        piso_minimo, piso_norma = self.data_manager.get_piso_minimo(input_data.pmi_date)
        capital_aplicado, piso_aplicado, piso_info, piso_proporcional = self._aplicar_piso_minimo(
            capital_formula, piso_minimo, piso_norma, input_data.incapacidad_pct
        )
        
        adicional_20_pct = capital_aplicado * 0.20 if input_data.incluir_20_pct else 0.0
        capital_base = capital_aplicado + adicional_20_pct
        
        ripte_coef, ripte_pmi, ripte_final = self.data_manager.get_ripte_coeficiente(
            input_data.pmi_date, input_data.final_date
        )
        ripte_actualizado = capital_base * ripte_coef
        
        dias_transcurridos = (input_data.final_date - input_data.pmi_date).days
        interes_puro_3_pct = ripte_actualizado * 0.03 * (dias_transcurridos / 365.0)
        total_ripte_3 = ripte_actualizado + interes_puro_3_pct
        
        tasa_activa_pct, total_tasa_activa = self.data_manager.calcular_tasa_activa(
            input_data.pmi_date, input_data.final_date, capital_base
        )
        
        inflacion_acum_pct = self.data_manager.calcular_inflacion(
            input_data.pmi_date, input_data.final_date
        )
        
        return Results(
            capital_formula=capital_formula,
            capital_base=capital_base,
            piso_aplicado=piso_aplicado,
            piso_info=piso_info,
            piso_monto=piso_minimo if piso_minimo else 0.0,
            piso_proporcional=piso_proporcional,
            piso_norma=piso_norma,
            adicional_20_pct=adicional_20_pct,
            ripte_coef=ripte_coef,
            ripte_pmi=ripte_pmi,
            ripte_final=ripte_final,
            ripte_actualizado=ripte_actualizado,
            interes_puro_3_pct=interes_puro_3_pct,
            total_ripte_3=total_ripte_3,
            tasa_activa_pct=tasa_activa_pct,
            total_tasa_activa=total_tasa_activa,
            inflacion_acum_pct=inflacion_acum_pct
        )
    
    def _calcular_capital_formula(self, input_data: InputData) -> float:
        """Calcula capital según fórmula"""
        return input_data.ibm * 53 * (65 / input_data.edad) * (input_data.incapacidad_pct / 100)
    
    def _aplicar_piso_minimo(self, capital_formula: float, piso_minimo: Optional[float], 
                           piso_norma: str, incapacidad_pct: float) -> Tuple[float, bool, str, float]:
        """Aplica piso mínimo si corresponde"""
        if piso_minimo is None:
            return capital_formula, False, "No se encontró piso mínimo para la fecha", 0.0
        
        piso_proporcional = piso_minimo * (incapacidad_pct / 100)
        
        if capital_formula >= piso_proporcional:
            return capital_formula, False, f"Supera piso mínimo {piso_norma}", piso_proporcional
        else:
            return piso_proporcional, True, f"Se aplica piso mínimo {piso_norma}", piso_proporcional

class NumberUtils:
    """Utilidades para formateo de números"""
    
    @staticmethod
    def format_money(amount: float) -> str:
        """Formatea cantidad como dinero argentino"""
        return f"$ {amount:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    
    @staticmethod
    def format_percentage(percentage: float) -> str:
        """Formatea porcentaje"""
        return f"{percentage:.2f}%".replace('.', ',')

# --- Carga forzada de datasets en cada ejecución ---
data_mgr = DataManager()
st.session_state.data_manager = data_mgr
st.session_state.calculator = Calculator(data_mgr)

if 'results' not in st.session_state:
    st.session_state.results = None
if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# Header personalizado
st.markdown("""
<div class="main-header">
    <h1>CALCULADORA INDEMNIZACIONES LEY 24.557</h1>
    <h2>Y ACTUALIZACIONES.</h2>
</div>
""", unsafe_allow_html=True)

# Sidebar para formulario
with st.sidebar:
    st.header("📋 Datos del Caso")
    
    pmi_date_input = st.date_input(
        "Fecha del siniestro (PMI)",
        value=date(2020, 1, 1),
        format="DD/MM/YYYY"
    )
    
    final_date_input = st.date_input(
        "Fecha final",
        value=date.today(),
        format="DD/MM/YYYY"
    )
    
    ibm = st.number_input(
        "Ingreso Base Mensual (IBM)",
        min_value=0.0,
        value=100000.0,
        step=1000.0,
        format="%.2f"
    )
    
    edad = st.number_input(
        "Edad del trabajador",
        min_value=18,
        max_value=100,
        value=45,
        step=1
    )
    
    incapacidad_pct = st.number_input(
        "Porcentaje de incapacidad (%)",
        min_value=0.01,
        max_value=100.0,
        value=50.0,
        step=0.1,
        format="%.2f"
    )
    
    incluir_20_pct = st.checkbox(
        "Incluir 20% (art. 3, Ley 26.773)",
        value=True
    )
      
    if st.button("🧮 CALCULAR", use_container_width=True, type="primary"):
        try:
            input_data = InputData(
                pmi_date=pmi_date_input,
                final_date=final_date_input,
                ibm=ibm,
                edad=edad,
                incapacidad_pct=incapacidad_pct,
                incluir_20_pct=incluir_20_pct
            )
            
            if input_data.pmi_date > input_data.final_date:
                st.error("La fecha PMI no puede ser posterior a la fecha final")
            else:
                st.session_state.results = st.session_state.calculator.calcular_indemnizacion(input_data)
                st.session_state.input_data = input_data
                st.success("✓ Cálculo realizado correctamente")
                st.rerun()
        except Exception as e:
            st.error(f"Error en el cálculo: {str(e)}")
      
    st.markdown("---")
    
    
# Main content - Resultados
if st.session_state.results is not None:
    results = st.session_state.results
    input_data = st.session_state.input_data
    
    # Tabs principales (agregamos tab6 para PDF)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resultados", 
        "📄 Sentencia", 
        "💰 Liquidación", 
        "📋 Mínimos SRT",
        "ℹ️ Información",
        "🖨️ Imprimir PDF"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Capital Base
            st.markdown(f"""
            <div class="result-card">
                <h3>CAPITAL BASE (INDEMNIZACIÓN LEY 24.557)</h3>
                <div class="result-amount">{NumberUtils.format_money(results.capital_base)}</div>
                <div class="result-detail">
                    Capital fórmula: {NumberUtils.format_money(results.capital_formula)}<br>
                    20%: {NumberUtils.format_money(results.adicional_20_pct) if results.adicional_20_pct > 0 else 'No aplica'}<br>
                    {results.piso_info}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # RIPTE + 3%
            highlight_class = "highlight-ripte" if results.total_ripte_3 >= results.total_tasa_activa else ""
            st.markdown(f"""
            <div class="result-card {highlight_class}">
                <h3>ACTUALIZACIÓN RIPTE + 3%</h3>
                <div class="result-amount">{NumberUtils.format_money(results.total_ripte_3)}</div>
                <div class="result-detail">
                    Coef. RIPTE: {results.ripte_coef:.6f}<br>
                    Total actualizado: {NumberUtils.format_money(results.ripte_actualizado)}<br>
                    3% puro: {NumberUtils.format_money(results.interes_puro_3_pct)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Tasa Activa
            highlight_class = "highlight-tasa" if results.total_tasa_activa > results.total_ripte_3 else ""
            st.markdown(f"""
            <div class="result-card {highlight_class}">
                <h3>ACTUALIZACIÓN TASA ACTIVA BNA</h3>
                <div class="result-amount">{NumberUtils.format_money(results.total_tasa_activa)}</div>
                <div class="result-detail">
                    Porcentual del período: {NumberUtils.format_percentage(results.tasa_activa_pct)}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
           # Inflación
            st.markdown(f"""
            <div class="result-card">
                <h3>INFLACIÓN ACUMULADA (Referencia)</h3>
                <div class="result-amount">{NumberUtils.format_percentage(results.inflacion_acum_pct)}</div>
                <div class="result-detail">
                    Inflación acumulada en el período
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Fórmula detallada
        st.markdown(f"""
        <div class="formula-box" style="color: #000;">
            <strong>Fórmula aplicada:</strong><br>
            IBM ({NumberUtils.format_money(input_data.ibm)}) × 53 × 65/edad({input_data.edad}) × Incapacidad ({input_data.incapacidad_pct}%)<br>
            <strong>Capital calculado:</strong> {NumberUtils.format_money(results.capital_formula)}
        </div>
        """, unsafe_allow_html=True)
    # 📊 Últimos Datos Disponibles 
    data_mgr = st.session_state.data_manager

    ultimo_ripte_txt = "RIPTE: N/D"
    ultimo_ipc_txt = "IPC: N/D"
    ultima_tasa_txt = "TASA ACTIVA: N/D"
    ultimo_piso_txt = "PISO SRT: N/D"

    # --- RIPTE ---
    if not data_mgr.ripte_data.empty:
        ultimo_ripte = data_mgr.ripte_data.iloc[-1]
        fecha_ripte = ultimo_ripte.get("fecha")
        valor_ripte = ultimo_ripte.get("ripte", 0)
        if pd.notnull(fecha_ripte):
            ultimo_ripte_txt = f"RIPTE {fecha_ripte.month}/{fecha_ripte.year}: {valor_ripte:,.0f}"

    # --- IPC: mostrar mes y año ---
    if not data_mgr.ipc_data.empty:
        ultimo_ipc = data_mgr.ipc_data.iloc[-1]
        mes_ipc = int(ultimo_ipc.get("mes", getattr(ultimo_ipc.get("fecha"), "month", 0)))
        año_ipc = int(ultimo_ipc.get("año", getattr(ultimo_ipc.get("fecha"), "year", 0)))
        variacion_ipc = ultimo_ipc.get("ipc", 0)
        ultimo_ipc_txt = f"IPC {mes_ipc}/{año_ipc}: {NumberUtils.format_percentage(variacion_ipc)}"

    # --- TASA ACTIVA: mostrar último día (columna 'hasta') ---
    if not data_mgr.tasa_data.empty:
        tasa_vista = data_mgr.tasa_data.copy()
        ultima_tasa = data_mgr.tasa_data.iloc[-1]
        valor_tasa = ultima_tasa.get("tasa", 0)
        fecha_hasta = ultima_tasa.get("hasta", None)
        fecha_txt = ""
        if pd.notnull(fecha_hasta):
            fecha_txt = fecha_hasta.strftime("%d/%m/%Y")
        ultima_tasa_txt = f"TASA ACTIVA {fecha_txt}: {NumberUtils.format_percentage(valor_tasa)}"

    # --- PISO SRT: mostrar período (desde / hasta) y resolución ---
    if not data_mgr.pisos_data.empty:
        ultimo_piso = data_mgr.pisos_data.iloc[-1]
        norma = ultimo_piso.get("resol", "")
        monto_piso = ultimo_piso.get("piso", 0)
        desde = ultimo_piso.get("desde", None)
        hasta = ultimo_piso.get("hasta", None)
        periodo = ""
        if pd.notnull(desde) and pd.notnull(hasta):
            periodo = f"{desde.strftime('%d/%m/%Y')} al {hasta.strftime('%d/%m/%Y')}"
        elif pd.notnull(desde):
            periodo = f"Desde {desde.strftime('%d/%m/%Y')}"
        elif pd.notnull(hasta):
            periodo = f"Hasta {hasta.strftime('%d/%m/%Y')}"
        ultimo_piso_txt = f"PISO SRT {norma} ({periodo}): {NumberUtils.format_money(monto_piso)}"

    # --- Render final con estilo original ---
    st.markdown(f"""
    <div class="formula-box" style="
        color: #000;
        font-family: 'Source Sans Pro', sans-serif;
        font-size: 14px;
        border: 1px solid #d3d3d3;
        border-radius: 8px;
        padding: 10px 15px;
        margin-top: 10px;
        background-color: #fff9c4;
        width: 100%;
">
        <strong>📊 Últimos Datos Disponibles:</strong><br>
        {ultimo_ripte_txt}<br>
        {ultimo_ipc_txt}<br>
        {ultima_tasa_txt}<br>
        {ultimo_piso_txt}
    </div>
    """, unsafe_allow_html=True)

    with tab2:
        st.subheader("📄 Texto para Sentencia")
        
        # Generar texto de sentencia según ejemplo
        mes_pmi = get_mes_nombre(input_data.pmi_date.month)
        anio_pmi = input_data.pmi_date.year
        
        # Determinar texto según si supera o no el piso
        if results.piso_aplicado:
            texto_piso = f"""El monto es inferior al piso mínimo determinado por la {results.piso_norma}, que multiplicado por el porcentaje de incapacidad ({input_data.incapacidad_pct}%) alcanza la suma de {NumberUtils.format_money(results.piso_proporcional)}, por lo que se aplica este último."""
        else:
            texto_piso = f"""Dicho monto supera el piso mínimo determinado por la {results.piso_norma}, que multiplicado por el porcentaje de incapacidad ({input_data.incapacidad_pct}%) alcanza la suma de {NumberUtils.format_money(results.piso_proporcional)}."""
        
        monto_letras = numero_a_letras(results.capital_base)
        
        sentencia_text = f"""a) Fórmula:
Valor de IBM ({NumberUtils.format_money(input_data.ibm)}) x 53 x 65/edad({input_data.edad}) x Incapacidad ({input_data.incapacidad_pct}%)
Capital calculado: {NumberUtils.format_money(results.capital_formula)}
{texto_piso}

b) {'20% Art. 3 Ley 26.773: ' + NumberUtils.format_money(results.adicional_20_pct) if input_data.incluir_20_pct else '20% Art. 3 Ley 26.773: no se aplica'}

Total: {NumberUtils.format_money(results.capital_base)}
SON {monto_letras}

c) Mientras la tasa legal aplicable (Tasa Activa Banco Nación) alcanzó para el período comprometido ({mes_pmi} {anio_pmi} a la fecha) un total del {NumberUtils.format_percentage(results.tasa_activa_pct)}, la inflación del mismo período alcanzó la suma de {NumberUtils.format_percentage(results.inflacion_acum_pct)}."""
        
        st.text_area("Texto de Sentencia", sentencia_text, height=450)
        
        if st.button("📋 Copiar Texto", key="copy_sentencia"):
            st.success("✓ Texto copiado al portapapeles")
    
    with tab3:
        st.subheader("💰 Liquidación Judicial")
        
        # Determinar método más favorable
        if results.total_ripte_3 >= results.total_tasa_activa:
            total_actualizacion = results.total_ripte_3
            metodo_usado = "tasa de variación RIPTE"
        else:
            total_actualizacion = results.total_tasa_activa
            metodo_usado = "Tasa Activa BNA"
        
        # Obtener fechas de RIPTE
        mes_final = get_mes_nombre(input_data.final_date.month)
        anio_final = input_data.final_date.year
        mes_pmi = get_mes_nombre(input_data.pmi_date.month)
        anio_pmi = input_data.pmi_date.year
        
        # Calcular porcentaje de incremento RIPTE
        pct_ripte = (results.ripte_coef - 1) * 100
        
        # Calcular tasas judiciales (2.2% según ejemplo)
        tasa_justicia = total_actualizacion * 0.022
        sobretasa_caja = tasa_justicia * 0.10
        total_final = total_actualizacion + tasa_justicia + sobretasa_caja
        
        # Convertir monto a letras
        monto_letras = numero_a_letras(total_final)
        
        liquidacion_text = f"""Quilmes, en la fecha en que se suscribe con firma digital (Ac. SCBA. 3975/20). 
**LIQUIDACION** que practica la Actuaria en el presente expediente. ** **

--Capital {NumberUtils.format_money(results.capital_base)} 
--Actualización mediante {metodo_usado}, ({get_mes_nombre(data_mgr.ripte_data.iloc[-1]['fecha'].month)}/{data_mgr.ripte_data.iloc[-1]['fecha'].year} {results.ripte_final:,.2f} -último índice publicado- / {mes_pmi} {anio_pmi} {results.ripte_pmi:,.2f} = coef {results.ripte_coef:.2f} = {pct_ripte:.0f}%) {NumberUtils.format_money(results.ripte_actualizado)} 
--Interés puro del 3% anual desde {input_data.pmi_date.strftime('%d/%m/%Y')} hasta {input_data.final_date.strftime('%d/%m/%Y')} {NumberUtils.format_money(results.interes_puro_3_pct)} 
--SUBTOTAL {NumberUtils.format_money(total_actualizacion)} 

*Tasa de Justicia (2,2%) {NumberUtils.format_money(tasa_justicia)} *
Sobretasa Contribución Caja de Abogados (10% de Tasa) {NumberUtils.format_money(sobretasa_caja)} 

**TOTAL** **{NumberUtils.format_money(total_final)}** 

Importa la presente liquidación la suma de {monto_letras}- 

De la liquidación practicada, traslado a las partes por el plazo de cinco (5) días, bajo apercibimiento de tenerla por consentida (art 59 de la Ley 15.057 - RC 1840/24 SCBA ) Notifíquese.-"""
        
        st.text_area("Liquidación", liquidacion_text, height=500)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Copiar Liquidación", key="copy_liquidacion"):
                st.success("✓ Texto copiado al portapapeles")
        with col2:
            if st.button("🖨️ Ir a Imprimir PDF", key="goto_print"):
                st.info("👉 Use la pestaña 'Imprimir PDF' para generar el documento completo")
    
    with tab4:
        st.subheader("📋 Mínimos de la SRT")
        
        if not st.session_state.data_manager.pisos_data.empty:
            df_pisos = st.session_state.data_manager.pisos_data.copy()
            
            # Formatear fechas
            df_pisos['desde'] = df_pisos['desde'].apply(lambda x: x.strftime('%d/%m/%Y') if isinstance(x, date) else str(x))
            df_pisos['hasta'] = df_pisos['hasta'].apply(lambda x: x.strftime('%d/%m/%Y') if isinstance(x, date) and not pd.isna(x) else 'Vigente')
            df_pisos['piso'] = df_pisos['piso'].apply(lambda x: NumberUtils.format_money(x))
            
            # Crear columna de enlace clicable
            def crear_link_html(enlace):
                enlace_str = str(enlace).strip()
                if enlace_str and enlace_str != '' and enlace_str.lower() != 'nan' and enlace_str.startswith('http'):
                    return f'<a href="{enlace_str}" target="_blank">Ver norma</a>'
                return 'N/A'
            
            # Crear DataFrame para mostrar
            df_display = pd.DataFrame({
                'Norma': df_pisos['resol'],
                'Vigencia Desde': df_pisos['desde'],
                'Vigencia Hasta': df_pisos['hasta'],
                'Monto Mínimo': df_pisos['piso'],
                'Enlace': df_pisos['enlace'].apply(crear_link_html)
            })
            
            # Mostrar tabla con HTML para los links
            st.markdown(
                df_display.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )
            
            st.markdown("---")
            st.caption("💡 Haga clic en 'Ver norma' para acceder al documento oficial")
        else:
            st.warning("No hay datos de pisos disponibles")
    
    with tab5:
        st.subheader("ℹ️ Información del Sistema")
        
        info_tab1, info_tab2, info_tab3 = st.tabs(["Fórmulas", "Fuentes", "Marco Legal"])
        
        with info_tab1:
            st.markdown("""
            ### FÓRMULAS APLICADAS:

            **1. CAPITAL BASE (Ley 24.557):**
            ```
            Capital = IBM × 53 × (% Incapacidad / 100) × (65 / Edad)
            ```
            - Se compara con piso mínimo vigente a la fecha PMI
            - Si el piso es mayor, se aplica el piso proporcional a la incapacidad
            - Se agrega 20% adicional según Art. 3 Ley 26.773 (excepto in itinere)

            **2. ACTUALIZACIÓN RIPTE + 3% **
            - Coeficiente RIPTE = RIPTE Final / RIPTE PMI
            - Capital actualizado = Capital Base × Coeficiente RIPTE
            - Interés puro 3% = Capital Actualizado RIPTE × 0.03 × (días / 365.25)
            - Total = Capital actualizado + Interés puro 3%

            **3. TASA ACTIVA BNA (Art. 12 inc. 2 Ley 24.557):**
            - Se aplica la tasa activa promedio del Banco Nación
            - Cálculo mensual prorrateado por días
            - Suma acumulativa sin capitalización

            **4. INFLACIÓN ACUMULADA:**
            ```
            Inflación = [(1 + r₁/100) × (1 + r₂/100) × ... × (1 + rₙ/100) - 1] × 100
            ```
            
            **CRITERIO DE APLICACIÓN:**
            Se aplica la actualización más favorable entre RIPTE+3% y Tasa Activa.
            La inflación se muestra como referencia comparativa.
            """)
        
        with info_tab2:
            st.markdown("""
            ### FUENTES DE DATOS:         
            Los datos se obtienen de las siguientes fuentes:
            
            1) Las **VARIACIONES DE LA TASA ACTIVA BANCO NACION** 
            de la tabla publicada por el Consejo Prof. de Cs. Ec. 
            [https://trivia.consejo.org.ar/]
            
            2) El **INDICE DE INFLACIÓN** se obtiene de la siguiente manera: 
            desde 2016 en adelante de los datos publicados en **INDEC** - Índice de Precios al Consumidor (IPC)
            [https://www.indec.gob.ar/](https://www.indec.gob.ar/)
            Con anterioridad a 2016 se aplica las tablas de "Inflación Mensual" del 
            **BCRA** - Banco Central - Tasas de referencia
            [https://www.bcra.gob.ar/](https://www.bcra.gob.ar/)

            3) Los indices **RIPTES** se obtienen de
            [https://www.argentina.gob.ar/trabajo/seguridadsocial/ripte/]

            4) Las tablas sobre minimos aplicables de las resoluciones de SRT y MTySS
            [https://www.srt.gob.ar/](https://www.srt.gob.ar/)
            """)
        
        with info_tab3:
            st.markdown("""
            ### MARCO NORMATIVO:

            **LEY 24.557 - RIESGOS DEL TRABAJO:**
            - Art. 14: Fórmula de cálculo de incapacidad permanente parcial
            - Art. 12 inc. 2: Actualización por tasa activa BNA

            **LEY 26.773 - RÉGIMEN DE ORDENAMIENTO LABORAL:**
            - Art. 3: Incremento del 20% sobre prestaciones dinerarias
            - Excepción: No aplica para accidentes in itinere

            **DECRETO 1694/2009:**
            - Actualización de prestaciones según RIPTE
            - Metodología de aplicación del coeficiente
            """)
    with tab6:
        st.subheader("🖨️ Generar PDF del Expediente")
        
        st.markdown("### 📋 Datos de Carátula")
        
        col1, col2 = st.columns(2)
        
        with col1:
            caratula_expediente = st.text_input(
                "Número de Expediente",
                placeholder="Ej: 12345/2023",
                key="pdf_expediente"
            )
            
            caratula_actor = st.text_input(
                "Actor/a",
                placeholder="Apellido, Nombre",
                key="pdf_actor"
            )
            
            caratula_demandado = st.text_input(
                "Demandado/a",
                placeholder="Nombre de la empresa/ART",
                key="pdf_demandado"
            )
        
        with col2:
            caratula_juzgado = st.text_input(
                "Tribunal",
                value="Tribunal de Trabajo",
                key="pdf_juzgado"
            )
            
            caratula_secretaria = st.text_input(
                "Secretaría",
                value="Unica",
                placeholder="Ej: Secretaría Nro. 1",
                key="pdf_secretaria"
            )
            
            caratula_fecha = st.date_input(
                "Fecha del informe",
                value=date.today(),
                format="DD/MM/YYYY",
                key="pdf_fecha"
            )
        
        st.markdown("---")
        
        if st.button("📄 GENERAR VISTA PREVIA PARA IMPRIMIR", use_container_width=True, type="primary", key="generar_pdf"):
            if not caratula_expediente or not caratula_actor:
                st.error("⚠️ Por favor complete al menos el Número de Expediente y el Actor/a")
            else:
                # Determinar método más favorable
                if results.total_ripte_3 >= results.total_tasa_activa:
                    metodo_favorable = "RIPTE + 3%"
                    monto_favorable = results.total_ripte_3
                else:
                    metodo_favorable = "Tasa Activa BNA"
                    monto_favorable = results.total_tasa_activa
                
                # Calcular tasas judiciales para incluir en el PDF
                tasa_justicia = monto_favorable * 0.022
                sobretasa_caja = tasa_justicia * 0.10
                total_final_pdf = monto_favorable + tasa_justicia + sobretasa_caja
                
                mes_pmi = get_mes_nombre(input_data.pmi_date.month)
                anio_pmi = input_data.pmi_date.year
                mes_final = get_mes_nombre(input_data.final_date.month)
                anio_final = input_data.final_date.year
                pct_ripte = (results.ripte_coef - 1) * 100
                
                # Generar HTML para imprimir
                html_content = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="UTF-8">
                    <title>Cálculo Indemnización - {caratula_expediente}</title>
                    <style>
                        @page {{
                            size: A4;
                            margin: 2cm;
                            }}
                        @media print {{
                            body {{
                                width: 21cm;
                                min-height: 29.7cm;
                            }}
                            .no-print {{
                                display: none;
                            }}
                        }}
                        body {{
                            font-family: Arial, sans-serif;
                            line-height: 1.6;
                            color: #333;
                            max-width: 21cm;
                            margin: 0 auto;
                            padding: 20px;
                            background: white;
                        }}
                        .header {{
                            text-align: center;
                            margin-bottom: 30px;
                            border-bottom: 3px solid #2E86AB;
                            padding-bottom: 20px;
                        }}
                        .header h1 {{
                            color: #2E86AB;
                            margin: 0;
                            font-size: 18px;
                        }}
                        .header h2 {{
                            color: #666;
                            margin: 5px 0;
                            font-size: 14px;
                            font-weight: normal;
                        }}
                        .caratula {{
                            background: #f8f9fa;
                            padding: 20px;
                            margin-bottom: 30px;
                            border-left: 4px solid #2E86AB;
                        }}
                        .caratula-row {{
                            margin-bottom: 8px;
                        }}
                        .caratula-label {{
                            font-weight: bold;
                            display: inline-block;
                            width: 150px;
                        }}
                        h3 {{
                            color: #2E86AB;
                            border-bottom: 2px solid #2E86AB;
                            padding-bottom: 8px;
                            margin-top: 30px;
                            font-size: 14px;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 20px 0;
                        }}
                        th, td {{
                            padding: 10px;
                            text-align: left;
                            border: 1px solid #ddd;
                        }}
                        th {{
                            background-color: #2E86AB;
                            color: white;
                            font-weight: bold;
                        }}
                        .highlight {{
                            background-color: #E8F5E8;
                            font-weight: bold;
                        }}
                        .amount {{
                            text-align: right;
                        }}
                        .formula-box {{
                            background: #e7f3ff;
                            border: 1px solid #b3d9ff;
                            padding: 15px;
                            margin: 20px 0;
                            border-radius: 5px;
                        }}
                        .footer {{
                            margin-top: 50px;
                            text-align: center;
                            font-size: 11px;
                            color: #666;
                            border-top: 1px solid #ddd;
                            padding-top: 20px;
                        }}
                        .section {{
                            page-break-inside: avoid;
                        }}
                    </style>
                </head>
                <body>
                    <div class="header">
                        <h1>JUDICIAL</h1>
                        <h2>{caratula_juzgado}</h2>
                    </div>
                    
                    <div class="caratula">
                        <div class="caratula-row">
                            <span class="caratula-label">Expediente:</span>
                            <span>{caratula_expediente}</span>
                        </div>
                        <div class="caratula-row">
                            <span class="caratula-label">Actor/a:</span>
                            <span>{caratula_actor}</span>
                        </div>
                        {f'<div class="caratula-row"><span class="caratula-label">Demandado/a:</span><span>{caratula_demandado}</span></div>' if caratula_demandado else ''}
                        <div class="caratula-row">
                            <span class="caratula-label">Fecha:</span>
                            <span>{caratula_fecha.strftime('%d/%m/%Y')}</span>
                        </div>
                    </div>
                    
                    <h1 style="text-align: center; color: #2E86AB;">CÁLCULO DE INDEMNIZACIÓN - LEY 24.557</h1>
                    
                    <div class="section">
                        <h3>📋 DATOS DEL CASO</h3>
                        <table>
                            <tr>
                                <th>Concepto</th>
                                <th style="text-align: right;">Valor</th>
                            </tr>
                            <tr>
                                <td>Fecha del Siniestro (PMI)</td>
                                <td class="amount">{input_data.pmi_date.strftime('%d/%m/%Y')}</td>
                            </tr>
                            <tr>
                                <td>Fecha de Cálculo</td>
                                <td class="amount">{input_data.final_date.strftime('%d/%m/%Y')}</td>
                            </tr>
                            <tr>
                                <td>Ingreso Base Mensual (IBM)</td>
                                <td class="amount">{NumberUtils.format_money(input_data.ibm)}</td>
                            </tr>
                            <tr>
                                <td>Edad del Trabajador</td>
                                <td class="amount">{input_data.edad} años</td>
                            </tr>
                            <tr>
                                <td>Porcentaje de Incapacidad</td>
                                <td class="amount">{input_data.incapacidad_pct}%</td>
                            </tr>
                            <tr>
                                <td>20% Art. 3 Ley 26.773</td>
                                <td class="amount">{'Sí' if input_data.incluir_20_pct else 'No'}</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h3>💰 RESULTADOS DEL CÁLCULO</h3>
                        <table>
                            <tr>
                                <th>Concepto</th>
                                <th style="text-align: right;">Monto</th>
                            </tr>
                            <tr>
                                <td>Capital Base (Fórmula)</td>
                                <td class="amount">{NumberUtils.format_money(results.capital_formula)}</td>
                            </tr>
                            <tr>
                                <td>{results.piso_info}</td>
                                <td class="amount">{NumberUtils.format_money(results.piso_proporcional) if results.piso_aplicado else '-'}</td>
                            </tr>
                            <tr>
                                <td>20% Art. 3 Ley 26.773</td>
                                <td class="amount">{NumberUtils.format_money(results.adicional_20_pct) if results.adicional_20_pct > 0 else '-'}</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>CAPITAL BASE TOTAL</strong></td>
                                <td class="amount"><strong>{NumberUtils.format_money(results.capital_base)}</strong></td>
                            </tr>
                            <tr>
                                <td>Actualización RIPTE + 3%</td>
                                <td class="amount">{NumberUtils.format_money(results.total_ripte_3)}</td>
                            </tr>
                            <tr>
                                <td>Actualización Tasa Activa BNA</td>
                                <td class="amount">{NumberUtils.format_money(results.total_tasa_activa)}</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>Método más favorable: {metodo_favorable}</strong></td>
                                <td class="amount"><strong>{NumberUtils.format_money(monto_favorable)}</strong></td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h3>📊 DETALLE ACTUALIZACIÓN RIPTE</h3>
                        <div class="formula-box">
                            <p><strong>RIPTE {mes_pmi}/{anio_pmi}:</strong> {results.ripte_pmi:,.2f}</p>
                            <p><strong>RIPTE {mes_final}/{anio_final}:</strong> {results.ripte_final:,.2f}</p>
                            <p><strong>Coeficiente:</strong> {results.ripte_coef:.2f} ({pct_ripte:.0f}%)</p>
                            <p><strong>Capital actualizado RIPTE:</strong> {NumberUtils.format_money(results.ripte_actualizado)}</p>
                            <p><strong>Interés puro 3% anual:</strong> {NumberUtils.format_money(results.interes_puro_3_pct)}</p>
                            <p style="font-size: 16px; margin-top: 10px;"><strong>TOTAL RIPTE + 3%: {NumberUtils.format_money(results.total_ripte_3)}</strong></p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>📈 DETALLE ACTUALIZACIÓN TASA ACTIVA</h3>
                        <div class="formula-box">
                            <p><strong>Tasa Activa BNA acumulada:</strong> {NumberUtils.format_percentage(results.tasa_activa_pct)}</p>
                            <p style="font-size: 16px; margin-top: 10px;"><strong>TOTAL TASA ACTIVA: {NumberUtils.format_money(results.total_tasa_activa)}</strong></p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>📉 INFLACIÓN (REFERENCIA)</h3>
                        <p><strong>Inflación acumulada del período (IPC):</strong> {NumberUtils.format_percentage(results.inflacion_acum_pct)}</p>
                    </div>
                    
                    <div class="section">
                        <h3>🧮 FÓRMULA APLICADA</h3>
                        <div class="formula-box">
                            <p>IBM ({NumberUtils.format_money(input_data.ibm)}) × 53 × 65/edad({input_data.edad}) × Incapacidad ({input_data.incapacidad_pct}%)</p>
                            <p><strong>Capital calculado:</strong> {NumberUtils.format_money(results.capital_formula)}</p>
                            <p style="margin-top: 10px;">{results.piso_info}</p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h3>💼 LIQUIDACIÓN JUDICIAL</h3>
                        <table>
                            <tr>
                                <td>Capital actualizado ({metodo_favorable})</td>
                                <td class="amount">{NumberUtils.format_money(monto_favorable)}</td>
                            </tr>
                            <tr>
                                <td>Tasa de Justicia (2,2%)</td>
                                <td class="amount">{NumberUtils.format_money(tasa_justicia)}</td>
                            </tr>
                            <tr>
                                <td>Sobretasa Contribución Caja de Abogados (10%)</td>
                                <td class="amount">{NumberUtils.format_money(sobretasa_caja)}</td>
                            </tr>
                            <tr class="highlight">
                                <td><strong>TOTAL FINAL</strong></td>
                                <td class="amount"><strong>{NumberUtils.format_money(total_final_pdf)}</strong></td>
                            </tr>
                        </table>
                        <p style="text-align: center; margin-top: 20px;"><strong>{numero_a_letras(total_final_pdf)}</strong></p>
                    </div>
                    
                    <div class="footer">
                        <p><em>Documento generado por Calculadora Indemnizaciones LRT</em></p>
                        <p><em>{caratula_juzgado}</em></p>
                        <p><em>Fecha de generación: {date.today().strftime('%d/%m/%Y')}</em></p>
                    </div>
                </body>
                </html>
                """
                
                st.success("✅ Vista previa generada exitosamente")
                st.info("💡 Presione Ctrl+P (o Cmd+P en Mac) en la vista previa para guardar como PDF")
                
                # Mostrar el HTML en un componente expandible
                with st.expander("👁️ Ver vista previa del documento", expanded=True):
                    st.components.v1.html(html_content, height=800, scrolling=True)
                
                # Botón para abrir en nueva ventana
                html_b64 = base64.b64encode(html_content.encode()).decode()
                href = f'<a href="data:text/html;base64,{html_b64}" download="Calculo_{caratula_expediente.replace("/", "-")}.html" target="_blank"><button style="background-color:#2E86AB; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer; font-weight:bold;">📄 ABRIR EN NUEVA VENTANA PARA IMPRIMIR</button></a>'
                st.markdown(href, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("💡 **Instrucciones:** Después de generar la vista previa, abra en nueva ventana y presione Ctrl+P (o Cmd+P en Mac) para guardar como PDF")

else:
    # Mostrar mensaje inicial
    st.info("👈 Complete los datos en el panel lateral y presione CALCULAR para obtener los resultados")
    
    # Mostrar información general
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Características
        - Cálculo automático según Ley 24.557
        - Actualización por RIPTE + 3%
        - Actualización por Tasa Activa BNA
        - Comparación con inflación (IPC)
        """)
    
    with col2:
        st.markdown("""
        ### ⚖️ Uso judicial
        - Para el apoyo en calculos sentencia
        - Para el calculo en las audiencias.
        - Para apoyo en la liquidación
        - Uso en secretaria y relatoria.
        """)
    
    with col3:
        st.markdown("""
        ### 📄 Documentos
        - Texto para sentencia
        - Liquidación judicial
        - Tabla de mínimos SRT
        - Generación de PDF para imprimir
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Calculadora Indemnizaciones LRT</strong><br>
    Tribunal de Trabajo<br>
    Versión 1.0 de prueba
    Los calculos deben ser verificados manualmente</p>
</div>
""", unsafe_allow_html=True)
