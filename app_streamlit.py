#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULADORA INDEMNIZACIONES LEY 24.557
Sistema de cálculo de indemnizaciones laborales
VERSIÓN CON REDONDEO CONTABLE A 2 DECIMALES
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
from decimal import Decimal, ROUND_HALF_UP

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

def redondear(valor):
    """
    Redondea a 2 decimales según criterio contable/judicial estándar.
    Usa redondeo aritmético (0.5 siempre hacia arriba).
    """
    if isinstance(valor, Decimal):
        return valor.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    return Decimal(str(valor)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

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
        "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
        "%Y/%m/%d", "%m/%d/%Y", "%d.%m.%Y",
        "%Y.%m.%d", "%d-%b-%Y", "%d-%B-%Y"
    ]
    
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    
    try:
        dt = pd.to_datetime(s, errors='coerce')
        if pd.notna(dt):
            return dt.date()
    except:
        pass
    
    return None

def days_in_month(fecha):
    """Retorna la cantidad de días del mes"""
    if fecha.month == 12:
        sig = date(fecha.year + 1, 1, 1)
    else:
        sig = date(fecha.year, fecha.month + 1, 1)
    return (sig - date(fecha.year, fecha.month, 1)).days

def numero_a_letras(numero):
    """Convierte un número a su representación en letras (español)"""
    
    def convertir_grupo(n):
        """Convierte un número de 1 a 999 en texto"""
        unidades = ['', 'UN', 'DOS', 'TRES', 'CUATRO', 'CINCO', 'SEIS', 'SIETE', 'OCHO', 'NUEVE']
        decenas = ['', '', 'VEINTE', 'TREINTA', 'CUARENTA', 'CINCUENTA', 'SESENTA', 'SETENTA', 'OCHENTA', 'NOVENTA']
        especiales = ['DIEZ', 'ONCE', 'DOCE', 'TRECE', 'CATORCE', 'QUINCE', 'DIECISEIS', 'DIECISIETE', 'DIECIOCHO', 'DIECINUEVE']
        centenas = ['', 'CIENTO', 'DOSCIENTOS', 'TRESCIENTOS', 'CUATROCIENTOS', 'QUINIENTOS', 'SEISCIENTOS', 'SETECIENTOS', 'OCHOCIENTOS', 'NOVECIENTOS']
        
        if n == 0:
            return ''
        elif n < 10:
            return unidades[n]
        elif n < 20:
            return especiales[n - 10]
        elif n < 100:
            d = n // 10
            u = n % 10
            if u == 0:
                return decenas[d]
            elif d == 2:
                return 'VEINTI' + unidades[u]
            else:
                return decenas[d] + ' Y ' + unidades[u]
        else:
            c = n // 100
            resto = n % 100
            texto_c = 'CIEN' if (c == 1 and resto == 0) else centenas[c]
            if resto == 0:
                return texto_c
            else:
                return texto_c + ' ' + convertir_grupo(resto)
    
    entero = int(numero)
    decimal = int(round((numero - entero) * 100))
    
    if entero == 0:
        texto = 'CERO'
    elif entero >= 1000000000:
        miles_mill = entero // 1000000000
        resto = entero % 1000000000
        texto = (convertir_grupo(miles_mill) if miles_mill > 1 else 'UN') + ' MIL MILLONES'
        if resto > 0:
            if resto >= 1000000:
                millones = resto // 1000000
                resto = resto % 1000000
                texto += ' ' + (convertir_grupo(millones) if millones > 1 else 'UN') + ' MILLÓN' + ('ES' if millones > 1 else '')
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
            self.tasa_data = self.tasa_data.dropna(subset=["desde"]).sort_values("desde").reset_index(drop=True)
        else:
            fecha_col = None
            for c in cols:
                if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                    fecha_col = c
                    break
            if fecha_col is None:
                fecha_col = cols[0]
            
            self.tasa_data["fecha"] = self.tasa_data[fecha_col].apply(safe_parse_date)
            self.tasa_data = self.tasa_data.dropna(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)

        val_col = None
        for c in cols:
            if ("tasa" in c) or ("valor" in c):
                val_col = c
                break
        if val_col is None:
            num_cols = self.tasa_data.select_dtypes(include="number").columns.tolist()
            val_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        if val_col not in self.tasa_data.columns:
            return
        
        self.tasa_data["tasa"] = pd.to_numeric(self.tasa_data[val_col], errors="coerce")

    def _norm_ipc(self):
        """Normalización IPC"""
        if self.ipc_data.empty:
            return
        cols = [c.lower() for c in self.ipc_data.columns]
        self.ipc_data.columns = cols
        
        fecha_col = None
        for c in cols:
            if ("fecha" in c) or ("periodo" in c) or ("mes" in c):
                fecha_col = c
                break
        if fecha_col is None:
            fecha_col = cols[0]
        
        val_col = None
        for c in cols:
            if ("ipc" in c) or ("inflacion" in c) or ("valor" in c) or ("indice" in c):
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
        
        fecha_col = None
        for c in cols:
            if ("fecha" in c) or ("desde" in c) or ("vigencia" in c):
                fecha_col = c
                break
        if fecha_col is None:
            fecha_col = cols[0]
        
        self.pisos_data["fecha"] = self.pisos_data[fecha_col].apply(safe_parse_date)
        
        monto_col = None
        for c in cols:
            if ("monto" in c) or ("piso" in c) or ("minimo" in c) or ("valor" in c):
                monto_col = c
                break
        if monto_col is None:
            num_cols = self.pisos_data.select_dtypes(include="number").columns.tolist()
            monto_col = num_cols[0] if num_cols else cols[1] if len(cols)>1 else cols[0]
        
        self.pisos_data["monto"] = pd.to_numeric(self.pisos_data[monto_col], errors="coerce")
        
        norma_col = None
        for c in cols:
            if ("norma" in c) or ("resolucion" in c) or ("res" in c):
                norma_col = c
                break
        
        if norma_col:
            self.pisos_data["norma"] = self.pisos_data[norma_col].astype(str)
        else:
            self.pisos_data["norma"] = "SRT"
        
        self.pisos_data = self.pisos_data.dropna(subset=["fecha", "monto"]).sort_values("fecha").reset_index(drop=True)

    def get_piso_minimo(self, fecha_pmi: date) -> Tuple[Optional[float], str]:
        """Obtiene el piso mínimo vigente"""
        if self.pisos_data.empty:
            return None, "No disponible"
        
        pisos_aplicables = self.pisos_data[self.pisos_data['fecha'] <= fecha_pmi]
        if pisos_aplicables.empty:
            return None, "No disponible"
        
        piso_vigente = pisos_aplicables.iloc[-1]
        return float(piso_vigente['monto']), str(piso_vigente['norma'])
    
    def get_ripte_coeficiente(self, fecha_pmi: date, fecha_final: date) -> Tuple[float, float, float]:
        """Calcula el coeficiente RIPTE"""
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
        """Cálculo de tasa activa CON REDONDEO A 2 DECIMALES"""
        if self.tasa_data.empty:
            return 0.0, capital_base
            
        # Usar Decimal para acumulación precisa
        total_aporte_pct = Decimal('0.0')
        
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
                
                # REDONDEO: Cada aporte parcial se redondea a 2 decimales
                aporte_pct = redondear(Decimal(str(valor_mensual_pct)) * (Decimal(str(dias_interseccion)) / Decimal('30.0')))
                total_aporte_pct = redondear(total_aporte_pct + aporte_pct)
        
        # REDONDEO: Total actualizado redondeado a 2 decimales
        capital_base_dec = Decimal(str(capital_base))
        total_actualizado = redondear(capital_base_dec * (Decimal('1.0') + total_aporte_pct / Decimal('100.0')))
        
        return float(total_aporte_pct), float(total_actualizado)
    
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
    """Motor de cálculos CON REDONDEO A 2 DECIMALES"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
    
    def calcular_indemnizacion(self, input_data: InputData) -> Results:
        """Realiza todos los cálculos CON REDONDEO EN CADA ETAPA"""
        
        # Cálculo de capital por fórmula
        capital_formula = self._calcular_capital_formula(input_data)
        
        # Aplicación de piso mínimo
        piso_minimo, piso_norma = self.data_manager.get_piso_minimo(input_data.pmi_date)
        capital_aplicado, piso_aplicado, piso_info, piso_proporcional = self._aplicar_piso_minimo(
            capital_formula, piso_minimo, piso_norma, input_data.incapacidad_pct
        )
        
        # REDONDEO: Adicional 20% redondeado a 2 decimales
        if input_data.incluir_20_pct:
            adicional_20_pct = float(redondear(Decimal(str(capital_aplicado)) * Decimal('0.20')))
        else:
            adicional_20_pct = 0.0
        
        # REDONDEO: Capital base redondeado a 2 decimales
        capital_base = float(redondear(Decimal(str(capital_aplicado)) + Decimal(str(adicional_20_pct))))
        
        # Cálculo RIPTE
        ripte_coef, ripte_pmi, ripte_final = self.data_manager.get_ripte_coeficiente(
            input_data.pmi_date, input_data.final_date
        )
        
        # REDONDEO: RIPTE actualizado redondeado a 2 decimales
        ripte_actualizado = float(redondear(Decimal(str(capital_base)) * Decimal(str(ripte_coef))))
        
        # REDONDEO: Interés puro 3% anual redondeado a 2 decimales
        dias_transcurridos = (input_data.final_date - input_data.pmi_date).days
        factor_dias = Decimal(str(dias_transcurridos)) / Decimal('365.0')
        interes_puro_3_pct = float(redondear(Decimal(str(ripte_actualizado)) * Decimal('0.03') * factor_dias))
        
        # REDONDEO: Total RIPTE + 3% redondeado a 2 decimales
        total_ripte_3 = float(redondear(Decimal(str(ripte_actualizado)) + Decimal(str(interes_puro_3_pct))))
        
        # Cálculo tasa activa (ya incluye redondeo interno)
        tasa_activa_pct, total_tasa_activa = self.data_manager.calcular_tasa_activa(
            input_data.pmi_date, input_data.final_date, capital_base
        )
        
        # Cálculo inflación (referencia)
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
        """Calcula capital según fórmula CON REDONDEO"""
        # REDONDEO: Resultado de la fórmula redondeado a 2 decimales
        capital = Decimal(str(input_data.ibm)) * Decimal('53') * (Decimal('65') / Decimal(str(input_data.edad))) * (Decimal(str(input_data.incapacidad_pct)) / Decimal('100'))
        return float(redondear(capital))
    
    def _aplicar_piso_minimo(self, capital_formula: float, piso_minimo: Optional[float], 
                           piso_norma: str, incapacidad_pct: float) -> Tuple[float, bool, str, float]:
        """Aplica piso mínimo si corresponde CON REDONDEO"""
        if piso_minimo is None:
            return capital_formula, False, "No se encontró piso mínimo para la fecha", 0.0
        
        # REDONDEO: Piso proporcional redondeado a 2 decimales
        piso_proporcional = float(redondear(Decimal(str(piso_minimo)) * (Decimal(str(incapacidad_pct)) / Decimal('100'))))
        
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
            # Método más favorable
            metodo_favorable = "RIPTE + 3%" if results.total_ripte_3 >= results.total_tasa_activa else "Tasa Activa BNA"
            monto_favorable = max(results.total_ripte_3, results.total_tasa_activa)
            
            st.markdown(f"""
            <div class="result-card" style="border-left-color: #F18F01;">
                <h3>MÉTODO MÁS FAVORABLE</h3>
                <div class="result-amount">{NumberUtils.format_money(monto_favorable)}</div>
                <div class="result-detail">
                    Método aplicable: {metodo_favorable}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tasa Activa
            highlight_class = "highlight-tasa" if results.total_tasa_activa > results.total_ripte_3 else ""
            st.markdown(f"""
            <div class="result-card {highlight_class}">
                <h3>ACTUALIZACIÓN TASA ACTIVA BNA</h3>
                <div class="result-amount">{NumberUtils.format_money(results.total_tasa_activa)}</div>
                <div class="result-detail">
                    Tasa acumulada: {NumberUtils.format_percentage(results.tasa_activa_pct)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Datos adicionales en fila inferior
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.markdown(f"""
            <div class="result-card">
                <h3>📊 RIPTE</h3>
                <div class="result-detail">
                    <strong>RIPTE PMI:</strong> {results.ripte_pmi:,.2f}<br>
                    <strong>RIPTE Final:</strong> {results.ripte_final:,.2f}<br>
                    <strong>Coeficiente:</strong> {results.ripte_coef:.6f}<br>
                    <strong>Variación:</strong> {NumberUtils.format_percentage((results.ripte_coef - 1) * 100)}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="result-card">
                <h3>💵 INFLACIÓN (IPC)</h3>
                <div class="result-detail">
                    <strong>Inflación acumulada:</strong><br>
                    {NumberUtils.format_percentage(results.inflacion_acum_pct)}<br>
                    <em style="font-size: 12px; color: #999;">Período: {input_data.pmi_date.strftime('%m/%Y')} - {input_data.final_date.strftime('%m/%Y')}</em>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            # Mostrar fecha del último dato disponible en cada dataset
            ultimo_ripte = data_mgr.ripte_data.iloc[-1] if not data_mgr.ripte_data.empty else None
            ultimo_tasa = data_mgr.tasa_data.iloc[-1] if not data_mgr.tasa_data.empty else None
            
            if ultimo_ripte is not None:
                fecha_ripte = ultimo_ripte['fecha']
                mes_ripte = get_mes_nombre(fecha_ripte.month)
                anio_ripte = fecha_ripte.year
            else:
                mes_ripte = "N/D"
                anio_ripte = ""
            
            if ultimo_tasa is not None:
                if 'hasta' in data_mgr.tasa_data.columns and not pd.isna(ultimo_tasa.get('hasta')):
                    fecha_tasa = ultimo_tasa['hasta']
                elif 'desde' in data_mgr.tasa_data.columns and not pd.isna(ultimo_tasa.get('desde')):
                    fecha_tasa = ultimo_tasa['desde']
                else:
                    fecha_tasa = ultimo_tasa.get('fecha', None)
                
                if fecha_tasa:
                    mes_tasa = get_mes_nombre(fecha_tasa.month)
                    anio_tasa = fecha_tasa.year
                else:
                    mes_tasa = "N/D"
                    anio_tasa = ""
            else:
                mes_tasa = "N/D"
                anio_tasa = ""
            
            st.markdown(f"""
            <div class="result-card">
                <h3>📅 ÚLTIMOS DATOS DISPONIBLES</h3>
                <div class="result-detail">
                    <strong>RIPTE:</strong> {mes_ripte} {anio_ripte}<br>
                    <strong>Tasa Activa:</strong> {mes_tasa} {anio_tasa}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("### 📄 Texto para Sentencia")
        
        metodo_favorable = "RIPTE + 3%" if results.total_ripte_3 >= results.total_tasa_activa else "Tasa Activa BNA"
        monto_favorable = max(results.total_ripte_3, results.total_tasa_activa)
        
        texto_sentencia = f"""
**CONSIDERANDOS:**

Que en fecha {input_data.pmi_date.strftime('%d/%m/%Y')} el trabajador sufrió un accidente laboral/enfermedad profesional que determinó una incapacidad del {input_data.incapacidad_pct}%.

Que corresponde calcular la indemnización conforme lo establecido en la Ley 24.557 y sus modificatorias, considerando un Ingreso Base Mensual (IBM) de {NumberUtils.format_money(input_data.ibm)}.

Que aplicando la fórmula legal [IBM × 53 × (65/edad) × porcentaje de incapacidad], resulta un capital de {NumberUtils.format_money(results.capital_formula)}.

{"Que corresponde aplicar el piso mínimo establecido por " + results.piso_norma + ", siendo el monto proporcional a la incapacidad de " + NumberUtils.format_money(results.piso_proporcional) + "." if results.piso_aplicado else ""}

{"Que conforme el artículo 3 de la Ley 26.773, corresponde adicionar un 20% sobre el capital base, ascendiendo a " + NumberUtils.format_money(results.adicional_20_pct) + "." if results.adicional_20_pct > 0 else ""}

Que el capital base indemnizatorio asciende a {NumberUtils.format_money(results.capital_base)}.

Que corresponde actualizar dicho capital desde la fecha del siniestro ({input_data.pmi_date.strftime('%d/%m/%Y')}) hasta la fecha de esta liquidación ({input_data.final_date.strftime('%d/%m/%Y')}).

Que comparando los métodos de actualización:
- RIPTE + 3% anual: {NumberUtils.format_money(results.total_ripte_3)}
- Tasa Activa BNA: {NumberUtils.format_money(results.total_tasa_activa)}

Que el método más favorable resulta ser **{metodo_favorable}**, ascendiendo el capital actualizado a {NumberUtils.format_money(monto_favorable)}.

**POR ELLO:**

**SE RESUELVE:**

I. HACER LUGAR a la demanda y CONDENAR a la demandada al pago de la suma de **{NumberUtils.format_money(monto_favorable)}** ({numero_a_letras(monto_favorable)}) en concepto de indemnización por incapacidad laboral conforme Ley 24.557, con más los intereses que correspondan según la tasa activa del Banco de la Nación Argentina desde la fecha de esta sentencia hasta su efectivo pago.

II. REGULAR los honorarios profesionales...

III. COSTAS a cargo de la demandada vencida.

REGÍSTRESE, NOTIFÍQUESE y OPORTUNAMENTE ARCHÍVESE.
        """
        
        st.text_area("", texto_sentencia, height=600)
        
        if st.button("📋 Copiar al portapapeles"):
            st.code(texto_sentencia, language=None)
            st.success("✓ Texto copiado (puede seleccionar todo con Ctrl+A)")
    
    with tab3:
        st.markdown("### 💰 Liquidación Judicial")
        
        metodo_favorable = "RIPTE + 3%" if results.total_ripte_3 >= results.total_tasa_activa else "Tasa Activa BNA"
        monto_favorable = max(results.total_ripte_3, results.total_tasa_activa)
        
        tasa_justicia = monto_favorable * 0.022
        sobretasa_caja = tasa_justicia * 0.10
        total_final = monto_favorable + tasa_justicia + sobretasa_caja
        
        liquidacion_data = {
            'Concepto': [
                'Capital actualizado (' + metodo_favorable + ')',
                'Tasa de Justicia (2,2%)',
                'Sobretasa Contribución Caja de Abogados (10%)',
                'TOTAL A ABONAR'
            ],
            'Importe': [
                NumberUtils.format_money(monto_favorable),
                NumberUtils.format_money(tasa_justicia),
                NumberUtils.format_money(sobretasa_caja),
                NumberUtils.format_money(total_final)
            ]
        }
        
        df_liquidacion = pd.DataFrame(liquidacion_data)
        st.table(df_liquidacion)
        
        st.markdown(f"""
        **Total en letras:** {numero_a_letras(total_final)}
        """)
        
        st.markdown("---")
        st.markdown("### 📋 Detalle del Cálculo")
        
        detalle_data = {
            'Ítem': [
                'Ingreso Base Mensual (IBM)',
                'Edad del trabajador',
                'Porcentaje de incapacidad',
                'Capital según fórmula',
                'Piso mínimo aplicable',
                '20% art. 3 Ley 26.773',
                'Capital base',
                'Fecha PMI',
                'Fecha de cálculo',
                'Días transcurridos',
                'Método de actualización',
                'Capital actualizado'
            ],
            'Valor': [
                NumberUtils.format_money(input_data.ibm),
                f"{input_data.edad} años",
                f"{input_data.incapacidad_pct}%",
                NumberUtils.format_money(results.capital_formula),
                NumberUtils.format_money(results.piso_proporcional) if results.piso_aplicado else "No aplica",
                NumberUtils.format_money(results.adicional_20_pct) if results.adicional_20_pct > 0 else "No aplica",
                NumberUtils.format_money(results.capital_base),
                input_data.pmi_date.strftime('%d/%m/%Y'),
                input_data.final_date.strftime('%d/%m/%Y'),
                f"{(input_data.final_date - input_data.pmi_date).days} días",
                metodo_favorable,
                NumberUtils.format_money(monto_favorable)
            ]
        }
        
        df_detalle = pd.DataFrame(detalle_data)
        st.table(df_detalle)
    
    with tab4:
        st.markdown("### 📋 Tabla de Mínimos SRT")
        
        if not data_mgr.pisos_data.empty:
            # Preparar datos para mostrar
            pisos_display = data_mgr.pisos_data.copy()
            pisos_display['fecha'] = pisos_display['fecha'].apply(lambda x: x.strftime('%d/%m/%Y') if pd.notna(x) else '')
            pisos_display['monto'] = pisos_display['monto'].apply(lambda x: NumberUtils.format_money(x) if pd.notna(x) else '')
            
            # Renombrar columnas para mejor visualización
            pisos_display = pisos_display.rename(columns={
                'fecha': 'Fecha Vigencia',
                'monto': 'Monto',
                'norma': 'Norma'
            })
            
            # Seleccionar solo las columnas relevantes
            columnas_mostrar = ['Fecha Vigencia', 'Monto', 'Norma']
            pisos_display = pisos_display[columnas_mostrar]
            
            st.dataframe(pisos_display, use_container_width=True, hide_index=True)
            
            # Destacar el piso aplicable
            if results.piso_monto > 0:
                st.info(f"**Piso aplicable para la fecha {input_data.pmi_date.strftime('%d/%m/%Y')}:** {NumberUtils.format_money(results.piso_monto)} ({results.piso_norma})")
                st.info(f"**Piso proporcional a {input_data.incapacidad_pct}%:** {NumberUtils.format_money(results.piso_proporcional)}")
        else:
            st.warning("No hay datos de pisos mínimos disponibles")
    
    with tab5:
        st.markdown("### ℹ️ Información sobre el Cálculo")
        
        col_info1, col_info2 = st.columns(2)
        
        with col_info1:
            st.markdown("""
            #### 📊 Método RIPTE + 3%
            
            Este método actualiza el capital base mediante:
            1. **Coeficiente RIPTE**: Relación entre RIPTE final y RIPTE del PMI
            2. **Interés puro 3%**: Aplicado anualmente sobre el capital actualizado por RIPTE
            
            **Fórmula:**
            ```
            Capital actualizado = Capital base × Coef. RIPTE
            Interés 3% = Capital actualizado × 3% × (días/365)
            Total = Capital actualizado + Interés 3%
            ```
            
            **Ventajas:**
            - Refleja la evolución salarial
            - Tasa de interés moderada
            - Jurisprudencia consolidada
            """)
        
        with col_info2:
            st.markdown("""
            #### 📈 Método Tasa Activa BNA
            
            Este método aplica la tasa activa promedio del Banco de la Nación Argentina.
            
            **Características:**
            - Se aplica mensualmente
            - Contempla variaciones de mercado
            - Puede ser más favorable en contextos de alta inflación
            
            **Cálculo:**
            - Se acumula la tasa de cada mes del período
            - Se aplica proporcionalmente a los días de cada mes
            """)
        
        st.markdown("---")
        
        st.markdown("""
        #### ⚖️ Normativa Aplicable
        
        - **Ley 24.557**: Sistema de Riesgos del Trabajo
        - **Ley 26.773 art. 3**: Adicional del 20%
        - **Decreto 1694/09**: Pisos mínimos indemnizatorios
        - **Resoluciones SRT**: Actualización de pisos mínimos
        
        #### 🧮 Fórmula de Cálculo (art. 14 inc. 2.a Ley 24.557)
        
        ```
        IBM × 53 × (65 / edad del trabajador) × % de incapacidad
        ```
        
        Donde:
        - **IBM**: Ingreso Base Mensual
        - **53**: Cantidad de meses (coeficiente legal)
        - **65**: Edad de retiro
        - **% incapacidad**: Porcentaje determinado por la comisión médica
        
        #### 💡 Redondeo Contable
        
        **Versión actual:** Esta calculadora aplica redondeo a 2 decimales en cada operación intermedia,
        siguiendo el estándar contable y judicial argentino (redondeo aritmético: 0.5 hacia arriba).
        
        Esto garantiza:
        - Reproducibilidad de los cálculos
        - Conformidad con normas contables
        - Trazabilidad en peritajes judiciales
        """)
    
    with tab6:
        st.markdown("### 🖨️ Generar Documento para Imprimir como PDF")
        
        st.info("""
        **Instrucciones:**
        1. Complete los datos de carátula a continuación
        2. Presione el botón "Generar Vista Previa"
        3. Se abrirá el documento en una nueva ventana
        4. Use la función de impresión del navegador (Ctrl+P o Cmd+P)
        5. Seleccione "Guardar como PDF" como impresora
        """)
        
        st.markdown("---")
        st.markdown("#### 📋 Datos de la Carátula")
        
        col_caratula1, col_caratula2 = st.columns(2)
        
        with col_caratula1:
            caratula_juzgado = st.text_input(
                "Juzgado/Tribunal",
                value="Cámara de Apelaciones del Trabajo"
            )
            
            caratula_expediente = st.text_input(
                "Expediente N°",
                value="12345/2024"
            )
            
            caratula_actor = st.text_input(
                "Actor/a",
                value=""
            )
        
        with col_caratula2:
            caratula_demandado = st.text_input(
                "Demandado/a",
                value=""
            )
            
            caratula_fecha = st.date_input(
                "Fecha del documento",
                value=date.today(),
                format="DD/MM/YYYY"
            )
        
        if st.button("📄 GENERAR VISTA PREVIA", type="primary", use_container_width=True):
            if not caratula_actor:
                st.error("⚠️ Debe completar el nombre del Actor/a")
            else:
                metodo_favorable = "RIPTE + 3%" if results.total_ripte_3 >= results.total_tasa_activa else "Tasa Activa BNA"
                monto_favorable = max(results.total_ripte_3, results.total_tasa_activa)
                
                tasa_justicia = monto_favorable * 0.022
                sobretasa_caja = tasa_justicia * 0.10
                total_final_pdf = monto_favorable + tasa_justicia + sobretasa_caja
                
                mes_pmi = get_mes_nombre(input_data.pmi_date.month)
                anio_pmi = input_data.pmi_date.year
                mes_final = get_mes_nombre(input_data.final_date.month)
                anio_final = input_data.final_date.year
                
                pct_ripte = (results.ripte_coef - 1) * 100
                
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
                        body {{
                            font-family: 'Times New Roman', serif;
                            font-size: 12pt;
                            line-height: 1.6;
                            color: #000;
                            max-width: 800px;
                            margin: 0 auto;
                            padding: 20px;
                        }}
                        h1 {{
                            text-align: center;
                            font-size: 18pt;
                            margin: 30px 0;
                            color: #2E86AB;
                        }}
                        h3 {{
                            color: #2E86AB;
                            font-size: 14pt;
                            margin-top: 25px;
                            margin-bottom: 15px;
                            border-bottom: 2px solid #2E86AB;
                            padding-bottom: 5px;
                        }}
                        table {{
                            width: 100%;
                            border-collapse: collapse;
                            margin: 15px 0;
                        }}
                        th, td {{
                            border: 1px solid #333;
                            padding: 8px;
                            text-align: left;
                        }}
                        th {{
                            background-color: #f0f0f0;
                            font-weight: bold;
                        }}
                        .amount {{
                            text-align: right;
                            font-family: 'Courier New', monospace;
                        }}
                        .highlight {{
                            background-color: #E8F5E8;
                            font-weight: bold;
                        }}
                        .section {{
                            margin: 25px 0;
                            page-break-inside: avoid;
                        }}
                        .formula-box {{
                            background-color: #f5f5f5;
                            border: 1px solid #ddd;
                            padding: 15px;
                            margin: 15px 0;
                            font-family: 'Courier New', monospace;
                        }}
                        .caratula {{
                            border: 2px solid #2E86AB;
                            padding: 15px;
                            margin-bottom: 30px;
                            background-color: #f9f9f9;
                        }}
                        .caratula-row {{
                            margin: 8px 0;
                            display: flex;
                            justify-content: space-between;
                        }}
                        .caratula-label {{
                            font-weight: bold;
                            min-width: 120px;
                        }}
                        .footer {{
                            margin-top: 40px;
                            text-align: center;
                            font-size: 10pt;
                            color: #666;
                            border-top: 1px solid #ccc;
                            padding-top: 20px;
                        }}
                        @media print {{
                            body {{
                                padding: 0;
                            }}
                            .no-print {{
                                display: none;
                            }}
                        }}
                    </style>
                </head>
                <body>
                    <div class="caratula">
                        <div class="caratula-row">
                            <span class="caratula-label">Juzgado:</span>
                            <span>{caratula_juzgado}</span>
                        </div>
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
    Versión 1.0 con redondeo contable a 2 decimales<br>
    Los calculos deben ser verificados manualmente</p>
</div>
""", unsafe_allow_html=True)
