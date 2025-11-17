#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULADORA DE DESPIDOS
Sistema de cálculo de indemnizaciones por despido con actualizaciones
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
import math
import base64
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib.enums import TA_CENTER, TA_RIGHT, TA_LEFT

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Despidos",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
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
    
    /* Info boxes */
    .info-section {
        background-color: #F8F9FA;
        border-left: 4px solid #F18F01;
        padding: 15px;
        border-radius: 8px;
        margin: 20px 0;
    }
    
    .info-section h4 {
        color: #2E86AB;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Función para parsear fechas
def safe_parse_date(s):
    """Función de parseo de fechas"""
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

# Función para días del mes
def days_in_month(d: date) -> int:
    """Días en el mes"""
    if d.month == 12:
        nxt = date(d.year + 1, 1, 1)
    else:
        nxt = date(d.year, d.month + 1, 1)
    return (nxt - date(d.year, d.month, 1)).days

# Cargar datasets
@st.cache_data
def cargar_datasets():
    """Carga los datasets de RIPTE, Tasa e IPC"""
    df_ripte = pd.read_csv("dataset_ripte.csv")
    df_ripte['fecha'] = pd.to_datetime(df_ripte['año'].astype(str) + '-' + 
                                       df_ripte['mes'].str[:3].map({
                                           'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
                                           'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
                                           'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
                                       }) + '-01')
    
    df_tasa = pd.read_csv("dataset_tasa.csv")
    df_tasa['Desde'] = pd.to_datetime(df_tasa['Desde'])
    df_tasa['Hasta'] = pd.to_datetime(df_tasa['Hasta'])
    df_tasa['Valor'] = df_tasa['Valor'].astype(str).str.replace(',', '.').astype(float)
    
    df_ipc = pd.read_csv("dataset_ipc.csv")
    df_ipc['periodo'] = pd.to_datetime(df_ipc['periodo'])
    
    return df_ripte, df_tasa, df_ipc

# Función para calcular antigüedad
def calcular_antiguedad(fecha_ingreso, fecha_despido):
    """Calcula años y meses de antigüedad"""
    años = fecha_despido.year - fecha_ingreso.year
    meses = fecha_despido.month - fecha_ingreso.month
    dias = fecha_despido.day - fecha_ingreso.day
    
    if dias < 0:
        meses -= 1
    
    if meses < 0:
        años -= 1
        meses += 12
    
    # Si los meses son mayor a 3, se considera un año completo adicional
    if meses > 3:
        años += 1
        meses = 0
    
    return años, meses

# Función para calcular días de vacaciones según antigüedad
def calcular_dias_vacaciones(años_antiguedad):
    """Calcula días de vacaciones según LCT 20744"""
    if años_antiguedad < 5:
        return 14
    elif años_antiguedad < 10:
        return 21
    elif años_antiguedad < 20:
        return 28
    else:
        return 35

# Función para actualizar por RIPTE
def actualizar_ripte(monto_base, fecha_inicial, fecha_final, df_ripte):
    """Actualiza un monto por RIPTE + 3%"""
    try:
        if df_ripte.empty:
            return monto_base
        
        fecha_pmi = pd.to_datetime(fecha_inicial)
        fecha_final_date = pd.to_datetime(fecha_final)
        
        # Obtener RIPTE inicial
        ripte_pmi_data = df_ripte[df_ripte['fecha'] <= fecha_pmi]
        if ripte_pmi_data.empty:
            ripte_pmi = float(df_ripte.iloc[0]['indice_ripte'])
        else:
            ripte_pmi = float(ripte_pmi_data.iloc[-1]['indice_ripte'])
        
        # Obtener RIPTE final
        ripte_final_data = df_ripte[df_ripte['fecha'] <= fecha_final_date]
        if ripte_final_data.empty:
            ripte_final = float(df_ripte.iloc[-1]['indice_ripte'])
        else:
            ripte_final = float(ripte_final_data.iloc[-1]['indice_ripte'])
        
        # Calcular coeficiente RIPTE
        coeficiente = ripte_final / ripte_pmi if ripte_pmi > 0 else 1.0
        
        # Aplicar RIPTE
        ripte_actualizado = monto_base * coeficiente
        
        # Aplicar 3% adicional
        interes_puro = ripte_actualizado * 0.03
        
        total_ripte_3 = ripte_actualizado + interes_puro
        
        return total_ripte_3
    except Exception as e:
        st.error(f"Error en cálculo de RIPTE: {str(e)}")
        return monto_base

# Función para actualizar por Tasa Activa
def actualizar_tasa(monto_base, fecha_inicial, fecha_final, df_tasa):
    """Actualiza un monto por Tasa Activa"""
    try:
        if df_tasa.empty:
            return monto_base
        
        fecha_pmi = pd.to_datetime(fecha_inicial)
        fecha_final_date = pd.to_datetime(fecha_final)
        
        # Convertir a date si es Timestamp
        if isinstance(fecha_pmi, pd.Timestamp):
            fecha_pmi = fecha_pmi.date()
        if isinstance(fecha_final_date, pd.Timestamp):
            fecha_final_date = fecha_final_date.date()
        
        total_aporte_pct = 0.0
        
        for _, row in df_tasa.iterrows():
            if "Desde" in df_tasa.columns and not pd.isna(row.get("Desde")):
                fecha_desde = row["Desde"]
            elif "desde" in df_tasa.columns and not pd.isna(row.get("desde")):
                fecha_desde = row["desde"]
            else:
                continue
                
            if "Hasta" in df_tasa.columns and not pd.isna(row.get("Hasta")):
                fecha_hasta = row["Hasta"]
            elif "hasta" in df_tasa.columns and not pd.isna(row.get("hasta")):
                fecha_hasta = row["hasta"]
            else:
                continue
            
            if isinstance(fecha_desde, pd.Timestamp):
                fecha_desde = fecha_desde.date()
            if isinstance(fecha_hasta, pd.Timestamp):
                fecha_hasta = fecha_hasta.date()
            
            inicio_interseccion = max(fecha_pmi, fecha_desde)
            fin_interseccion = min(fecha_final_date, fecha_hasta)
            
            if inicio_interseccion <= fin_interseccion:
                dias_interseccion = (fin_interseccion - inicio_interseccion).days + 1
                
                if "Valor" in df_tasa.columns and not pd.isna(row.get("Valor")):
                    valor_mensual_pct = float(row["Valor"])
                elif "valor" in df_tasa.columns and not pd.isna(row.get("valor")):
                    valor_mensual_pct = float(row["valor"])
                elif "tasa" in df_tasa.columns and not pd.isna(row.get("tasa")):
                    valor_mensual_pct = float(row["tasa"])
                else:
                    continue
                
                aporte_pct = valor_mensual_pct * (dias_interseccion / 30.0)
                total_aporte_pct += aporte_pct
        
        total_actualizado = monto_base * (1.0 + total_aporte_pct / 100.0)
        
        return total_actualizado
    except Exception as e:
        st.error(f"Error en cálculo de tasa: {str(e)}")
        return monto_base

# Función para calcular IPC acumulado
def calcular_ipc_acumulado(fecha_inicial, fecha_final, df_ipc):
    """Calcula el IPC acumulado entre dos fechas"""
    try:
        if df_ipc.empty:
            return 0.0
        
        fecha_pmi = pd.to_datetime(fecha_inicial)
        fecha_final_date = pd.to_datetime(fecha_final)
        
        # Convertir a fecha como objeto date para usar replace
        if isinstance(fecha_pmi, pd.Timestamp):
            fecha_pmi = fecha_pmi.date()
        if isinstance(fecha_final_date, pd.Timestamp):
            fecha_final_date = fecha_final_date.date()
        
        fecha_inicio_mes = pd.Timestamp(fecha_pmi.replace(day=1))
        fecha_final_mes = pd.Timestamp(fecha_final_date.replace(day=1))
        
        ipc_periodo = df_ipc[
            (pd.to_datetime(df_ipc['periodo']) >= fecha_inicio_mes) &
            (pd.to_datetime(df_ipc['periodo']) <= fecha_final_mes)
        ]
        
        if ipc_periodo.empty:
            return 0.0
        
        factor_acumulado = 1.0
        for _, row in ipc_periodo.iterrows():
            variacion = row['variacion_mensual']
            if not pd.isna(variacion):
                factor_acumulado *= (1 + variacion / 100)
        
        inflacion_acumulada = (factor_acumulado - 1) * 100
        return inflacion_acumulada
    except Exception as e:
        st.error(f"Error en cálculo de IPC: {str(e)}")
        return 0.0

# Función para formatear montos
def formato_moneda(valor):
    """Formatea un valor como moneda argentina"""
    return f"$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# Función para generar PDF
def generar_pdf(datos_calculo, datos_actualizacion):
    """Genera un PDF con los resultados"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, 
                           topMargin=2*cm, bottomMargin=2*cm)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        textColor=colors.HexColor('#2E86AB'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    elements.append(Paragraph("LIQUIDACIÓN DE INDEMNIZACIÓN POR DESPIDO", title_style))
    
    # Expediente y carátula si están disponibles
    if datos_calculo.get('nro_expediente') or datos_calculo.get('caratula'):
        expediente_style = ParagraphStyle(
            'Expediente',
            parent=styles['Normal'],
            fontSize=11,
            textColor=colors.HexColor('#666666'),
            spaceAfter=10,
            alignment=TA_CENTER
        )
        
        if datos_calculo.get('nro_expediente'):
            elements.append(Paragraph(f"<b>Expediente Nro:</b> {datos_calculo['nro_expediente']}", expediente_style))
        
        if datos_calculo.get('caratula'):
            elements.append(Paragraph(f"<b>Carátula:</b> {datos_calculo['caratula']}", expediente_style))
    
    elements.append(Spacer(1, 0.5*cm))
    
    # Datos del trabajador
    data_trabajador = [
        ['Fecha de Ingreso:', datos_calculo['fecha_ingreso']],
        ['Fecha de Despido:', datos_calculo['fecha_despido']],
        ['Antigüedad:', f"{datos_calculo['años']} años"],
        ['Salario Mensual Bruto:', formato_moneda(datos_calculo['salario'])],
        ['Preaviso:', datos_calculo['preaviso']],
    ]
    
    t1 = Table(data_trabajador, colWidths=[6*cm, 8*cm])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F5E8')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(t1)
    elements.append(Spacer(1, 0.7*cm))
    
    # Conceptos
    elements.append(Paragraph("DETALLE DE CONCEPTOS", styles['Heading2']))
    elements.append(Spacer(1, 0.3*cm))
    
    data_conceptos = [
        ['Concepto', 'Importe'],
        ['Antigüedad Art. 245', formato_moneda(datos_calculo['antiguedad_245'])],
    ]
    
    if datos_calculo.get('sustitutiva_preaviso', 0) > 0:
        data_conceptos.append(['Sustitutiva de Preaviso', formato_moneda(datos_calculo['sustitutiva_preaviso'])])
        data_conceptos.append(['SAC Preaviso', formato_moneda(datos_calculo['sac_preaviso'])])
    
    data_conceptos.extend([
        ['Días trabajados del Mes', formato_moneda(datos_calculo['dias_trabajados'])],
        ['Integración mes de Despido', formato_moneda(datos_calculo['integracion_mes'])],
        ['SAC Integración mes', formato_moneda(datos_calculo['sac_integracion'])],
        ['SAC Proporcional', formato_moneda(datos_calculo['sac_proporcional'])],
        ['Vacaciones no Gozadas', formato_moneda(datos_calculo['vacaciones'])],
        ['SAC Vacaciones', formato_moneda(datos_calculo['sac_vacaciones'])],
    ])
    
    t2 = Table(data_conceptos, colWidths=[10*cm, 4*cm])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(t2)
    elements.append(Spacer(1, 0.5*cm))
    
    # Total
    data_total = [
        ['INDEMNIZACIÓN TOTAL', formato_moneda(datos_calculo['total'])]
    ]
    
    t3 = Table(data_total, colWidths=[10*cm, 4*cm])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F18F01')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
    ]))
    
    elements.append(t3)
    elements.append(Spacer(1, 0.7*cm))
    
    # Actualizaciones
    elements.append(Paragraph("ACTUALIZACIONES", styles['Heading2']))
    elements.append(Spacer(1, 0.3*cm))
    
    data_act = [
        ['Método', 'Monto Actualizado'],
        ['Actualización RIPTE + 3%', formato_moneda(datos_actualizacion['ripte'])],
        ['Actualización Tasa Activa', formato_moneda(datos_actualizacion['tasa'])],
    ]
    
    t4 = Table(data_act, colWidths=[10*cm, 4*cm])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#28a745')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ]))
    
    elements.append(t4)
    elements.append(Spacer(1, 0.5*cm))
    
    # Nota
    nota_style = ParagraphStyle(
        'Note',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.grey,
        alignment=TA_CENTER
    )
    elements.append(Paragraph("Nota: Los resultados indicados son aproximados.", nota_style))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# Header
st.markdown("""
<div class="main-header">
    <h1>⚖️ CALCULADORA DE DESPIDOS</h1>
    <h2>Sistema de Cálculo de Indemnizaciones Laborales</h2>
</div>
""", unsafe_allow_html=True)

# Cargar datasets
df_ripte, df_tasa, df_ipc = cargar_datasets()

# Formulario de entrada y resultados en dos columnas
col_inputs, col_results = st.columns([1, 1])

with col_inputs:
    st.subheader("📋 Datos del Trabajador")
    
    fecha_ingreso = st.date_input(
        "Fecha de Ingreso",
        value=date(2020, 11, 5),
        min_value=date(1990, 1, 1),
        max_value=date.today(),
        format="DD/MM/YYYY",
        key="fecha_ingreso_input"
    )

    fecha_despido = st.date_input(
        "Fecha de Despido",
        value=date(2025, 11, 16),
        min_value=fecha_ingreso,
        max_value=date.today(),
        format="DD/MM/YYYY",
        key="fecha_despido_input"
    )

    fecha_liquidacion = st.date_input(
        "Fecha de Liquidación",
        value=date.today(),
        min_value=fecha_despido,
        max_value=date.today() + timedelta(days=365),
        format="DD/MM/YYYY",
        key="fecha_liquidacion_input"
    )

    salario = st.number_input(
        "Salario Mensual Bruto ($)",
        min_value=0.0,
        value=150000.0,
        step=1000.0,
        format="%.2f",
        key="salario_input"
    )

    se_pago_preaviso = st.checkbox("¿Se pagó preaviso?", value=False, key="preaviso_checkbox")
    
    calcular_btn = st.button("🧮 CALCULAR INDEMNIZACIÓN", use_container_width=True, type="primary", key="calcular_button")

with col_results:
    if calcular_btn:
        
        # Calcular antigüedad
        años, meses = calcular_antiguedad(fecha_ingreso, fecha_despido)
        
        # Calcular conceptos con Decimal para precisión
        # 1. Antigüedad Art. 245
        antiguedad_245 = Decimal(str(salario)) * Decimal(str(años))
        
        # 2. Sustitutiva de preaviso
        if not se_pago_preaviso:
            if años < 5:
                sustitutiva_preaviso = Decimal(str(salario)) * Decimal('1')
            else:
                sustitutiva_preaviso = Decimal(str(salario)) * Decimal('2')
            sac_preaviso = sustitutiva_preaviso / Decimal('12')
        else:
            sustitutiva_preaviso = Decimal('0')
            sac_preaviso = Decimal('0')
        
        # 3. Días trabajados del mes
        dias_mes = days_in_month(fecha_despido)
        dias_trabajados_mes = fecha_despido.day
        dias_trabajados = (Decimal(str(salario)) / Decimal(str(dias_mes))) * Decimal(str(dias_trabajados_mes))
        
        # 4. Integración mes de despido
        if fecha_despido.day == dias_mes:
            integracion_mes = Decimal('0')
            sac_integracion = Decimal('0')
        else:
            dias_integracion = dias_mes - dias_trabajados_mes
            integracion_mes = (Decimal(str(salario)) / Decimal(str(dias_mes))) * Decimal(str(dias_integracion))
            sac_integracion = integracion_mes / Decimal('12')
        
        # 5. SAC Proporcional
        if fecha_despido.month <= 6:
            dias_desde_sac = (fecha_despido - date(fecha_despido.year, 1, 1)).days
        else:
            dias_desde_sac = (fecha_despido - date(fecha_despido.year, 7, 1)).days
        
        sac_proporcional = (Decimal(str(salario)) / Decimal('365')) * Decimal(str(dias_desde_sac))
        
        # 6. Vacaciones no gozadas
        dias_vacaciones = calcular_dias_vacaciones(años)
        valor_dia_vacaciones = Decimal(str(salario)) / Decimal('25')
        vacaciones = valor_dia_vacaciones * Decimal(str(dias_vacaciones))
        sac_vacaciones = vacaciones / Decimal('12')
        
        # Total - redondear cada concepto a 2 decimales
        antiguedad_245 = antiguedad_245.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        sustitutiva_preaviso = sustitutiva_preaviso.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        sac_preaviso = sac_preaviso.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        dias_trabajados = dias_trabajados.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        integracion_mes = integracion_mes.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        sac_integracion = sac_integracion.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        sac_proporcional = sac_proporcional.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        vacaciones = vacaciones.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        sac_vacaciones = sac_vacaciones.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        
        total = (antiguedad_245 + sustitutiva_preaviso + sac_preaviso + 
                 dias_trabajados + integracion_mes + sac_integracion + 
                 sac_proporcional + vacaciones + sac_vacaciones)
        
        # Guardar datos en session_state
        st.session_state.datos_calculo = {
            'fecha_ingreso': fecha_ingreso.strftime("%d/%m/%Y"),
            'fecha_despido': fecha_despido.strftime("%d/%m/%Y"),
            'fecha_liquidacion': fecha_liquidacion.strftime("%d/%m/%Y"),
            'años': años,
            'meses': meses,
            'salario': float(salario),
            'preaviso': 'Se pagó' if se_pago_preaviso else 'Sin preaviso',
            'antiguedad_245': float(antiguedad_245),
            'sustitutiva_preaviso': float(sustitutiva_preaviso),
            'sac_preaviso': float(sac_preaviso),
            'dias_trabajados': float(dias_trabajados),
            'integracion_mes': float(integracion_mes),
            'sac_integracion': float(sac_integracion),
            'sac_proporcional': float(sac_proporcional),
            'vacaciones': float(vacaciones),
            'sac_vacaciones': float(sac_vacaciones),
            'total': float(total)
        }
        
        # Calcular actualizaciones
        total_float = st.session_state.datos_calculo['total']
        
        actualizado_ripte = actualizar_ripte(total_float, fecha_despido, fecha_liquidacion, df_ripte)
        actualizado_tasa = actualizar_tasa(total_float, fecha_despido, fecha_liquidacion, df_tasa)
        ipc_acumulado = calcular_ipc_acumulado(fecha_despido, fecha_liquidacion, df_ipc)
        
        st.session_state.datos_actualizacion = {
            'ripte': actualizado_ripte,
            'tasa': actualizado_tasa,
            'ipc': ipc_acumulado
        }

# Mostrar resultados si existen
if 'datos_calculo' in st.session_state:
    with col_results:
        st.subheader("💰 Liquidación")
        
        datos = st.session_state.datos_calculo
        
        # Mostrar conceptos de manera compacta
        st.write(f"**Antigüedad Art. 245:** {formato_moneda(datos['antiguedad_245'])}")
        
        if datos['sustitutiva_preaviso'] > 0:
            st.write(f"**Sustitutiva de Preaviso:** {formato_moneda(datos['sustitutiva_preaviso'])}")
            st.write(f"**SAC Preaviso:** {formato_moneda(datos['sac_preaviso'])}")
        
        st.write(f"**Días trabajados del Mes:** {formato_moneda(datos['dias_trabajados'])}")
        
        if datos['integracion_mes'] > 0:
            st.write(f"**Integración mes de Despido:** {formato_moneda(datos['integracion_mes'])}")
            st.write(f"**SAC Integración:** {formato_moneda(datos['sac_integracion'])}")
        
        st.write(f"**SAC Proporcional:** {formato_moneda(datos['sac_proporcional'])}")
        st.write(f"**Vacaciones no Gozadas:** {formato_moneda(datos['vacaciones'])}")
        st.write(f"**SAC Vacaciones:** {formato_moneda(datos['sac_vacaciones'])}")
        
        st.markdown(f"""
        <div style="background-color: #F18F01; padding: 8px 15px; border-radius: 5px; text-align: center; margin-top: 15px; display: inline-block; width: 100%;">
            <span style="color: white; font-weight: bold; font-size: 14px; margin-right: 10px;">INDEMNIZACIÓN TOTAL:</span>
            <span style="font-size: 18px; font-weight: bold; color: white;">{formato_moneda(datos['total'])}</span>
        </div>
        """, unsafe_allow_html=True)

# Actualizaciones centradas debajo
if 'datos_actualizacion' in st.session_state:
    st.markdown("---")
    st.subheader(f"📈 ACTUALIZACIONES AL {st.session_state.datos_calculo['fecha_liquidacion']}")
    
    col_act1, col_act2, col_act3 = st.columns(3)
    
    datos_act = st.session_state.datos_actualizacion
    
    with col_act1:
        st.markdown(f"""
        <div class="result-card highlight-ripte">
            <h3>RIPTE + 3%</h3>
            <div class="result-amount">{formato_moneda(datos_act['ripte'])}</div>
            <div class="result-detail">Desde {st.session_state.datos_calculo['fecha_despido']} hasta {st.session_state.datos_calculo['fecha_liquidacion']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_act2:
        st.markdown(f"""
        <div class="result-card highlight-tasa">
            <h3>Tasa Activa</h3>
            <div class="result-amount">{formato_moneda(datos_act['tasa'])}</div>
            <div class="result-detail">Desde {st.session_state.datos_calculo['fecha_despido']} hasta {st.session_state.datos_calculo['fecha_liquidacion']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_act3:
        st.markdown(f"""
        <div class="result-card" style="border-left-color: #A23B72;">
            <h3>IPC (Ref.)</h3>
            <div class="result-amount">{datos_act['ipc']:.2f}%</div>
            <div class="result-detail">Variación inflacionaria del período</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Últimos datos disponibles
    st.markdown("---")
    
    # Obtener últimos datos disponibles
    ultimo_ripte_txt = ""
    ultimo_ipc_txt = ""
    ultima_tasa_txt = ""
    
    # RIPTE
    if not df_ripte.empty:
        ultimo_ripte = df_ripte.iloc[-1]
        fecha_ripte = ultimo_ripte['fecha']
        valor_ripte = ultimo_ripte['indice_ripte']
        if pd.notnull(fecha_ripte):
            if isinstance(fecha_ripte, pd.Timestamp):
                mes_ripte = fecha_ripte.month
                año_ripte = fecha_ripte.year
            else:
                mes_ripte = fecha_ripte.month
                año_ripte = fecha_ripte.year
            ultimo_ripte_txt = f"RIPTE {mes_ripte}/{año_ripte}: {valor_ripte:,.0f}"
    
    # IPC
    if not df_ipc.empty:
        ultimo_ipc = df_ipc.iloc[-1]
        fecha_ipc = ultimo_ipc['periodo']
        variacion_ipc = ultimo_ipc['variacion_mensual']
        if pd.notnull(fecha_ipc):
            if isinstance(fecha_ipc, pd.Timestamp):
                mes_ipc = fecha_ipc.month
                año_ipc = fecha_ipc.year
            else:
                fecha_ipc = pd.to_datetime(fecha_ipc)
                mes_ipc = fecha_ipc.month
                año_ipc = fecha_ipc.year
            ultimo_ipc_txt = f"IPC {mes_ipc}/{año_ipc}: {variacion_ipc:.2f}%"
    
    # TASA ACTIVA - tomar el primer registro (dataset ordenado descendente)
    if not df_tasa.empty:
        ultima_tasa = df_tasa.iloc[0]
        valor_tasa = ultima_tasa['Valor']
        fecha_hasta = ultima_tasa['Hasta']
        if pd.notnull(fecha_hasta):
            if isinstance(fecha_hasta, pd.Timestamp):
                fecha_txt = fecha_hasta.strftime("%d/%m/%Y")
            else:
                fecha_txt = pd.to_datetime(fecha_hasta).strftime("%d/%m/%Y")
            ultima_tasa_txt = f"TASA ACTIVA {fecha_txt}: {valor_tasa:.2f}%"
    
    # Mostrar cuadro
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
        {ultima_tasa_txt}
    </div>
    """, unsafe_allow_html=True)
    
    # Botón de PDF
    st.markdown("---")
    
    if st.button("📄 IMPRIMIR PDF", use_container_width=True, key="generar_pdf_button"):
        st.session_state.mostrar_campos_pdf = True
    
    # Mostrar campos solo si se presionó el botón
    if st.session_state.get('mostrar_campos_pdf', False):
        st.subheader("📋 Datos para el PDF")
        
        col_exp1, col_exp2 = st.columns(2)
        
        with col_exp1:
            nro_expediente = st.text_input("Nro. de Expediente", key="nro_expediente_pdf")
        
        with col_exp2:
            caratula = st.text_input("Carátula", key="caratula_pdf")
        
        # Generar PDF directamente
        st.session_state.datos_calculo['nro_expediente'] = nro_expediente
        st.session_state.datos_calculo['caratula'] = caratula
        
        pdf_buffer = generar_pdf(st.session_state.datos_calculo, st.session_state.datos_actualizacion)
        
        st.download_button(
            label="📥 DESCARGAR PDF",
            data=pdf_buffer,
            file_name=f"liquidacion_despido_{st.session_state.datos_calculo['fecha_despido'].replace('/', '')}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_pdf_button"
        )

# Información sobre cálculos (fuera del flujo principal)
st.markdown("---")

# Información sobre cálculos
with st.expander("ℹ️ INFORMACIÓN SOBRE CÁLCULOS Y FUENTES"):
    st.markdown("""
    ### 📘 Marco Legal - Ley 20.744 (LCT)
    
    **Antigüedad (Art. 245):** Se calcula 1 mes de salario por cada año de servicio o fracción mayor a 3 meses.
    
    **Sustitutiva de Preaviso:** 
    - Antigüedad menor a 5 años: 1 mes de salario
    - Antigüedad mayor a 4 años: 2 meses de salario
    
    **SAC Preaviso:** Doceava parte de la sustitutiva de preaviso.
    
    **Días Trabajados:** Se divide el salario por la cantidad de días del mes y se multiplica por los días trabajados durante el mes de despido.
    
    **Integración Mes de Despido:** Corresponde a los días que restan para completar el mes. No se paga si el despido coincide con el último día del mes.
    
    **SAC Integración:** Doceava parte de la integración del mes.
    
    **SAC Proporcional:** Se calcula proporcionalmente desde el último aguinaldo pagado (enero o julio) hasta la fecha de despido.
    
    **Vacaciones no Gozadas:** Según antigüedad:
    - Menos de 5 años: 14 días corridos
    - De 5 a 10 años: 21 días corridos
    - De 10 a 20 años: 28 días corridos
    - Más de 20 años: 35 días corridos
    
    **SAC Vacaciones:** Doceava parte del valor de las vacaciones.
    
    ### 📊 Métodos de Actualización
    
    **RIPTE + 3%:** Remuneración Imponible Promedio de los Trabajadores Estables, más un 3% adicional.
    - **Fuente:** Secretaría de Seguridad Social - Ministerio de Trabajo
    
    **Tasa Activa:** Tasa activa promedio del Banco Nación
    - **Fuente:** Banco de la Nación Argentina
    
    **IPC (Referencia):** Índice de Precios al Consumidor
    - **Fuente:** INDEC - Instituto Nacional de Estadística y Censos
    
    ### 🔢 Detalles Técnicos
    
    Todos los cálculos se realizan utilizando precisión decimal para garantizar exactitud legal.
    Los redondeos se aplican según normas contables argentinas (Resoluciones Técnicas 17 y 41).
    
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>CALCULADORA DE DESPIDOS</strong></p>
    <p>Sistema de Cálculo de Indemnizaciones Laborales</p>
    <p style="font-size: 12px;">Los resultados son aproximados y no constituyen asesoramiento legal.</p>
</div>
""", unsafe_allow_html=True)