#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CALCULADORA DE INGRESO BASE MENSUAL (IBM)
Ley 24.557 - Art. 12 Inc. 1
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from decimal import Decimal, ROUND_HALF_UP

# Configuración de la página
st.set_page_config(
    page_title="Calculadora IBM - Ley 24557",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personalizado
st.markdown("""
<style>
    button[kind="header"], footer, 
    [data-testid="stHeader"] svg[viewBox="0 0 16 16"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Cargar dataset RIPTE
@st.cache_data
def cargar_ripte():
    """Carga el dataset RIPTE"""
    df = pd.read_csv("dataset_ripte.csv", encoding='latin-1')
    
    # Renombrar columnas por posición (evitar problemas con ñ)
    df.columns = ['anio', 'mes', 'indice_ripte', 'variacion_mensual', 'monto_en_pesos']
    
    # Crear columna de fecha
    df['fecha'] = pd.to_datetime(
        df['anio'].astype(str) + '-' + 
        df['mes'].str[:3].map({
            'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Ago': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'
        }) + '-01'
    )
    return df

def obtener_ripte(df_ripte, año, mes):
    """Obtiene el índice RIPTE para un año y mes"""
    fila = df_ripte[
        (df_ripte['anio'] == año) & 
        (df_ripte['mes'].str.lower().str[:3] == mes.lower()[:3])
    ]
    if not fila.empty:
        return float(fila.iloc[0]['indice_ripte'])
    return None

def calcular_variacion_ripte(df_ripte, año_desde, mes_desde, año_hasta, mes_hasta):
    """Calcula la variación RIPTE entre dos fechas"""
    indice_desde = obtener_ripte(df_ripte, año_desde, mes_desde)
    indice_hasta = obtener_ripte(df_ripte, año_hasta, mes_hasta)
    
    if indice_desde is None or indice_hasta is None or indice_desde == 0:
        return None
    
    return (indice_hasta - indice_desde) / indice_desde

def obtener_meses_anteriores(fecha_pmi, cantidad=12):
    """Obtiene lista de meses anteriores a la PMI"""
    meses = []
    fecha = fecha_pmi
    for i in range(cantidad):
        fecha = fecha - relativedelta(months=1)
        meses.append(fecha)
    meses.reverse()
    return meses

def obtener_nombre_mes(fecha):
    """Obtiene nombre del mes en formato mes-año"""
    meses = ['ene', 'feb', 'mar', 'abr', 'may', 'jun', 
             'jul', 'ago', 'sep', 'oct', 'nov', 'dic']
    return f"{meses[fecha.month-1]}.-{str(fecha.year)[2:]}"

def obtener_dias_mes(año, mes):
    """Obtiene días de un mes"""
    if mes == 12:
        sig_mes = date(año + 1, 1, 1)
    else:
        sig_mes = date(año, mes + 1, 1)
    
    ultimo = sig_mes - relativedelta(days=1)
    return ultimo.day

def formatear_moneda(valor):
    """Formatea como moneda argentina"""
    if valor is None:
        return "$0,00"
    decimal_val = Decimal(str(valor))
    redondeado = decimal_val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    valor_str = f"{redondeado:,.2f}"
    valor_str = valor_str.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"${valor_str}"

def formatear_porcentaje(valor):
    """Formatea como porcentaje"""
    if valor is None:
        return "N/A"
    return f"{valor:.6f}".replace(".", ",")

def generar_texto_plano(datos, fecha_pmi, ibm):
    """Genera texto para copiar a Word"""
    texto = "CÁLCULO DEL INGRESO BASE MENSUAL (IBM)\n"
    texto += "Ley 24.557 - Art. 12 Inc. 1\n"
    texto += "=" * 80 + "\n\n"
    
    texto += f"Fecha PMI: {fecha_pmi.strftime('%d/%m/%Y')}\n\n"
    
    texto += "DETALLE DE SALARIOS ACTUALIZADOS:\n"
    texto += "-" * 80 + "\n"
    texto += f"{'Período':<12} {'Salario':<15} {'RIPTE':<10} {'Variación':<12} {'Actualizado':<15} {'Días':<5}\n"
    texto += "-" * 80 + "\n"
    
    total_orig = Decimal('0')
    total_act = Decimal('0')
    total_dias = 0
    meses_datos = 0
    
    for d in datos:
        if d['incluir'] and d['salario'] > 0:
            total_orig += Decimal(str(d['salario']))
            total_act += Decimal(str(d['salario_act']))
            total_dias += d['dias']
            meses_datos += 1
            
            var = formatear_porcentaje(d['variacion']) if d['variacion'] else "N/A"
            
            texto += f"{d['periodo']:<12} {formatear_moneda(d['salario']):<15} "
            texto += f"{d['ripte']:<10.2f} {var:<12} "
            texto += f"{formatear_moneda(d['salario_act']):<15} {d['dias']:<5}\n"
    
    texto += "-" * 80 + "\n"
    texto += f"{'TOTALES':<12} {formatear_moneda(total_orig):<15} "
    texto += f"{'':10} {'':12} {formatear_moneda(total_act):<15} {total_dias:<5}\n"
    texto += "=" * 80 + "\n\n"
    
    texto += f"Meses con datos: {meses_datos}\n"
    texto += f"Total original: {formatear_moneda(total_orig)}\n"
    texto += f"Total actualizado: {formatear_moneda(total_act)}\n\n"
    
    texto += "=" * 80 + "\n"
    texto += f"INGRESO BASE MENSUAL (IBM): {formatear_moneda(ibm)}\n"
    texto += "=" * 80 + "\n"
    
    return texto

# Cargar datos
try:
    df_ripte = cargar_ripte()
except Exception as e:
    st.error(f"Error al cargar RIPTE: {str(e)}")
    st.stop()

# Título
st.markdown("# 📊 CALCULADORA IBM - LEY 24.557")
st.markdown("### Ingreso Base Mensual - Art. 12 Inc. 1")
st.markdown("---")

# Fecha PMI
col_fecha1, col_fecha2, col_fecha3 = st.columns([1, 2, 1])
with col_fecha2:
    fecha_pmi = st.date_input(
        "📅 Fecha PMI (Primera Manifestación Invalidante)",
        value=date(2021, 12, 1),
        format="DD/MM/YYYY"
    )

st.markdown("---")

# Obtener 12 meses anteriores
meses = obtener_meses_anteriores(fecha_pmi, 12)

# Inicializar session_state
if 'salarios' not in st.session_state:
    st.session_state.salarios = {}

# TABLA DE CÁLCULO
st.subheader("💰 Tabla de Cálculo de Salarios")

# Encabezados de la tabla
col_headers = st.columns([0.5, 1.2, 1.5, 1, 1.2, 1.5, 0.8])
with col_headers[0]:
    st.markdown("**✓**")
with col_headers[1]:
    st.markdown("**Período**")
with col_headers[2]:
    st.markdown("**Salario**")
with col_headers[3]:
    st.markdown("**RIPTE**")
with col_headers[4]:
    st.markdown("**Variación**")
with col_headers[5]:
    st.markdown("**Actualizado**")
with col_headers[6]:
    st.markdown("**Días**")

st.markdown("---")

datos_calc = []

# Filas de la tabla
for mes in meses:
    nombre = obtener_nombre_mes(mes)
    key = f"{mes.year}_{mes.month}"
    
    cols = st.columns([0.5, 1.2, 1.5, 1, 1.2, 1.5, 0.8])
    
    # Checkbox
    with cols[0]:
        incluir = st.checkbox("", value=True, key=f"c_{key}", label_visibility="collapsed")
    
    # Período
    with cols[1]:
        st.text(nombre)
    
    # Input Salario
    with cols[2]:
        salario = st.number_input(
            f"Salario {nombre}",
            min_value=0.0,
            value=0.0,
            step=1000.0,
            format="%.2f",
            key=f"s_{key}",
            label_visibility="collapsed"
        )
    
    # Calcular variación RIPTE
    mes_nombre = nombre.split('.-')[0]
    año_mes = mes.year
    año_pmi = fecha_pmi.year
    mes_pmi = obtener_nombre_mes(fecha_pmi).split('.-')[0]
    
    variacion = calcular_variacion_ripte(df_ripte, año_mes, mes_nombre, año_pmi, mes_pmi)
    
    # Calcular salario actualizado
    if variacion is not None and salario > 0:
        salario_act = salario * (1 + variacion)
    else:
        salario_act = salario
    
    # Obtener RIPTE
    ripte = obtener_ripte(df_ripte, año_mes, mes_nombre)
    dias = obtener_dias_mes(mes.year, mes.month)
    
    # Mostrar RIPTE
    with cols[3]:
        if ripte:
            st.text(f"{ripte:.2f}")
        else:
            st.text("N/A")
    
    # Mostrar Variación
    with cols[4]:
        if variacion is not None:
            st.text(formatear_porcentaje(variacion))
        else:
            st.text("N/A")
    
    # Mostrar Actualizado
    with cols[5]:
        if salario > 0:
            st.text(formatear_moneda(salario_act))
        else:
            st.text("-")
    
    # Mostrar Días
    with cols[6]:
        st.text(str(dias))
    
    datos_calc.append({
        'periodo': nombre,
        'salario': salario,
        'ripte': ripte if ripte else 0,
        'variacion': variacion,
        'salario_act': salario_act,
        'dias': dias,
        'incluir': incluir
    })

# Línea separadora
st.markdown("---")

# TOTALES Y IBM
total_orig = sum(Decimal(str(d['salario'])) for d in datos_calc if d['incluir'] and d['salario'] > 0)
total_act = sum(Decimal(str(d['salario_act'])) for d in datos_calc if d['incluir'] and d['salario'] > 0)
total_dias = sum(d['dias'] for d in datos_calc if d['incluir'] and d['salario'] > 0)
meses_datos = sum(1 for d in datos_calc if d['incluir'] and d['salario'] > 0)

# Calcular IBM
if meses_datos > 0:
    ibm = total_act / Decimal(str(meses_datos))
    ibm = ibm.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
else:
    ibm = Decimal('0')

# Mostrar totales en la tabla
col_tot = st.columns([0.5, 1.2, 1.5, 1, 1.2, 1.5, 0.8])
with col_tot[0]:
    st.markdown("")
with col_tot[1]:
    st.markdown("**TOTALES**")
with col_tot[2]:
    st.markdown(f"**{formatear_moneda(total_orig)}**")
with col_tot[3]:
    st.markdown("")
with col_tot[4]:
    st.markdown("")
with col_tot[5]:
    st.markdown(f"**{formatear_moneda(total_act)}**")
with col_tot[6]:
    st.markdown(f"**{total_dias}**")

st.markdown("---")

# Resultado IBM
col_ibm1, col_ibm2, col_ibm3 = st.columns([1, 2, 1])
with col_ibm2:
    st.success("**INGRESO BASE MENSUAL (IBM)**")
    st.markdown(f"# {formatear_moneda(ibm)}")
    st.caption(f"Promedio de {meses_datos} meses con datos")

st.markdown("---")

# Botón para mostrar texto plano
if st.button("📋 Copiar Cuadro", use_container_width=True, type="primary"):
    texto = generar_texto_plano(datos_calc, fecha_pmi, ibm)
    st.text_area(
        "Texto para copiar a Word",
        value=texto,
        height=400,
        key="texto_plano"
    )

# Información legal
st.markdown("---")
with st.expander("ℹ️ BASE LEGAL - LEY 24.557 ART. 12 INC. 1"):
    st.markdown("""
    ### Artículo 12 inciso 1 - Ley 24.557
    
    *"A los fines del cálculo del valor del ingreso base se considerará el promedio mensual 
    de todos los salarios devengados -de conformidad con lo establecido por el artículo 1° 
    del Convenio N° 95 de la OIT- por el trabajador durante el año anterior a la primera 
    manifestación invalidante, o en el tiempo de prestación de servicio si fuera menor. 
    Los salarios mensuales tomados a fin de establecer el promedio se actualizarán mes a mes 
    aplicándose la variación del índice Remuneraciones Imponibles Promedio de los Trabajadores 
    Estables (RIPTE), elaborado y difundido por el MINISTERIO DE SALUD Y DESARROLLO SOCIAL."*
    
    ### Metodología de Cálculo
    
    1. **Período**: 12 meses anteriores a la PMI (o menor si trabajó menos tiempo)
    2. **Actualización**: Cada salario se actualiza por variación RIPTE desde su mes hasta el mes de la PMI
    3. **Promedio**: El IBM es el promedio de los salarios actualizados
    
    **Fórmula:**
    - Variación RIPTE = (RIPTE PMI - RIPTE Mes) / RIPTE Mes
    - Salario Actualizado = Salario × (1 + Variación RIPTE)
    - IBM = Suma Salarios Actualizados / Cantidad de Meses con Datos
    """)

# Footer
st.markdown("---")
st.caption("**CALCULADORA IBM** | Ley 24.557 Art. 12 Inc. 1 | Actualización RIPTE")