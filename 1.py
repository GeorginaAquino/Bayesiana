import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from faker import Faker

# Configuración de la página
st.set_page_config(
    page_title="Paradoja del Falso Positivo - VIH",
    page_icon="🧬",
    layout="wide"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .formula-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 8px;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Encabezado principal
st.markdown("""
<div class="main-header">
    <h1>🧬 Paradoja del Falso Positivo</h1>
    <h3>Aplicación del Teorema de Bayes en Pruebas de VIH</h3>
    <p>Demostración de por qué más del 95% de tests positivos pueden ser falsos</p>
</div>
""", unsafe_allow_html=True)

# Sidebar con controles
st.sidebar.header("⚙️ Parámetros de la Prueba")
st.sidebar.markdown("---")

# Prevalencia
st.sidebar.markdown("### 📊 Prevalencia")
prevalence = st.sidebar.slider(
    "P(A) - Probabilidad de tener VIH (%)",
    min_value=0.01,
    max_value=10.0,
    value=0.1,
    step=0.01,
    format="%.2f"
)
st.sidebar.caption(f"🔢 En fracción: {prevalence/100:.6f}")

st.sidebar.markdown("---")

# Sensibilidad
st.sidebar.markdown("### 🎯 Sensibilidad")
sensitivity = st.sidebar.slider(
    "P(B|A) - Test positivo si tiene VIH (%)",
    min_value=50,
    max_value=100,
    value=95,
    step=1
)
st.sidebar.caption("Capacidad de detectar enfermos")

st.sidebar.markdown("---")

# Especificidad
st.sidebar.markdown("### ✅ Especificidad")
specificity = st.sidebar.slider(
    "P(Bᶜ|Aᶜ) - Test negativo si NO tiene VIH (%)",
    min_value=50,
    max_value=100,
    value=98,
    step=1
)
st.sidebar.caption("Capacidad de identificar sanos")

st.sidebar.markdown("---")

# Tamaño de población
population_size = st.sidebar.number_input(
    "👥 Tamaño de población",
    min_value=10,
    max_value=1000000,
    value=100000,
    step=1000
)

# Convertir a proporciones
prob_a = prevalence / 100  # P(A)
prob_b_given_a = sensitivity / 100  # P(B|A)
prob_bc_given_ac = specificity / 100  # P(Bᶜ|Aᶜ)
prob_b_given_ac = 1 - prob_bc_given_ac  # P(B|Aᶜ)
prob_ac = 1 - prob_a  # P(Aᶜ)

# ==================== CÁLCULOS BAYESIANOS ====================
st.header("📐 Cálculos del Teorema de Bayes")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📋 Datos del Problema")
    st.markdown(f"""
    **Definiciones:**
    - **A** = "la persona tiene VIH"
    - **Aᶜ** = "la persona NO tiene VIH"
    - **B** = "la prueba resulta positiva"
    - **Bᶜ** = "la prueba resulta negativa"
    
    **Probabilidades conocidas:**
    - P(A) = {prob_a:.6f} = {prevalence}% (Prevalencia)
    - P(Aᶜ) = {prob_ac:.6f} = {100-prevalence:.2f}%
    - P(B|A) = {prob_b_given_a:.2f} = {sensitivity}% (Sensibilidad)
    - P(Bᶜ|Aᶜ) = {prob_bc_given_ac:.2f} = {specificity}% (Especificidad)
    - P(B|Aᶜ) = {prob_b_given_ac:.2f} = {100-specificity}% (1 - Especificidad)
    """)

with col2:
    st.markdown("### 🎯 Objetivo")
    st.info("""
    **Queremos calcular: P(A|B)**
    
    ¿Cuál es la probabilidad de que una persona **realmente tenga VIH** 
    dado que su **prueba resultó positiva**?
    
    Esta es la pregunta clave que responde el Teorema de Bayes.
    """)

# Fórmula del Teorema de Bayes
st.markdown("### 🧮 Teorema de Bayes")

st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|A^c) \cdot P(A^c)}")

# Cálculo paso a paso
st.markdown("### 📊 Cálculo Paso a Paso")

numerador = prob_b_given_a * prob_a
denominador_parte1 = prob_b_given_a * prob_a
denominador_parte2 = prob_b_given_ac * prob_ac
denominador = denominador_parte1 + denominador_parte2
prob_a_given_b = numerador / denominador if denominador > 0 else 0

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 1️⃣ Numerador")
    st.code(f"P(B|A) × P(A)\n= {prob_b_given_a:.4f} × {prob_a:.6f}\n= {numerador:.8f}")

with col2:
    st.markdown("#### 2️⃣ Denominador")
    st.code(f"P(B|A)×P(A) + P(B|Aᶜ)×P(Aᶜ)\n= {denominador_parte1:.8f}\n+ {denominador_parte2:.8f}\n= {denominador:.8f}")

with col3:
    st.markdown("#### 3️⃣ Resultado")
    st.code(f"P(A|B) = {numerador:.8f}/{denominador:.8f}\n\n= {prob_a_given_b:.6f}")

# Resultado destacado
st.markdown("---")
st.markdown("## 🎯 RESULTADO FINAL")

percentage_result = prob_a_given_b * 100
false_positive_rate = 100 - percentage_result

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 3rem; 
                border-radius: 20px; 
                text-align: center;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);">
        <h2 style="color: white; margin: 0; font-size: 1.5rem;">P(A|B)</h2>
        <h1 style="color: white; font-size: 4rem; margin: 1rem 0; font-weight: bold;">
            {percentage_result:.2f}%
        </h1>
        <p style="color: white; font-size: 1.2rem; margin: 0;">
            Probabilidad de tener VIH<br>dado que el test es positivo
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Interpretación de la paradoja
col1, col2 = st.columns(2)

with col1:
    st.error(f"""
    ### ⚠️ ¡LA PARADOJA!
    
    Aunque la prueba tiene:
    - ✅ **{sensitivity}%** de sensibilidad
    - ✅ **{specificity}%** de especificidad
    
    Si tu test resulta **positivo**, solo tienes:
    - 🎯 **{percentage_result:.2f}%** de probabilidad de tener VIH
    - ❌ **{false_positive_rate:.2f}%** de probabilidad de ser un falso positivo
    
    **¡Más del {false_positive_rate:.0f}% de los tests positivos son FALSOS!**
    """)

with col2:
    st.success(f"""
    ### ✅ Interpretación
    
    De cada **{denominador/prob_a:.0f} personas** con test positivo:
    - Solo **1 persona** realmente tiene VIH
    - Las otras **{(denominador/prob_a)-1:.0f} personas** están sanas
    
    Esto ocurre porque:
    1. La prevalencia es muy baja ({prevalence}%)
    2. Hay muchas más personas sanas que enfermas
    3. Incluso un {100-specificity}% de falsos positivos en población sana genera muchos casos
    """)

# ==================== SIMULACIÓN DE POBLACIÓN ====================
st.header("👥 Simulación de Población")

# Calcular números absolutos
n_enfermos = int(population_size * prob_a)
n_sanos = population_size - n_enfermos

# Resultados de tests
vp = int(n_enfermos * prob_b_given_a)  # Verdaderos Positivos
fn = n_enfermos - vp  # Falsos Negativos
vn = int(n_sanos * prob_bc_given_ac)  # Verdaderos Negativos
fp = n_sanos - vn  # Falsos Positivos

total_positivos = vp + fp
total_negativos = fn + vn

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="👥 Población Total",
        value=f"{population_size:,}",
        help="Tamaño total de la población"
    )

with col2:
    st.metric(
        label="🔴 Con VIH (A)",
        value=f"{n_enfermos:,}",
        delta=f"{(n_enfermos/population_size)*100:.3f}%",
        help="Personas que realmente tienen VIH"
    )

with col3:
    st.metric(
        label="🟢 Sin VIH (Aᶜ)",
        value=f"{n_sanos:,}",
        delta=f"{(n_sanos/population_size)*100:.2f}%",
        help="Personas que NO tienen VIH"
    )

with col4:
    st.metric(
        label="🧪 Tests Positivos",
        value=f"{total_positivos:,}",
        delta=f"{(total_positivos/population_size)*100:.2f}%",
        help="Total de pruebas con resultado positivo"
    )

st.markdown("---")

# Matriz de confusión
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📊 Matriz de Confusión")
    
    confusion_df = pd.DataFrame({
        'Condición Real': ['Tiene VIH (A)', 'NO tiene VIH (Aᶜ)', 'TOTAL'],
        'Test Positivo (B)': [
            f"{vp:,} (VP)",
            f"{fp:,} (FP)",
            f"{total_positivos:,}"
        ],
        'Test Negativo (Bᶜ)': [
            f"{fn:,} (FN)",
            f"{vn:,} (VN)",
            f"{total_negativos:,}"
        ],
        'TOTAL': [
            f"{n_enfermos:,}",
            f"{n_sanos:,}",
            f"{population_size:,}"
        ]
    })
    
    st.dataframe(confusion_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### 📌 Leyenda")
    st.markdown("""
    - **VP**: Verdaderos Positivos
    - **FP**: Falsos Positivos
    - **VN**: Verdaderos Negativos
    - **FN**: Falsos Negativos
    
    **Métricas:**
    - Sensibilidad = VP/(VP+FN)
    - Especificidad = VN/(VN+FP)
    - **VPP = VP/(VP+FP)** ⭐
    - VPN = VN/(VN+FN)
    """)

# ==================== VISUALIZACIONES ====================
st.header("📊 Visualizaciones")

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Tests Positivos", 
    "📊 Matriz Confusión", 
    "📈 Métricas",
    "🔄 Comparación"
])

with tab1:
    st.markdown("### 🥧 Composición de Tests Positivos")
    st.markdown(f"**De los {total_positivos:,} tests positivos:**")
    
    # Gráfico de torta
    fig_pie = go.Figure(data=[go.Pie(
        labels=[
            f'Verdaderos Positivos<br>{vp:,} personas',
            f'Falsos Positivos<br>{fp:,} personas'
        ],
        values=[vp, fp],
        marker_colors=['#22c55e', '#ef4444'],
        hole=0.4,
        textinfo='label+percent',
        textfont_size=14
    )])
    
    fig_pie.update_layout(
        title={
            'text': f'Total: {total_positivos:,} tests positivos',
            'x': 0.5,
            'xanchor': 'center'
        },
        height=500,
        showlegend=True,
        annotations=[{
            'text': f'VPP<br>{percentage_result:.1f}%',
            'x': 0.5,
            'y': 0.5,
            'font_size': 20,
            'showarrow': False
        }]
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"""
        ✅ **Verdaderos Positivos: {vp:,}**
        - {(vp/total_positivos)*100:.2f}% de los positivos
        - Realmente tienen VIH
        """)
    
    with col2:
        st.error(f"""
        ❌ **Falsos Positivos: {fp:,}**
        - {(fp/total_positivos)*100:.2f}% de los positivos
        - NO tienen VIH (error de la prueba)
        """)

with tab2:
    st.markdown("### 📊 Gráfico de Barras - Matriz de Confusión")
    
    fig_bar = go.Figure()
    
    fig_bar.add_trace(go.Bar(
        name='Test Positivo',
        x=['Tiene VIH', 'NO tiene VIH'],
        y=[vp, fp],
        text=[f'VP: {vp:,}', f'FP: {fp:,}'],
        textposition='auto',
        marker_color=['#22c55e', '#ef4444'],
        hovertemplate='<b>%{x}</b><br>Test Positivo: %{y:,}<extra></extra>'
    ))
    
    fig_bar.add_trace(go.Bar(
        name='Test Negativo',
        x=['Tiene VIH', 'NO tiene VIH'],
        y=[fn, vn],
        text=[f'FN: {fn:,}', f'VN: {vn:,}'],
        textposition='auto',
        marker_color=['#f59e0b', '#3b82f6'],
        hovertemplate='<b>%{x}</b><br>Test Negativo: %{y:,}<extra></extra>'
    ))
    
    fig_bar.update_layout(
        title='Resultados de Pruebas por Condición Real',
        xaxis_title='Condición Real del Paciente',
        yaxis_title='Número de Personas',
        barmode='group',
        height=500,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.markdown("### 📈 Comparación de Métricas del Test")
    
    # Calcular todas las métricas
    vpn = (vn / total_negativos * 100) if total_negativos > 0 else 0
    
    metrics_data = pd.DataFrame({
        'Métrica': [
            'Sensibilidad',
            'Especificidad', 
            'VPP (P(A|B))',
            'VPN'
        ],
        'Valor': [sensitivity, specificity, percentage_result, vpn],
        'Descripción': [
            'Detecta enfermos correctamente',
            'Identifica sanos correctamente',
            'Prob. enfermedad si test +',
            'Prob. salud si test -'
        ],
        'Tipo': [
            'Característica del Test',
            'Característica del Test',
            'Valor Predictivo',
            'Valor Predictivo'
        ]
    })
    
    fig_metrics = px.bar(
        metrics_data,
        x='Métrica',
        y='Valor',
        color='Tipo',
        text='Valor',
        color_discrete_map={
            'Característica del Test': '#3b82f6',
            'Valor Predictivo': '#8b5cf6'
        },
        hover_data=['Descripción']
    )
    
    fig_metrics.update_traces(
        texttemplate='%{text:.2f}%',
        textposition='outside'
    )
    
    fig_metrics.update_layout(
        title='Comparación de Todas las Métricas',
        yaxis_title='Porcentaje (%)',
        height=500,
        yaxis_range=[0, 105]
    )
    
    st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Tabla de métricas
    st.markdown("#### 📋 Tabla de Métricas Detallada")
    metrics_detail = pd.DataFrame({
        'Métrica': ['Sensibilidad', 'Especificidad', 'VPP', 'VPN'],
        'Fórmula': [
            'VP / (VP + FN)',
            'VN / (VN + FP)',
            'VP / (VP + FP)',
            'VN / (VN + FN)'
        ],
        'Cálculo': [
            f'{vp:,} / ({vp:,} + {fn:,})',
            f'{vn:,} / ({vn:,} + {fp:,})',
            f'{vp:,} / ({vp:,} + {fp:,})',
            f'{vn:,} / ({vn:,} + {fn:,})'
        ],
        'Valor (%)': [
            f'{sensitivity:.2f}',
            f'{specificity:.2f}',
            f'{percentage_result:.2f}',
            f'{vpn:.2f}'
        ]
    })
    
    st.dataframe(metrics_detail, use_container_width=True, hide_index=True)

with tab4:
    st.markdown("### 🔄 Comparación: Características del Test vs Valores Predictivos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        #### 📋 Características del Test
        (Intrínsecas al test, NO dependen de prevalencia)
        """)
        
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=sensitivity,
            title={'text': "Sensibilidad<br>P(B|A)"},
            delta={'reference': 90},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#22c55e"},
                'steps': [
                    {'range': [0, 70], 'color': "#fee2e2"},
                    {'range': [70, 90], 'color': "#fef3c7"},
                    {'range': [90, 100], 'color': "#d1fae5"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 95
                }
            }
        ))
        fig1.update_layout(height=300)
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=specificity,
            title={'text': "Especificidad<br>P(Bᶜ|Aᶜ)"},
            delta={'reference': 95},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#3b82f6"},
                'steps': [
                    {'range': [0, 70], 'color': "#fee2e2"},
                    {'range': [70, 90], 'color': "#fef3c7"},
                    {'range': [90, 100], 'color': "#dbeafe"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 98
                }
            }
        ))
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.warning("""
        #### 🎯 Valores Predictivos
        (SÍ dependen de la prevalencia)
        """)
        
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=percentage_result,
            title={'text': "VPP<br>P(A|B)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#8b5cf6"},
                'steps': [
                    {'range': [0, 30], 'color': "#fee2e2"},
                    {'range': [30, 70], 'color': "#fef3c7"},
                    {'range': [70, 100], 'color': "#e9d5ff"}
                ]
            }
        ))
        fig3.update_layout(height=300)
        st.plotly_chart(fig3, use_container_width=True)
        
        fig4 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=vpn,
            title={'text': "VPN<br>P(Aᶜ|Bᶜ)"},
            delta={'reference': 99},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "#f59e0b"},
                'steps': [
                    {'range': [0, 70], 'color': "#fee2e2"},
                    {'range': [70, 90], 'color': "#fef3c7"},
                    {'range': [90, 100], 'color': "#fef3c7"}
                ]
            }
        ))
        fig4.update_layout(height=300)
        st.plotly_chart(fig4, use_container_width=True)

# ==================== EXPLICACIÓN DE LA PARADOJA ====================
st.header("💡 ¿Por qué ocurre esta Paradoja?")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🔑 Factores Clave
    
    La paradoja ocurre debido a tres factores:
    """)
    
    st.info(f"""
    **1. Baja Prevalencia ({prevalence}%)**
    - Solo {n_enfermos:,} de {population_size:,} tienen VIH
    - Ratio: 1 enfermo por cada {int(1/(prob_a)):.0f} personas
    """)
    
    st.success(f"""
    **2. Gran Población Sana**
    - {n_sanos:,} personas NO tienen VIH
    - Son el {(n_sanos/population_size)*100:.2f}% de la población
    """)
    
    st.error(f"""
    **3. Acumulación de Falsos Positivos**
    - {100-specificity}% de {n_sanos:,} = **{fp:,} falsos positivos**
    - {sensitivity}% de {n_enfermos:,} = **{vp:,} verdaderos positivos**
    - **Ratio: {fp/vp if vp > 0 else 0:.1f} falsos por cada verdadero**
    """)

with col2:
    st.markdown("""
    ### 📊 Visualización del Problema
    """)
    
    # Crear visualización de proporción
    fig_sankey = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=[
                f"Población<br>{population_size:,}",
                f"Con VIH<br>{n_enfermos:,}",
                f"Sin VIH<br>{n_sanos:,}",
                f"Test +<br>{total_positivos:,}",
                f"Test -<br>{total_negativos:,}"
            ],
            color=["#94a3b8", "#ef4444", "#22c55e", "#f59e0b", "#3b82f6"]
        ),
        link=dict(
            source=[0, 0, 1, 1, 2, 2],
            target=[1, 2, 3, 4, 3, 4],
            value=[n_enfermos, n_sanos, vp, fn, fp, vn],
            color=["#fecaca", "#bbf7d0", "#22c55e", "#fbbf24", "#ef4444", "#3b82f6"]
        )
    )])
    
    fig_sankey.update_layout(
        title="Flujo: Población → Condición Real → Resultado Test",
        height=400,
        font_size=12
    )
    
    st.plotly_chart(fig_sankey, use_container_width=True)
    
    st.markdown(f"""
    **Observa cómo:**
    - Los {fp:,} falsos positivos (rojo) dominan
    - Los {vp:,} verdaderos positivos (verde) son minoría
    - Por eso VPP = {percentage_result:.2f}%
    """)

# Conclusión
st.markdown("---")
st.markdown("### 🎓 Conclusión")

col1, col2 = st.columns(2)

with col1:
    st.success("""
    #### ✅ Lecciones Aprendidas
    
    1. **La prevalencia importa más de lo que parece**
       - Afecta dramáticamente los valores predictivos
       - Tests buenos pueden tener VPP bajos en poblaciones de baja prevalencia
    
    2. **No confundir características con valores predictivos**
       - Sensibilidad/Especificidad ≠ VPP/VPN
       - Las primeras son del test, las segundas dependen de la población
    
    3. **Tests negativos son más confiables (en baja prevalencia)**
       - VPN suele ser muy alto: {vpn:.2f}%
       - Buenos para descartar enfermedad
    """)

with col2:
    st.warning("""
    #### ⚠️ Implicaciones Prácticas
    
    1. **Tests de screening**
       - Requieren confirmación con otros tests
       - Un positivo NO es diagnóstico definitivo
    
    2. **Comunicación con pacientes**
       - Explicar probabilidades reales
       - Evitar alarmas innecesarias
    
    3. **Política de salud pública**
       - Considerar prevalencia al diseñar programas
       - Balance costo-beneficio de testing masivo
    """)

# ==================== GENERACIÓN DE DATOS ====================
st.header("🎲 Generación de Dataset Simulado")

st.info("""
Esta sección genera datos sintéticos usando **Python Faker** que siguen las probabilidades calculadas.
Útil para análisis posteriores, visualizaciones o reportes.
""")

col1, col2 = st.columns([1, 1])

with col1:
    n_samples = st.number_input(
        "Número de registros a generar:",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100
    )

with col2:
    generar = st.button("🎲 Generar Dataset", type="primary", use_container_width=True)

if generar:
    with st.spinner('Generando datos simulados con Faker...'):
        fake = Faker('es_ES')  # Faker en español
        np.random.seed(42)  # Para reproducibilidad
        
        # Generar datos
        data = []
        
        for i in range(n_samples):
            # Determinar si tiene VIH basado en prevalencia
            tiene_vih = np.random.random() < prob_a
            
            # Determinar resultado del test
            if tiene_vih:
                # Si tiene VIH, aplicar sensibilidad
                test_positivo = np.random.random() < prob_b_given_a
            else:
                # Si NO tiene VIH, aplicar complemento de especificidad
                test_positivo = np.random.random() < prob_b_given_ac
            
            # Clasificar resultado
            if tiene_vih and test_positivo:
                clasificacion = 'Verdadero Positivo (VP)'
                estado = '✅ VP'
            elif tiene_vih and not test_positivo:
                clasificacion = 'Falso Negativo (FN)'
                estado = '⚠️ FN'
            elif not tiene_vih and test_positivo:
                clasificacion = 'Falso Positivo (FP)'
                estado = '❌ FP'
            else:
                clasificacion = 'Verdadero Negativo (VN)'
                estado = '✅ VN'
            
            # Crear registro
            data.append({
                'ID': f'PAC-{i+1:05d}',
                'Nombre': fake.name(),
                'Edad': np.random.randint(18, 75),
                'Sexo': np.random.choice(['M', 'F'], p=[0.52, 0.48]),
                'Ciudad': fake.city(),
                'Provincia': fake.state(),
                'Fecha_Test': fake.date_between(start_date='-1y', end_date='today'),
                'Tiene_VIH': 'Sí' if tiene_vih else 'No',
                'Test_Resultado': 'Positivo' if test_positivo else 'Negativo',
                'Clasificación': clasificacion,
                'Estado': estado,
                'Prob_Real': prob_a_given_b if test_positivo else (1 - vpn/100)
            })
        
        df = pd.DataFrame(data)
        
        # Mostrar estadísticas del dataset generado
        st.success(f"✅ Dataset generado exitosamente con {len(df):,} registros")
        
        # Estadísticas
        st.markdown("### 📊 Estadísticas del Dataset Generado")
        
        col1, col2, col3, col4 = st.columns(4)
        
        vp_count = len(df[df['Estado'] == '✅ VP'])
        fp_count = len(df[df['Estado'] == '❌ FP'])
        vn_count = len(df[df['Estado'] == '✅ VN'])
        fn_count = len(df[df['Estado'] == '⚠️ FN'])
        
        with col1:
            st.metric("Verdaderos Positivos", vp_count, 
                     delta=f"{(vp_count/len(df))*100:.2f}%")
        with col2:
            st.metric("Falsos Positivos", fp_count,
                     delta=f"{(fp_count/len(df))*100:.2f}%")
        with col3:
            st.metric("Verdaderos Negativos", vn_count,
                     delta=f"{(vn_count/len(df))*100:.2f}%")
        with col4:
            st.metric("Falsos Negativos", fn_count,
                     delta=f"{(fn_count/len(df))*100:.2f}%")
        
        # Verificación del VPP en el dataset
        tests_positivos_df = df[df['Test_Resultado'] == 'Positivo']
        if len(tests_positivos_df) > 0:
            vpp_real = (vp_count / len(tests_positivos_df)) * 100
            st.metric(
                "VPP en Dataset Generado",
                f"{vpp_real:.2f}%",
                delta=f"Teórico: {percentage_result:.2f}%",
                help="Debería estar cerca del valor teórico"
            )
        
        # Mostrar muestra del dataset
        st.markdown("### 📋 Muestra del Dataset (Primeros 20 registros)")
        st.dataframe(
            df.head(20).style.applymap(
                lambda x: 'background-color: #dcfce7' if x == '✅ VP' or x == '✅ VN' 
                else 'background-color: #fee2e2' if x == '❌ FP' 
                else 'background-color: #fef3c7' if x == '⚠️ FN'
                else '',
                subset=['Estado']
            ),
            use_container_width=True
        )
        
        # Análisis adicional
        st.markdown("### 📈 Análisis del Dataset")
        
        tab1, tab2, tab3 = st.tabs(["Por Edad", "Por Sexo", "Por Ciudad"])
        
        with tab1:
            # Distribución por edad
            df['Grupo_Edad'] = pd.cut(df['Edad'], bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['18-25', '26-35', '36-45', '46-55', '56+'])
            
            edad_stats = df.groupby('Grupo_Edad')['Test_Resultado'].value_counts().unstack(fill_value=0)
            
            fig_edad = px.bar(
                edad_stats.reset_index(),
                x='Grupo_Edad',
                y=['Positivo', 'Negativo'],
                title='Distribución de Resultados por Grupo de Edad',
                labels={'value': 'Cantidad', 'Grupo_Edad': 'Grupo de Edad'},
                barmode='group'
            )
            st.plotly_chart(fig_edad, use_container_width=True)
        
        with tab2:
            # Distribución por sexo
            sexo_stats = df.groupby('Sexo')['Clasificación'].value_counts().unstack(fill_value=0)
            
            fig_sexo = px.bar(
                sexo_stats.reset_index(),
                x='Sexo',
                y=sexo_stats.columns.tolist(),
                title='Distribución de Clasificaciones por Sexo',
                labels={'value': 'Cantidad'},
                barmode='stack'
            )
            st.plotly_chart(fig_sexo, use_container_width=True)
        
        with tab3:
            # Top ciudades
            top_ciudades = df['Ciudad'].value_counts().head(10)
            
            fig_ciudad = px.bar(
                x=top_ciudades.index,
                y=top_ciudades.values,
                title='Top 10 Ciudades con Más Tests',
                labels={'x': 'Ciudad', 'y': 'Cantidad de Tests'}
            )
            fig_ciudad.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_ciudad, use_container_width=True)
        
        # Opciones de descarga
        st.markdown("### 💾 Descargar Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f'dataset_vih_simulado_{n_samples}.csv',
                mime='text/csv',
                use_container_width=True
            )
        
        with col2:
            # Excel
            from io import BytesIO
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Datos', index=False)
                
                # Agregar hoja de resumen
                resumen = pd.DataFrame({
                    'Parámetro': [
                        'Prevalencia (%)',
                        'Sensibilidad (%)',
                        'Especificidad (%)',
                        'VPP (%)',
                        'VPN (%)',
                        'Total Registros',
                        'Verdaderos Positivos',
                        'Falsos Positivos',
                        'Verdaderos Negativos',
                        'Falsos Negativos'
                    ],
                    'Valor': [
                        prevalence,
                        sensitivity,
                        specificity,
                        percentage_result,
                        vpn,
                        len(df),
                        vp_count,
                        fp_count,
                        vn_count,
                        fn_count
                    ]
                })
                resumen.to_excel(writer, sheet_name='Resumen', index=False)
            
            st.download_button(
                label="📥 Descargar Excel",
                data=buffer.getvalue(),
                file_name=f'dataset_vih_simulado_{n_samples}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                use_container_width=True
            )
        
        with col3:
            # JSON
            json_str = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Descargar JSON",
                data=json_str,
                file_name=f'dataset_vih_simulado_{n_samples}.json',
                mime='application/json',
                use_container_width=True
            )

# ==================== CALCULADORA INTERACTIVA ====================
st.header("🧮 Calculadora Rápida de Bayes")

st.info("Ingresa valores específicos para calcular P(A|B) rápidamente")

col1, col2, col3 = st.columns(3)

with col1:
    calc_prev = st.number_input("Prevalencia (%)", min_value=0.001, max_value=100.0, value=0.1, step=0.001, key="calc_prev")
with col2:
    calc_sens = st.number_input("Sensibilidad (%)", min_value=0.0, max_value=100.0, value=95.0, step=0.1, key="calc_sens")
with col3:
    calc_spec = st.number_input("Especificidad (%)", min_value=0.0, max_value=100.0, value=98.0, step=0.1, key="calc_spec")

if st.button("🔢 Calcular", use_container_width=True):
    p_a = calc_prev / 100
    p_b_a = calc_sens / 100
    p_bc_ac = calc_spec / 100
    p_b_ac = 1 - p_bc_ac
    
    num = p_b_a * p_a
    den = (p_b_a * p_a) + (p_b_ac * (1 - p_a))
    result = (num / den) * 100 if den > 0 else 0
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        ### Cálculo:
        ```
        P(A|B) = (0.{calc_sens:.0f} × {p_a:.6f}) / 
                 [(0.{calc_sens:.0f} × {p_a:.6f}) + 
                  ({p_b_ac:.2f} × {1-p_a:.6f})]
        
        P(A|B) = {num:.8f} / {den:.8f}
        
        P(A|B) = {result:.6f}
        ```
        """)
    
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                    padding: 2rem; 
                    border-radius: 15px; 
                    text-align: center;
                    margin-top: 1rem;">
            <h3 style="color: white; margin: 0;">Resultado</h3>
            <h1 style="color: white; font-size: 3rem; margin: 0.5rem 0;">
                {result:.2f}%
            </h1>
            <p style="color: white; margin: 0;">P(A|B)</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
### 📚 Referencias y Recursos

**Conceptos Clave:**
- **Teorema de Bayes**: Permite actualizar probabilidades con nueva información
- **Sensibilidad (P(B|A))**: Proporción de enfermos correctamente identificados
- **Especificidad (P(Bᶜ|Aᶜ))**: Proporción de sanos correctamente identificados
- **VPP (P(A|B))**: Probabilidad de enfermedad dado test positivo
- **VPN (P(Aᶜ|Bᶜ))**: Probabilidad de no enfermedad dado test negativo

**Fórmulas:**
""")

st.latex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)} = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|A^c) \cdot P(A^c)}")

st.markdown("""
**Bibliografía:**
- Bayes, T. (1763). "An Essay towards solving a Problem in the Doctrine of Chances"
- Altman, D. G., & Bland, J. M. (1994). "Diagnostic tests 2: Predictive values"
- Gigerenzer, G. (2002). "Calculated Risks: How to Know When Numbers Deceive You"

**Aplicaciones:**
- Pruebas de screening médico
- Detección de enfermedades raras
- Tests diagnósticos en general
- Análisis de riesgos
- Machine Learning y clasificación

---

<div style="text-align: center; color: #64748b; padding: 2rem;">
    <p><strong>🧬 Aplicación de Inferencia Bayesiana - Paradoja del Falso Positivo</strong></p>
    <p>Desarrollado con Streamlit, Plotly y Python Faker</p>
    <p>© 2025 - Proyecto Académico de Estadística Bayesiana</p>
</div>
""", unsafe_allow_html=True)