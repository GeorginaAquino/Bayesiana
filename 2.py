import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from dataclasses import dataclass
from typing import Tuple, List
from abc import ABC, abstractmethod


@dataclass
class InferenceResult:
    """Clase para almacenar resultados de inferencia"""
    mean: float
    ci_lower: float
    ci_upper: float
    method: str
    additional_info: dict = None


class Dataset:
    """Clase para manejar datasets y sus estadísticas"""
    
    def __init__(self, name: str, data: np.ndarray, description: str):
        self.name = name
        self.data = np.array(data)
        self.description = description
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Calcula estadísticas descriptivas"""
        self.n = len(self.data)
        self.mean = np.mean(self.data)
        self.median = np.median(self.data)
        self.std = np.std(self.data, ddof=1)
        self.variance = np.var(self.data, ddof=1)
        self.min = np.min(self.data)
        self.max = np.max(self.data)
        self.q25 = np.percentile(self.data, 25)
        self.q75 = np.percentile(self.data, 75)
    
    def test_normality(self) -> dict:
        """Realiza test de normalidad"""
        # Cálculo de asimetría y curtosis
        skewness = stats.skew(self.data)
        kurtosis = stats.kurtosis(self.data)
        
        # Test de Shapiro-Wilk
        if self.n < 5000:  # Shapiro-Wilk es válido para n < 5000
            shapiro_stat, shapiro_p = stats.shapiro(self.data)
        else:
            shapiro_stat, shapiro_p = None, None
        
        # Criterio simple: si skewness y kurtosis están cerca de 0
        is_normal = abs(skewness) < 1 and abs(kurtosis) < 1
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_stat': shapiro_stat,
            'shapiro_p': shapiro_p,
            'is_normal': is_normal
        }
    
    def get_histogram_data(self, bins: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Genera datos para histograma"""
        counts, bin_edges = np.histogram(self.data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts
    
    def get_stats_dict(self) -> dict:
        """Retorna diccionario con estadísticas"""
        return {
            'n': self.n,
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'variance': self.variance,
            'min': self.min,
            'max': self.max,
            'q25': self.q25,
            'q75': self.q75
        }


class InferenceMethod(ABC):
    """Clase abstracta para métodos de inferencia"""
    
    @abstractmethod
    def compute_confidence_interval(self, data: np.ndarray, confidence_level: float) -> InferenceResult:
        """Calcula intervalo de confianza"""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Retorna nombre del método"""
        pass


class ParametricInference(InferenceMethod):
    """Inferencia Paramétrica usando distribución t-Student"""
    
    def compute_confidence_interval(self, data: np.ndarray, confidence_level: float) -> InferenceResult:
        """Calcula IC usando distribución t-Student"""
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        se = std / np.sqrt(n)
        
        # Valor crítico de t-Student
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        
        # Intervalo de confianza
        margin_error = t_critical * se
        ci_lower = mean - margin_error
        ci_upper = mean + margin_error
        
        additional_info = {
            'se': se,
            'margin_error': margin_error,
            't_critical': t_critical,
            'df': n - 1
        }
        
        return InferenceResult(
            mean=mean,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="Paramétrico (t-Student)",
            additional_info=additional_info
        )
    
    def get_method_name(self) -> str:
        return "Paramétrico"


class NonParametricInference(InferenceMethod):
    """Inferencia No Paramétrica usando Bootstrap"""
    
    def __init__(self, n_bootstrap: int = 1000):
        self.n_bootstrap = n_bootstrap
        self.bootstrap_means = None
    
    def compute_confidence_interval(self, data: np.ndarray, confidence_level: float) -> InferenceResult:
        """Calcula IC usando Bootstrap"""
        n = len(data)
        bootstrap_means = []
        
        # Generar muestras bootstrap
        np.random.seed(42)  # Para reproducibilidad
        for _ in range(self.n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        self.bootstrap_means = np.array(bootstrap_means)
        
        # Calcular percentiles
        alpha = 1 - confidence_level
        ci_lower = np.percentile(self.bootstrap_means, 100 * alpha/2)
        ci_upper = np.percentile(self.bootstrap_means, 100 * (1 - alpha/2))
        mean_bootstrap = np.mean(self.bootstrap_means)
        
        additional_info = {
            'n_bootstrap': self.n_bootstrap,
            'bootstrap_std': np.std(self.bootstrap_means),
            'bootstrap_means': self.bootstrap_means
        }
        
        return InferenceResult(
            mean=mean_bootstrap,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            method="No Paramétrico (Bootstrap)",
            additional_info=additional_info
        )
    
    def get_method_name(self) -> str:
        return "No Paramétrico (Bootstrap)"


class Visualizer:
    """Clase para generar visualizaciones"""
    
    @staticmethod
    def plot_histogram(dataset: Dataset, title: str = "Histograma de Datos"):
        """Genera histograma usando Plotly"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=dataset.data,
            nbinsx=15,
            name='Frecuencia',
            marker_color='#6366f1'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=dataset.name,
            yaxis_title='Frecuencia',
            showlegend=False,
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_bootstrap_distribution(bootstrap_means: np.ndarray, ci_lower: float, ci_upper: float):
        """Genera histograma de distribución bootstrap"""
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=bootstrap_means,
            nbinsx=30,
            name='Bootstrap Means',
            marker_color='#10b981'
        ))
        
        # Agregar líneas verticales para IC
        fig.add_vline(x=ci_lower, line_dash="dash", line_color="red", 
                      annotation_text=f"CI Lower: {ci_lower:.2f}")
        fig.add_vline(x=ci_upper, line_dash="dash", line_color="red",
                      annotation_text=f"CI Upper: {ci_upper:.2f}")
        
        fig.update_layout(
            title="Distribución Bootstrap de Medias",
            xaxis_title='Media Bootstrap',
            yaxis_title='Frecuencia',
            showlegend=False,
            template='plotly_white',
            height=400
        )
        
        return fig
    
    @staticmethod
    def plot_comparison(param_result: InferenceResult, nonparam_result: InferenceResult, 
                       data_min: float, data_max: float):
        """Genera gráfico comparativo de intervalos de confianza"""
        fig = go.Figure()
        
        # Intervalo Paramétrico
        fig.add_trace(go.Scatter(
            x=[param_result.ci_lower, param_result.ci_upper],
            y=['Paramétrico', 'Paramétrico'],
            mode='lines+markers',
            name='Paramétrico',
            line=dict(color='#3b82f6', width=8),
            marker=dict(size=12)
        ))
        
        # Media Paramétrica
        fig.add_trace(go.Scatter(
            x=[param_result.mean],
            y=['Paramétrico'],
            mode='markers',
            name='Media Param',
            marker=dict(size=15, color='#1e40af', symbol='diamond')
        ))
        
        # Intervalo No Paramétrico
        fig.add_trace(go.Scatter(
            x=[nonparam_result.ci_lower, nonparam_result.ci_upper],
            y=['No Paramétrico', 'No Paramétrico'],
            mode='lines+markers',
            name='No Paramétrico',
            line=dict(color='#10b981', width=8),
            marker=dict(size=12)
        ))
        
        # Media No Paramétrica
        fig.add_trace(go.Scatter(
            x=[nonparam_result.mean],
            y=['No Paramétrico'],
            mode='markers',
            name='Media Bootstrap',
            marker=dict(size=15, color='#065f46', symbol='diamond')
        ))
        
        fig.update_layout(
            title="Comparación de Intervalos de Confianza",
            xaxis_title='Valor',
            yaxis_title='Método',
            showlegend=True,
            template='plotly_white',
            height=300,
            xaxis=dict(range=[data_min * 0.95, data_max * 1.05])
        )
        
        return fig


class InferenceAnalyzer:
    """Clase principal que coordina el análisis"""
    
    def __init__(self):
        self.datasets = self._load_datasets()
        self.parametric_method = ParametricInference()
        self.visualizer = Visualizer()
    
    def _load_datasets(self) -> dict:
        """Carga los datasets predefinidos"""
        datasets = {
            'salarios': Dataset(
                name='Salarios Mensuales (USD)',
                data=np.array([2800, 3200, 2900, 3500, 3100, 2750, 3300, 3400, 2950, 3150, 
                              3250, 3050, 3350, 3450, 2850, 3550, 3650, 3000, 3100, 3200,
                              3300, 2900, 3400, 3500, 3150, 3250, 3350, 3050, 3150, 3250,
                              3450, 3550, 2950, 3050, 3150, 3350, 3450, 3550, 3650, 3750,
                              8500, 9200, 7800]),
                description='Salarios de 43 empleados en una empresa de tecnología (incluye outliers)'
            ),
            'tiempos': Dataset(
                name='Tiempos de Respuesta (ms)',
                data=np.array([125, 132, 118, 145, 128, 135, 142, 138, 130, 148,
                              155, 122, 140, 136, 150, 128, 133, 147, 152, 129,
                              137, 143, 126, 134, 141, 149, 131, 139, 146, 127,
                              135, 142, 138, 145, 151, 128, 136, 144, 132, 140]),
                description='Tiempos de respuesta de un servidor web (40 mediciones)'
            )
        }
        return datasets
    
    def analyze_dataset(self, dataset_key: str, confidence_level: float, 
                       n_bootstrap: int) -> Tuple[InferenceResult, InferenceResult]:
        """Realiza análisis completo de un dataset"""
        dataset = self.datasets[dataset_key]
        
        # Inferencia Paramétrica
        param_result = self.parametric_method.compute_confidence_interval(
            dataset.data, confidence_level
        )
        
        # Inferencia No Paramétrica
        nonparam_method = NonParametricInference(n_bootstrap=n_bootstrap)
        nonparam_result = nonparam_method.compute_confidence_interval(
            dataset.data, confidence_level
        )
        
        return param_result, nonparam_result
    
    def get_recommendation(self, dataset_key: str, param_result: InferenceResult, 
                          nonparam_result: InferenceResult) -> dict:
        """Genera recomendación basada en test de normalidad"""
        dataset = self.datasets[dataset_key]
        normality = dataset.test_normality()
        
        if normality['is_normal']:
            recommendation = "AMBOS MÉTODOS SON VÁLIDOS"
            explanation = f"""
            Los datos muestran características de normalidad:
            - Asimetría: {normality['skewness']:.3f} (ideal: cercano a 0)
            - Curtosis: {normality['kurtosis']:.3f} (ideal: cercano a 0)
            
            **Recomendación:** El enfoque paramétrico es preferible por ser más 
            eficiente estadísticamente y producir intervalos más precisos.
            El enfoque no paramétrico confirma los resultados.
            """
            method_preference = "parametric"
        else:
            recommendation = "USAR ENFOQUE NO PARAMÉTRICO"
            explanation = f"""
            Los datos muestran desviación de la normalidad:
            - Asimetría: {normality['skewness']:.3f} (alto si |valor| > 1)
            - Curtosis: {normality['kurtosis']:.3f} (alto si |valor| > 1)
            
            **Recomendación:** El enfoque no paramétrico (Bootstrap) es más confiable 
            porque NO asume normalidad y es robusto ante desviaciones distribucionales 
            y valores atípicos.
            """
            method_preference = "nonparametric"
        
        return {
            'recommendation': recommendation,
            'explanation': explanation,
            'normality': normality,
            'preference': method_preference
        }


def main():
    """Función principal de Streamlit"""
    
    # Configuración de página
    st.set_page_config(
        page_title="Inferencia Paramétrica vs No Paramétrica",
        page_icon="📊",
        layout="wide"
    )
    
    # Título principal
    st.title("📊 Inferencia Paramétrica vs No Paramétrica")
    st.markdown("---")
    
    # Inicializar analizador
    analyzer = InferenceAnalyzer()
    
    # Sidebar con controles
    st.sidebar.header("⚙️ Configuración")
    
    dataset_key = st.sidebar.selectbox(
        "Selecciona Dataset:",
        options=list(analyzer.datasets.keys()),
        format_func=lambda x: analyzer.datasets[x].name
    )
    
    confidence_level = st.sidebar.slider(
        "Nivel de Confianza:",
        min_value=0.90,
        max_value=0.99,
        value=0.95,
        step=0.01,
        format="%.2f"
    )
    
    n_bootstrap = st.sidebar.slider(
        "Número de Remuestras Bootstrap:",
        min_value=500,
        max_value=5000,
        value=1000,
        step=500
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Cambia los parámetros para ver cómo afectan los resultados")
    
    # Tabs principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Introducción", 
        "📊 Datos y Exploración", 
        "📐 Enfoque Paramétrico",
        "🔄 Enfoque No Paramétrico",
        "🎯 Comparación"
    ])
    
    # TAB 1: Introducción
    with tab1:
        st.header("¿Qué es la Inferencia Estadística?")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #FFFFFF; padding: 20px; border-radius: 10px; border: 4px solid #3B82F6; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #1E40AF;'>📐 Inferencia Paramétrica</h3>
                <p style='color: #1F2937;'><strong>Asume</strong> que los datos siguen una distribución específica (ej: Normal)</p>
                <h4 style='color: #1F2937;'>✓ Ventajas:</h4>
                <ul style='color: #374151;'>
                    <li>Más eficiente con muestras pequeñas</li>
                    <li>Intervalos de confianza más precisos</li>
                    <li>Métodos bien establecidos</li>
                </ul>
                <h4 style='color: #1F2937;'>✗ Limitaciones:</h4>
                <ul style='color: #374151;'>
                    <li>Requiere supuestos distribucionales</li>
                    <li>Sensible a valores atípicos</li>
                    <li>Puede ser inexacta si no se cumplen supuestos</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: #FFFFFF; padding: 20px; border-radius: 10px; border: 4px solid #10B981; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #065F46;'>🔄 Inferencia No Paramétrica</h3>
                <p style='color: #1F2937;'><strong>NO asume</strong> distribución específica. Usa los datos directamente.</p>
                <h4 style='color: #1F2937;'>✓ Ventajas:</h4>
                <ul style='color: #374151;'>
                    <li>Sin supuestos distribucionales</li>
                    <li>Robusta ante valores atípicos</li>
                    <li>Funciona con distribuciones complejas</li>
                </ul>
                <h4 style='color: #1F2937;'>✗ Limitaciones:</h4>
                <ul style='color: #374151;'>
                    <li>Requiere más datos para precisión</li>
                    <li>Computacionalmente más intensiva</li>
                    <li>Puede tener menor poder estadístico</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.success("🎯 **Objetivo:** Comparar ambos enfoques para estimar la media poblacional e intervalos de confianza usando datos reales.")
    
    # Obtener dataset actual
    dataset = analyzer.datasets[dataset_key]
    
    # TAB 2: Datos y Exploración
    with tab2:
        st.header("📊 Exploración de Datos")
        
        st.info(f"**Dataset:** {dataset.name} - {dataset.description}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Estadísticas Descriptivas")
            stats_dict = dataset.get_stats_dict()
            
            stats_df = pd.DataFrame({
                'Estadística': ['Tamaño (n)', 'Media (x̄)', 'Mediana', 'Desv. Estándar (s)', 
                               'Varianza', 'Mínimo', 'Máximo', 'Q1', 'Q3'],
                'Valor': [
                    f"{stats_dict['n']}",
                    f"{stats_dict['mean']:.2f}",
                    f"{stats_dict['median']:.2f}",
                    f"{stats_dict['std']:.2f}",
                    f"{stats_dict['variance']:.2f}",
                    f"{stats_dict['min']:.2f}",
                    f"{stats_dict['max']:.2f}",
                    f"{stats_dict['q25']:.2f}",
                    f"{stats_dict['q75']:.2f}"
                ]
            })
            
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔍 Test de Normalidad")
            normality = dataset.test_normality()
            
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Asimetría (Skewness)", f"{normality['skewness']:.3f}")
                st.caption("Ideal: cercano a 0 (±0.5)")
            
            with col2_2:
                st.metric("Curtosis (Kurtosis)", f"{normality['kurtosis']:.3f}")
                st.caption("Ideal: cercano a 0 (±0.5)")
            
            if normality['is_normal']:
                st.success("✓ Los datos parecen seguir una distribución normal")
            else:
                st.warning("✗ Los datos muestran desviación de la normalidad")
        
        st.markdown("---")
        st.subheader("📊 Histograma de Datos")
        fig_hist = analyzer.visualizer.plot_histogram(dataset)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Realizar análisis
    param_result, nonparam_result = analyzer.analyze_dataset(
        dataset_key, confidence_level, n_bootstrap
    )
    
    # TAB 3: Enfoque Paramétrico
    with tab3:
        st.header("📐 Enfoque Paramétrico (Distribución t-Student)")
        
        st.info("**Supuesto:** Asumimos que los datos provienen de una distribución Normal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Fórmulas Utilizadas")
            st.latex(r"SE = \frac{s}{\sqrt{n}} = " + f"{param_result.additional_info['se']:.4f}")
            st.latex(r"ME = t_{\alpha/2} \times SE")
            st.latex(r"IC = \bar{x} \pm ME")
            
            st.markdown(f"""
            - **Grados de libertad:** {param_result.additional_info['df']}
            - **Valor crítico t:** {param_result.additional_info['t_critical']:.4f}
            - **Error Estándar:** {param_result.additional_info['se']:.4f}
            - **Margen de Error:** {param_result.additional_info['margin_error']:.4f}
            """)
        
        with col2:
            st.subheader("🎯 Resultados")
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Media", f"{param_result.mean:.2f}")
            with col2_2:
                st.metric("IC Inferior", f"{param_result.ci_lower:.2f}")
            with col2_3:
                st.metric("IC Superior", f"{param_result.ci_upper:.2f}")
            
            st.metric("Amplitud del IC", 
                     f"{param_result.ci_upper - param_result.ci_lower:.2f}")
        
        st.success(f"""
        **Interpretación:** Con un {confidence_level*100:.0f}% de confianza, estimamos que la 
        verdadera media poblacional se encuentra entre **{param_result.ci_lower:.2f}** y 
        **{param_result.ci_upper:.2f}**.
        """)
    
    # TAB 4: Enfoque No Paramétrico
    with tab4:
        st.header("🔄 Enfoque No Paramétrico (Bootstrap)")
        
        st.info("**Método:** Remuestreo con reemplazo - NO asumimos ninguna distribución")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Algoritmo Bootstrap")
            st.markdown(f"""
            1. Tomar muestra aleatoria con reemplazo (tamaño n={dataset.n})
            2. Calcular la media de la muestra
            3. Repetir pasos 1-2 **{n_bootstrap:,} veces**
            4. Usar percentiles para construir el IC
            
            - **Percentil inferior ({(1-confidence_level)/2*100:.1f}%):** {nonparam_result.ci_lower:.2f}
            - **Percentil superior ({(1-(1-confidence_level)/2)*100:.1f}%):** {nonparam_result.ci_upper:.2f}
            """)
        
        with col2:
            st.subheader("🎯 Resultados")
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Media Bootstrap", f"{nonparam_result.mean:.2f}")
            with col2_2:
                st.metric("IC Inferior", f"{nonparam_result.ci_lower:.2f}")
            with col2_3:
                st.metric("IC Superior", f"{nonparam_result.ci_upper:.2f}")
            
            st.metric("Amplitud del IC", 
                     f"{nonparam_result.ci_upper - nonparam_result.ci_lower:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Distribución Bootstrap de Medias")
        fig_bootstrap = analyzer.visualizer.plot_bootstrap_distribution(
            nonparam_result.additional_info['bootstrap_means'],
            nonparam_result.ci_lower,
            nonparam_result.ci_upper
        )
        st.plotly_chart(fig_bootstrap, use_container_width=True)
        
        st.success(f"""
        **Interpretación:** Con un {confidence_level*100:.0f}% de confianza, estimamos que la 
        verdadera media poblacional se encuentra entre **{nonparam_result.ci_lower:.2f}** y 
        **{nonparam_result.ci_upper:.2f}** usando {n_bootstrap:,} remuestras bootstrap.
        """)
    
    # TAB 5: Comparación
    with tab5:
        st.header("🎯 Comparación y Conclusiones")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: #EFF6FF; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #1E40AF;'>📐 Paramétrico</h3>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Media", f"{param_result.mean:.2f}")
            st.metric("IC", f"[{param_result.ci_lower:.2f}, {param_result.ci_upper:.2f}]")
            st.metric("Amplitud", f"{param_result.ci_upper - param_result.ci_lower:.2f}")
        
        with col2:
            st.markdown("""
            <div style='background-color: #F0FDF4; padding: 20px; border-radius: 10px;'>
                <h3 style='color: #065F46;'>🔄 No Paramétrico</h3>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Media", f"{nonparam_result.mean:.2f}")
            st.metric("IC", f"[{nonparam_result.ci_lower:.2f}, {nonparam_result.ci_upper:.2f}]")
            st.metric("Amplitud", f"{nonparam_result.ci_upper - nonparam_result.ci_lower:.2f}")
        
        st.markdown("---")
        st.subheader("📊 Visualización Comparativa")
        fig_comparison = analyzer.visualizer.plot_comparison(
            param_result, nonparam_result, dataset.min, dataset.max
        )
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        st.markdown("---")
        st.subheader("💡 Recomendación")
        
        recommendation = analyzer.get_recommendation(dataset_key, param_result, nonparam_result)
        
        if recommendation['preference'] == 'parametric':
            st.success(f"✓ **{recommendation['recommendation']}**")
        else:
            st.warning(f"⚠️ **{recommendation['recommendation']}**")
        
        st.markdown(recommendation['explanation'])
        
        st.markdown("---")
        st.subheader("🔬 Diferencias Observadas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Media Estimada:**")
            st.write(f"Paramétrico: {param_result.mean:.3f}")
            st.write(f"Bootstrap: {nonparam_result.mean:.3f}")
            st.write(f"Diferencia: {abs(param_result.mean - nonparam_result.mean):.3f}")
        
        with col2:
            st.markdown("**Límite Inferior IC:**")
            st.write(f"Paramétrico: {param_result.ci_lower:.2f}")
            st.write(f"Bootstrap: {nonparam_result.ci_lower:.2f}")
            st.write(f"Diferencia: {abs(param_result.ci_lower - nonparam_result.ci_lower):.2f}")
        
        with col3:
            st.markdown("**Límite Superior IC:**")
            st.write(f"Paramétrico: {param_result.ci_upper:.2f}")
            st.write(f"Bootstrap: {nonparam_result.ci_upper:.2f}")
            st.write(f"Diferencia: {abs(param_result.ci_upper - nonparam_result.ci_upper):.2f}")
        
        st.markdown("---")
        st.info("""
        ### 🎓 Conclusiones Generales
        
        **Usa Inferencia Paramétrica cuando:**
        - ✓ Los datos siguen distribución normal
        - ✓ Tienes muestra pequeña (n < 30)
        - ✓ No hay valores atípicos significativos
        - ✓ Quieres mayor eficiencia estadística
        
        **Usa Inferencia No Paramétrica cuando:**
        - ✓ Distribución desconocida o no normal
        - ✓ Presencia de valores atípicos
        - ✓ Datos con asimetría marcada
        - ✓ Muestra grande (bootstrap funciona mejor)
        
        💡 **Recomendación práctica:** Aplica ambos métodos y compara. Si difieren 
        significativamente, investiga la causa y favorece el método no paramétrico.
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #6B7280;'>
        
        <p>Streamlit + Plotly + NumPy + SciPy</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":

    main()

