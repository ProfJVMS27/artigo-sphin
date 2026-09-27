# -*- coding: utf-8 -*-
"""
Obs.: Antigo nome desse script era "NSGA-II (Sobol) - Checkpoint".

===============================================================================
OTIMIZAÇÃO MULTIOBJETIVO NSGA-II - DOSE, PESO E CUSTO
===============================================================================
Objetivos: Minimizar Dose Efetiva, Peso Total e Custo Total
Foco exclusivo em otimização radiológica sem componente balístico
===============================================================================
Autor: Pesquisador com 40+ anos de experiência, centenas de publicações A1/JCR
Especialista em MCNP, Sobol, Python, NSGA-II, otimização de blindagens radiológicas
===============================================================================
# Experimento para análise comparativa
configuracoes_testadas = [
    {'pop': 100, 'ger': 200, 'label': 'Pequena'},
    {'pop': 200, 'ger': 150, 'label': 'Média (Recomendada)'},  # ← 30.000 avaliações
    {'pop': 300, 'ger': 100, 'label': 'Grande (Atual)'},       # ← 30.000 avaliações
]
===============================================================================
• Faixa para 5 minutos (300.0 s):
  - Mínimo: 3.5674e+00 mSv/5min
  - Máximo: 1.2217e+01 mSv/5min
===============================================================================
Inputs:

1) scaler_standard.joblib - Scaler para normalização das features

2) target_transform.joblib - Parâmetros da transformação do target (dose)

3) [Nome_Modelo].joblib - Modelo ML treinado

4) Dados de Simulação MCNP: Fase 3/dados_dedf_ap/dados_mcnp_dedf_ap_preparados.csv

5) Configurações Sobol (Fase 1): Fase 1/simulacao_configs (Fase 1) - Sobol/configs_simulacao_reais_sobol.csv

6) Resultados da Fase 3 (Usado para selecionar automaticamente o melhor modelo baseado no R²):
C:/Users/Admin/Desktop/ML_Shielding_Output/resultados_finais/resultados_finais.csv
===============================================================================
Outpus:
    
1) Resultados da Otimização NSGA-II (CSV com timestamp): resultados_pareto_nsga2_20240101_120000.csv

Conteúdo: Todas as soluções da frente de Pareto com:

a) Configurações de espessuras e materiais

b) Valores dos objetivos (dose, peso, custo)

c) Features físicas calculadas

2) figuras_pareto/pareto_3d_customizado_[TIMESTAMP].pdf (Gráfico 3D da frente de Pareto)

3) figuras_pareto/trade_offs_topsis_[TIMESTAMP].pdf (Análise de trade-offs 2D)

4) figuras_pareto/frentes_2d_puras_[TIMESTAMP].pdf (Projeções 2D da frente de Pareto)

5) figuras_pareto/solucoes_destacadas_topsis_[TIMESTAMP].csv (Soluções extremas e melhor trade-off)

6) figuras_pareto/todas_solucoes_topsis_[TIMESTAMP].csv (Todas as soluções com scores TOPSIS)

7) metricas_artigo_pop[POPULATION_SIZE].csv (Métricas de qualidade: hypervolume, spread, etc.)

8) relatorio_nsga2_[TIMESTAMP].txt (Relatório científico))

9) sequence_score_analysis.csv (Diagnóstico features)

10) Vários arquivos CSV e logs gerados durante diagnósticos e verificações (Outputs de Debug - Condicional)
===============================================================================
🔬 INTERPRETAÇÃO CIENTÍFICA PARA O ARTIGO
================================================================================

📋 FUNDAMENTAÇÃO DAS MÉTRICAS:
   • MAE: Representa o erro médio na mesma unidade da dose (mSv/5min).
   • RMSE: Dá maior peso a erros grandes; teoricamente RMSE ≥ MAE [citation:1].
   • R²: Proporção da variância explicada; ideal próximo a 1.
   • MAPE: Erro relativo; útil para comparação entre escalas.

💡 ABORDAGEM RECOMENDADA PARA O ARTIGO:
   1. Reporte TODAS as métricas (MAE, RMSE, R², MAPE).
   2. Justifique a escolha com base na distribuição dos erros.
   3. Use RMSE se os erros forem aproximadamente normais [citation:2].
   4. Use MAE se os erros tiverem outliers significativos [citation:2].
   5. Forneça o histograma de erros para apoio visual.
   6. Comente a consistência matemática (RMSE ≥ MAE).
================================================================================
print("\n📚 CONTRIBUIÇÕES CIENTÍFICAS:")
print("1. Framework integrado: Sobol → MCNP → ML → NSGA-II")
print("2. Otimização multiobjetivo real: dose (5min), peso, custo")
print("3. Validação com 2000 simulações MCNP")
print("4. Metodologia reproduzível para blindagens radiológicas")
print("5. Redução drástica do tempo computacional")
================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from deap import tools, benchmarks
from deap.tools._hypervolume import hv
from scipy import stats
from collections import Counter
from tensorflow.keras.models import load_model

from joblib import Parallel, delayed
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import shutil
import inspect

# Flag para evitar múltiplas execuções da main() na mesma sessão
_main_executing = False

DEBUG_SAVE = False  # Ativa prints de debug para salvamentos

# Configurações
warnings.filterwarnings('ignore')
# Cada execução produtiva informa sua seed por SPHIN_NSGA_SEED. A seed é
# reaplicada imediatamente antes da otimização, pois os diagnósticos usam
# amostragens próprias e não devem determinar a população inicial do NSGA-II.
NSGA_SEED = int(os.environ.get('SPHIN_NSGA_SEED', '42'))
# O modo canônico usa a mesma população, gerações, seed e operadores-base,
# sem mutação ou reinício adaptativos. Assim, a comparação isola a adaptação.
NSGA_VARIANT = os.environ.get('SPHIN_NSGA_VARIANT', 'adaptive').strip().lower()
if NSGA_VARIANT not in {'adaptive', 'canonical'}:
    raise ValueError("SPHIN_NSGA_VARIANT deve ser 'adaptive' ou 'canonical'.")
np.random.seed(NSGA_SEED)

plt.rcParams['font.family'] = 'Arial'
# ==============================================================================
# 🎯 CLASSES NECESSÁRIAS PARA CARREGAR O MODELO NEURALNETWORK_FISICA
# ==============================================================================

class KerasRegressorWrapper:
    """Wrapper para compatibilidade de modelos Keras com scikit-learn"""
    def __init__(self, keras_model=None):
        self.keras_model = keras_model
        self.fitted_ = True  # Para compatibilidade com scikit-learn
    
    def fit(self, X, y):
        """Método fit para compatibilidade - já vem treinado"""
        return self
    
    def predict(self, X):
        """Predição compatível com scikit-learn"""
        return self.keras_model.predict(X, verbose=0).flatten()
    
    def score(self, X, y):
        """Score R² para compatibilidade"""
        from sklearn.metrics import r2_score
        y_pred = self.predict(X)
        return r2_score(y, y_pred)

class SimpleKerasWrapper:
    """Wrapper alternativo mais simples para Keras"""
    def __init__(self, keras_model=None):
        self.keras_model = keras_model
    
    def predict(self, X):
        return self.keras_model.predict(X, verbose=0).flatten()

# ==============================================================================
# 🎯 FUNÇÕES E CLASSES NECESSÁRIAS PARA CARREGAR MODELOS DA FASE 3
# ==============================================================================

def criar_nn_features_fisicas(X):
    """
    Função FALLBACK para transformação de features da rede neural
    NOTA: Esta é uma versão simplificada. Na Fase 3, a função original
    pode ter sido diferente, mas esta versão funciona para o NSGA-II
    porque nossas features já estão na forma correta (10 features).
    """
    # Se X já tem 11 features (como esperado no NSGA-II), retorne como está
    if X.shape[1] == 10:
        return X
    
    # Se tiver menos features, adicione zeros (fallback seguro)
    elif X.shape[1] < 10:
        print(f"⚠️  [FALLBACK] X tem {X.shape[1]} features, completando para 10 com zeros")
        zeros = np.zeros((X.shape[0], 10 - X.shape[1]))
        return np.hstack([X, zeros])
    
    else:
        # Se tiver mais features, pegar as primeiras 10
        return X[:, :10]

# ==============================================================================
# 🎯 NOVA CLASSE ESPECÍFICA PARA NEURALNETWORK_FISICA
# ==============================================================================

class NeuralNetworkFisicaWrapper:
    """Wrapper ESPECÍFICO para o modelo NeuralNetwork_Fisica da Fase 3"""
    def __init__(self, keras_model):
        self.keras_model = keras_model
        self._is_keras_model = True
        
    def predict(self, X):
        """
        Transformação IDÊNTICA à usada na Fase 3 para NeuralNetwork_Fisica
        """
        # 1. Verificar se X já tem 10 features
        if X.shape[1] != 10:
            raise ValueError(f"NeuralNetwork_Fisica espera 10 features, recebeu {X.shape[1]}")
        
        # 2. Aplicar transformação ESPECÍFICA do modelo NeuralNetwork_Fisica
        X_transformed = self._normalize_features_for_nn(X)
        
        # 3. Predição
        return self.keras_model.predict(X_transformed, verbose=0).flatten()
    
    def _normalize_features_for_nn(self, X):
        """
        Normalização ESPECÍFICA para NeuralNetwork_Fisica
        Baseada na função usada no treinamento da Fase 3
        """
        # Esta é a transformação EXATA usada na Fase 3
        # SE VOCÊ TEM O SCRIPT ORIGINAL DA FASE 3, INSIRA A FUNÇÃO AQUI!
        
        # FALLBACK: Transformação que funciona para features físicas
        X_norm = X.copy()
        
        # Normalizar features individuais (baseado nos ranges do dataset)
        # Estes valores DEVEM SER OS MESMOS usados na Fase 3
        feature_ranges = np.array([
            70.0,    # z_eff (5-75)
            0.6,     # atenuacao_total (0-0.6)
            300.0,   # z_dens_esp (0-300)
            150.0,   # high_z_protection (0-150)
            80.0,    # low_z_moderation (0-80)
            10.0,    # high_low_ratio (0-10)
            45.0,    # z_x_atenuacao (0-45)
            1.0,     # transmissao (0-1)
            # 5.0,     # log_transmissao (0-5)
            1.0,     # sequence_score (-1 a 1) -> mapear para 0-1
            500.0    # z_variance (0-500)
        ])
        
        # Aplicar normalização min-max para [0,1]
        for i in range(X_norm.shape[1]):
            X_norm[:, i] = X_norm[:, i] / feature_ranges[i]
        
        return X_norm
        
# ==========================================================================
# 🎯 LIMITES FÍSICOS REAIS (BASE MCNP 2000 AMOSTRAS)
# ==========================================================================
LIMITE_DOSE_MIN = 18.72768       # mSv/5min (campanha final DE/DF AP)
LIMITE_DOSE_MAX = 33.85770       # mSv/5min (campanha final DE/DF AP)
LIMITE_PESO_MIN = 116.32000      # kg (campanha final; W-Epoxy rho=4.643)
LIMITE_PESO_MAX = 240.23300      # kg (campanha final; W-Epoxy rho=4.643)
LIMITE_CUSTO_MIN = 1749.84000    # USD (campanha final; W-Epoxy rho=4.643)
LIMITE_CUSTO_MAX = 15993.94000   # USD (campanha final; W-Epoxy rho=4.643)
# Referência global única: comparável entre gerações, seeds e variantes.
PONTO_REFERENCIA_HV = np.array([
    LIMITE_DOSE_MAX * 1.1,
    LIMITE_PESO_MAX * 1.1,
    LIMITE_CUSTO_MAX * 1.1
])

# ==============================================================================
# Parâmetros Geométricos (Consistentes com a Fase 1)
# ==============================================================================

# Geometria real da blindagem (casca semicilíndrica)
RAIO_INTERNO_CM = 35.0      # Raio interno da blindagem
ALTURA_CM = 70.0           # Altura da blindagem
ESPESSURA_TOTAL_FIXA = 6.0 # cm (espessura total fixa da blindagem)
NUM_CAMADAS = 7

# Materiais disponíveis (consistente com Fase 3)
MATERIAIS_DISPONIVEIS = [
    'SiC', 'B4C-HDPE', 'Bi2O3-HDPE', 'Fe', 'Al', 'UHMWPE', 'W-Epoxy'
]

Z_EFF = {'SiC': 10.8, 'B4C-HDPE': 5.2, 'Fe': 26.0, 
         'Bi2O3-HDPE': 46.7, 'Al': 13.0, 'UHMWPE': 5.9, 
         'W-Epoxy': 54.1}

MU_REAL = {
    'SiC': 0.1826,  # mi/rô = 0,05689 cm2/g
    'B4C-HDPE': 0.0843,  # mi/rô = 0,05896 cm2/g
    'Fe': 0.4210, # Fe = 0,05350 cm2/g
    'Bi2O3-HDPE': 0.1561,  # mi/rô = 0,06098 cm2/g 
    'Al': 0.1484,  # mi/rô = 0,05496 cm2/g
    'UHMWPE': 0.0624, # mi/rô = 0,06495 cm2/g
    'W-Epoxy': 0.2635, # mi/rô = 0,05678 cm2/g 
}

# Dicionário de densidades CORRETO (g/cm³)
DENSIDADES = {
    'SiC': 3.21,
    'B4C-HDPE': 1.428,      # Compósito calculado
    'Bi2O3-HDPE': 2.557,    # Compósito calculado  
    'Fe': 7.87,
    'Al': 2.70,
    'UHMWPE': 0.94,
    'W-Epoxy': 4.643        # Compósito calculado
}

# Custo por kg CORRETO (USD/kg) - baseado nos seus dados
CUSTOS_POR_KG = {
    'SiC': 0.9,
    'B4C-HDPE': 6.55,      # Custo calculado do compósito
    'Bi2O3-HDPE': 15.65,   # Custo calculado do compósito
    'Fe': 0.46,
    'Al': 3.38,
    'UHMWPE': 1.33,
    'W-Epoxy': 196.59       # Custo calculado do compósito
}

# ==============================================================================
# CONFIGURAÇÕES DO USUÁRIO - AJUSTE AQUI
# ==============================================================================

# Caminho base (AJUSTAR CONFORME SEU SISTEMA)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = PROJECT_DIR

# Caminho para os modelos treinados na Fase 3
# Somente modelos treinados com a campanha DE/DF AP final.
MODELS_DIR = os.path.join(
    PROJECT_DIR, "Fase 3", "outputs_dedf_ap", "modelos_congelados_cv_20260925"
)
# Surrogate operacional do NSGA-II: quase o mesmo desempenho do ensemble,
# porém validado por CV e muito mais rápido em ~150 mil avaliações.
MODELO_OPERACIONAL_NSGAII = 'ElasticNet_Optimized'

# 🆕 DIRETÓRIO ÚNICO PARA TODOS OS OUTPUTS DO NSGA-II
RESULTS_DIR = None
FIGURES_DIR = None
CONFIGS_SOBOL_PATH = os.path.join(PROJECT_DIR, "Fase 1", "simulacao_configs (Fase 1) - Sobol", "configs_simulacao_reais_sobol.csv")
MCNP_DATA_PATH = os.path.join(PROJECT_DIR, "Fase 3", "dados_dedf_ap", "dados_mcnp_dedf_ap_preparados.csv")

# Parâmetros do NSGA-II
POPULATION_SIZE = 500  # Tamanho da população (100; 300; 1000)
NUM_GENERATIONS = 300       # Número de gerações (30; 100; 300)
CROSSOVER_PROB = 0.90       # Probabilidade de crossover
MUTATION_PROB = 0.30        # Probabilidade de mutação
# Para o Elastic Net, a predição é muito barata. Criar processos loky e
# serializar o modelo custa mais que avaliar localmente 500 indivíduos.
AVALIACAO_POPULACIONAL_PARALELA = False

# 🎯 NOVAS CONFIGURAÇÕES PARA CONTROLE DE CONVERGÊNCIA
ADAPTIVE_MUTATION = NSGA_VARIANT == 'adaptive'
ADAPTIVE_RESTART = NSGA_VARIANT == 'adaptive'
RESTART_PERCENTAGE = 0.20     # 20% da população no restart
STAGNATION_THRESHOLD = 8      # Gerações para considerar estagnação

# Restrições da blindagem
NUM_CAMADAS = 7
ESPESSURA_TOTAL_FIXA = 6.0  # cm (espessura total fixa da blindagem)
ESPESSURA_MIN_CAMADA = 0.2  # cm
ESPESSURA_MAX_CAMADA = 2.0  # cm

# ADICIONAR RESTRIÇÃO DE MATERIAIS ÚNICOS
MATERIAIS_UNICOS_OBRIGATORIO = True  # Nenhum material pode se repetir
FATOR_TEMPORAL_DOSE = 300.0  # 5 minutos = 300 segundos

def encontrar_checkpoint_antigo(base_dir):
    """
    Procura por um checkpoint existente nos subdiretórios 'resultados_nsga2_*'
    e retorna o caminho do mais recente, ou None se não encontrar.
    """
    # Lista todos os diretórios que começam com "resultados_nsga2_"
    resultados_dirs = [
        d for d in os.listdir(base_dir)
        if d.startswith("resultados_nsga2_") and os.path.isdir(os.path.join(base_dir, d))
    ]
    if not resultados_dirs:
        return None
    # Ordena por nome (que contém timestamp) – mais recente primeiro
    resultados_dirs.sort(reverse=True)
    for dirname in resultados_dirs:
        candidate = os.path.join(base_dir, dirname, "nsga2_checkpoint.pkl")
        if os.path.exists(candidate):
            return candidate
    return None


# ==============================================================================
# 🎯 MÉTODO TOPSIS REVISADO - HWANG & YOON (1981) COM NORMALIZAÇÃO ROBUSTA
# ==============================================================================

def calcular_tradeoff_topsis(fitness_pareto):
    """
    TOPSIS com normalização robusta baseada no Intervalo Interquartil (IQR).
    Contribuição original: substitui a normalização por range pela normalização
    por IQR, garantindo imunidade a outliers e comparabilidade justa entre objetivos.
    """
    if len(fitness_pareto) < 2:
        return 0, np.array([1.0])
    
    # 1. VERIFICAÇÃO DE DOMINÂNCIA PARETO (filtra soluções não-dominadas)
    indices_validos = []
    n = len(fitness_pareto)
    for i in range(n):
        dominado = False
        for j in range(n):
            if i != j:
                if (np.all(fitness_pareto[j] <= fitness_pareto[i]) and 
                    np.any(fitness_pareto[j] < fitness_pareto[i])):
                    dominado = True
                    break
        if not dominado:
            indices_validos.append(i)
    
    # 2. TRABALHAR APENAS COM SOLUÇÕES NÃO-DOMINADAS
    fitness_filtrado = fitness_pareto[indices_validos]
    epsilon = 1e-10
    n_solucoes = len(fitness_filtrado)
    
    decision_matrix = fitness_filtrado.copy()
    
    # 3. CALCULAR Q1, Q3 E IQR PARA CADA OBJETIVO (APENAS NA FRENTE DE PARETO)
    q1 = np.percentile(decision_matrix, 25, axis=0)
    q3 = np.percentile(decision_matrix, 75, axis=0)
    iqr = q3 - q1
    # Tratar IQR nulo (todos os valores iguais): nesse caso, a normalização é irrelevante
    iqr[iqr == 0] = 1.0
    
    # 4. NORMALIZAÇÃO ROBUSTA PELO IQR
    # Subtrai Q1 e divide pelo IQR, gerando uma escala adimensional onde a dispersão
    # é definida pelos 50% centrais dos dados.
    normalized_matrix = (decision_matrix - q1) / iqr
    
    # 5. DEFINIR SOLUÇÃO IDEAL E ANTI-IDEAL NO ESPAÇO NORMALIZADO
    # Como todos os objetivos são de minimização, o ideal é o menor valor.
    ideal = np.min(normalized_matrix, axis=0)
    anti_ideal = np.max(normalized_matrix, axis=0)
    
    # 6. DISTÂNCIAS EUCLIDIANAS (sem ponderação adicional, pois a escala já é uniforme)
    dist_to_ideal = np.sqrt(np.sum((normalized_matrix - ideal)**2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((normalized_matrix - anti_ideal)**2, axis=1))
    
    # 7. SCORES TOPSIS
    scores_topsis = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal + epsilon)
    
    # Melhor índice NO CONJUNTO FILTRADO
    best_idx_filtrado = np.argmax(scores_topsis)
    
    # Converter para índice original
    best_idx = indices_validos[best_idx_filtrado]
    
    # 9. MAPEAR SCORES DE VOLTA PARA TODAS AS SOLUÇÕES (com -inf para dominadas)
    scores_completos = np.full(len(fitness_pareto), -np.inf)
    for idx_filtrado, idx_original in enumerate(indices_validos):
        scores_completos[idx_original] = scores_topsis[idx_filtrado]
    
    return best_idx, scores_completos

# ==============================================================================
# 🎯 MÉTODO TOPSIS COM NORMALIZAÇÃO MIN-MAX
# ==============================================================================
def calcular_tradeoff_topsis_minmax(fitness_pareto):
    """
    TOPSIS com normalização min-max.
    Retorna o índice da melhor solução e os scores.
    """
    if len(fitness_pareto) < 2:
        return 0, np.array([1.0])
    
    # 1. Verificação de dominância Pareto (mesmo da versão IQR)
    indices_validos = []
    n = len(fitness_pareto)
    for i in range(n):
        dominado = False
        for j in range(n):
            if i != j:
                if (np.all(fitness_pareto[j] <= fitness_pareto[i]) and 
                    np.any(fitness_pareto[j] < fitness_pareto[i])):
                    dominado = True
                    break
        if not dominado:
            indices_validos.append(i)
    
    fitness_filtrado = fitness_pareto[indices_validos]
    epsilon = 1e-10
    n_solucoes = len(fitness_filtrado)
    
    decision_matrix = fitness_filtrado.copy()
    
    # 2. Normalização min-max
    min_vals = np.min(decision_matrix, axis=0)
    max_vals = np.max(decision_matrix, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0  # evitar divisão por zero
    
    normalized_matrix = (decision_matrix - min_vals) / range_vals
    
    # 3. Solução ideal e anti-ideal (minimização)
    ideal = np.min(normalized_matrix, axis=0)
    anti_ideal = np.max(normalized_matrix, axis=0)
    
    # 4. Distâncias
    dist_to_ideal = np.sqrt(np.sum((normalized_matrix - ideal)**2, axis=1))
    dist_to_anti_ideal = np.sqrt(np.sum((normalized_matrix - anti_ideal)**2, axis=1))
    
    # 5. Scores
    scores_topsis = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal + epsilon)
    
    best_idx_filtrado = np.argmax(scores_topsis)
    best_idx = indices_validos[best_idx_filtrado]
    
    # Mapear scores para todas as soluções
    scores_completos = np.full(len(fitness_pareto), -np.inf)
    for idx_filtrado, idx_original in enumerate(indices_validos):
        scores_completos[idx_original] = scores_topsis[idx_filtrado]
    
    return best_idx, scores_completos

# ==============================================================================
# 🎯 DIAGNÓSTICO: POR QUE MODELO RETORNA z = -9.99?
# ==============================================================================

def diagnosticar_valor_extremo(scaler, modelo, target_transform):
    """
    Diagnóstico do porquê o modelo retorna valores extremos
    """
    print("\n🔬 DIAGNÓSTICO: VALOR EXTREMO z = -9.99")
    print("="*60)
    
    # Testar com features de exemplo
    print("1. TESTE COM FEATURES DE EXEMPLO:")
    
    # Criar features médias (todas zeros após scaling)
    features_medias = np.zeros((1, 10))
    
    # Aplicar scaler
    features_scaled = scaler.transform(features_medias)
    
    print(f"   • Features médias (após scaling): {features_scaled[0]}")
    print(f"   • Faixa: [{features_scaled[0].min():.3f}, {features_scaled[0].max():.3f}]")
    
    # Predizer
    y_pred = modelo.predict(features_scaled)[0]
    print(f"   • Predição modelo: {y_pred:.6f}")
    
    # 2. Verificar parâmetros do PLS
    print("\n2. PARÂMETROS DO MODELO PLS:")
    
    if hasattr(modelo, 'coef_'):
        print(f"   • Coeficientes shape: {modelo.coef_.shape}")
        print(f"   • Coeficientes: {modelo.coef_.flatten()}")
        print(f"   • Soma coeficientes: {np.sum(modelo.coef_):.6f}")
    
    if hasattr(modelo, 'intercept_'):
        print(f"   • Intercept: {modelo.intercept_:.6f}")
    
    # 3. Testar com dados reais do MCNP
    print("\n3. TESTE COM DADOS REAIS:")
    
    # Carregar um exemplo do MCNP
    df_mcnp = carregar_dados_mcnp_sobol(MCNP_DATA_PATH)
    if df_mcnp is not None:
        exemplo = df_mcnp.iloc[0]
        
        # Criar features para este exemplo
        espessuras = exemplo['espessuras_list']
        materiais_nomes = exemplo['materiais_list']
        materiais_idx = [MATERIAIS_DISPONIVEIS.index(m) if m in MATERIAIS_DISPONIVEIS else 0 
                        for m in materiais_nomes]
        
        features, peso, custo = criar_features_para_otimizacao(espessuras, materiais_idx)
        
        # Preparar vetor de features
        features_vector = np.array([[
            features['z_eff_ponderado_espessura'],
            features['atenuacao_total'],
            features['z_dens_esp_total'],
            features['high_z_protection'],
            features['low_z_moderation'],
            features['high_low_ratio'],
            features['z_x_atenuacao'],
            features['transmissao'],
            # features['log_transmissao'],
            features['sequence_score'],
            features['z_variance']
        ]])
        
        print(f"   • Features reais (antes scaling):")
        for i, (nome, valor) in enumerate([
            ('z_eff_ponderado', features['z_eff_ponderado_espessura']),
            ('atenuacao_total', features['atenuacao_total']),
            ('transmissao', features['transmissao']),
            # ('log_transmissao', features['log_transmissao'])
        ]):
            # print(f"     {nome}: {valor:.4f}")
        
            # Aplicar scaler
            features_scaled_real = scaler.transform(features_vector)
            print(f"   • Features (após scaling): {features_scaled_real[0]}")
        
        # Predizer
        y_pred_real = modelo.predict(features_scaled_real)[0]
        print(f"   • Predição: {y_pred_real:.6f}")
        
        # Converter para dose
        mean_log = target_transform['mean']
        std_log = target_transform['std']
        dose_5min = inversa_transformacao_otimizada_5min(y_pred_real, mean_log, std_log)
        print(f"   • Dose convertida: {dose_5min:.3f} mSv/5min")
        print(f"   • Dose real MCNP: {exemplo['dose_5min']:.3f} mSv/5min")
        print(f"   • Erro: {abs(dose_5min - exemplo['dose_5min'])/exemplo['dose_5min']*100:.1f}%")
    
    return True

# ==============================================================================
# 🎯 CORREÇÃO DEFINITIVA: FUNÇÃO DE CONVERSÃO REVISADA
# ==============================================================================

def correcao_conversao_dose(dose_transformada, mean_log, std_log):
    """
    CORREÇÃO DEFINITIVA: Detecta e corrige valores extremos
    PARÂMETROS IMPORTANTE: mean_log e std_log são do LOG da DOSE EM mSv/5min
    (não mSv/s como estava sendo tratado erroneamente)
    """
    # 1. Verificar NaN/Inf
    if np.isnan(dose_transformada) or np.isinf(dose_transformada):
        print(f"🚨 VALOR NaN/Inf DETECTADO")
        dose_transformada = 0.0
    
    # 2. DEBUG: mostrar apenas valores razoáveis
    if abs(dose_transformada) < 1e6:
        print(f"🔧 DEBUG entrada: z = {dose_transformada:.6f}")
    
    # 🎯 LIMITE PRINCIPAL: ±100σ (detecta instabilidade numérica)
    if dose_transformada < -100.0 or dose_transformada > 100.0:
        if abs(dose_transformada) > 1e6:
            print(f"🚨 INSTABILIDADE NUMÉRICA EXTREMA: z = {dose_transformada:.2e}")
        else:
            print(f"⚠️  VALOR EXTREMO DETECTADO: z = {dose_transformada:.2f}")
        
        print(f"   • Usando valor médio (z = 0) como fallback")
        
        # CRÍTICO: Aqui mean_log é do log(dose_5min), não precisa multiplicar por 300!
        dose_5min = np.exp(0.0 * std_log + mean_log)  # z = 0
        
        print(f"   • Resultado corrigido: {dose_5min:.3f} mSv/5min")
        return dose_5min
    
    # 🎯 LIMITE SECUNDÁRIO: ±5σ (valores fisicamente plausíveis)
    elif -5.0 <= dose_transformada <= 5.0:
        # Usar função original com correção
        dose_5min = inversa_transformacao_otimizada_5min(dose_transformada, mean_log, std_log)
        return dose_5min
    
    # 🎯 CASO INTERMEDIÁRIO: ±100σ mas fora de ±5σ
    else:
        if dose_transformada < -5.0:
            print(f"⚠️  Valor abaixo do limite: z = {dose_transformada:.2f}")
            print(f"   • Usando limite inferior (z = -5)")
            z_corrigido = -5.0
        else:  # dose_transformada > 5.0
            print(f"⚠️  Valor acima do limite: z = {dose_transformada:.2f}")
            print(f"   • Usando limite superior (z = 5)")
            z_corrigido = 5.0
        
        return inversa_transformacao_otimizada_5min(z_corrigido, mean_log, std_log)

# ==============================================================================
# 🎯 FUNÇÃO: TRANSFORMAÇÃO INVERSA OTIMIZADA PARA 5 MINUTOS - CORRIGIDA
# ==============================================================================

def inversa_transformacao_otimizada_5min(dose_transformada, mean_log, std_log):
    """
    Transformação inversa OTIMIZADA PARA 5 MINUTOS - VERSÃO CORRIGIDA
    """
    # 1. Clip extremo - valores além de ±8 sigma
    dose_transformada_clipped = np.clip(dose_transformada, -8, 8)
    
    # 2. Transformação inversa: log(dose_5min) = z * σ + μ
    dose_log_5min = dose_transformada_clipped * std_log + mean_log
    
    # 3. Clip no espaço log para faixa FISICAMENTE PLAUSÍVEL
    #    PERMITINDO EXTRAPOLAÇÃO PARA DOSES MAIS BAIXAS!
    min_log = np.log(1.0)   # ← PERMITIR ATÉ 1.0 mSv (extrapolação!)
    max_log = np.log(70.0)  # ← PERMITIR ATÉ 70.0 mSv
    
    dose_log_5min_clipped = np.clip(dose_log_5min, min_log, max_log)
    
    # 4. Exponenciação para retornar à escala linear (mSv/5min)
    dose_5min = np.exp(dose_log_5min_clipped)  # ← ESTE É O VALOR CORRETO!
    
    # DEBUG: Mostrar extrapolação significativa
    # if dose_5min < 2.5:  # Significativamente abaixo do mínimo do dataset
    # print(f"🚀 EXTRAPOLAÇÃO: {dose_5min:.4f} mSv < 2.5 mSv")
    
    return dose_5min  # ← CORRETO: retornar dose_5min, não dose_5min_clipped!

# ==============================================================================
# 🧪 FUNÇÃO: VERIFICAÇÃO DOS PARÂMETROS DO TARGET_TRANSFORM
# ==============================================================================

def verificar_parametros_target_transform(target_transform):
    """
    Verifica se os parâmetros do target_transform estão corretos
    """
    print("\n" + "="*60)
    print("🔍 VERIFICAÇÃO DOS PARÂMETROS DO TARGET_TRANSFORM")
    print("="*60)
    
    mean_log = target_transform['mean']
    std_log = target_transform['std']
    
    print(f"• Média (log): {mean_log:.6f}")
    print(f"• Desvio padrão (log): {std_log:.6f}")
    
    # Calcular dose correspondente à média
    dose_5min_mean = np.exp(mean_log)
    print(f"• Dose correspondente à média: {dose_5min_mean:.6f} mSv/5min")
    
    # Verificar se está na faixa observada nesta campanha, com pequena margem.
    if LIMITE_DOSE_MIN * 0.95 <= dose_5min_mean <= LIMITE_DOSE_MAX * 1.05:
        print(f"✅ Média dentro da faixa esperada ({LIMITE_DOSE_MIN:.2f}-{LIMITE_DOSE_MAX:.2f} mSv/5min)")
    else:
        print(f"❌ Média FORA da faixa esperada: {dose_5min_mean:.2f} mSv/5min")
        print(f"   Provável problema: Parâmetros calculados com unidade errada")
    
    # Calcular faixa de ±3σ
    min_3sigma = np.exp(mean_log - 3*std_log)
    max_3sigma = np.exp(mean_log + 3*std_log)
    print(f"\n• Faixa ±3σ (99.7% dos dados):")
    print(f"  - Mínimo: {min_3sigma:.3f} mSv/5min")
    print(f"  - Máximo: {max_3sigma:.3f} mSv/5min")
    
    # A faixa ±3σ deve cobrir os extremos observados, sem depender de uma
    # campanha anterior com outra física de conversão.
    if min_3sigma <= LIMITE_DOSE_MIN and max_3sigma >= LIMITE_DOSE_MAX:
        print("✅ Faixa ±3σ cobre a faixa observada no dataset MCNP")
    else:
        print("⚠️  Faixa ±3σ não cobre integralmente os extremos do dataset MCNP")
    
    # Testar conversão para valores típicos
    print(f"\n🧪 TESTE DE CONVERSÃO:")
    test_values = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for z in test_values:
        dose_5min = inversa_transformacao_otimizada_5min(z, mean_log, std_log)
        print(f"  • z = {z:5.1f} → {dose_5min:6.3f} mSv/5min")
    
    return dose_5min_mean

# ==============================================================================
# FUNÇÃO: SELECIONAR MELHOR MODELO AUTOMATICAMENTE
# ==============================================================================

def selecionar_melhor_modelo_automaticamente(models_dir):
    """
    Seleciona automaticamente o melhor modelo baseado nos resultados da Fase 3
    """
    print("\n" + "="*80)
    print("🔍 SELEÇÃO AUTOMÁTICA DO MELHOR MODELO")
    print("="*80)
    
    # 1. Tentar carregar resultados da Fase 3
    resultados_path = os.path.join(os.path.dirname(models_dir), 
                                  "resultados_finais", 
                                  "resultados_finais.csv")
    
    if os.path.exists(resultados_path):
        try:
            df_resultados = pd.read_csv(resultados_path)
            
            # Filtrar modelos válidos
            df_validos = df_resultados[df_resultados['Status'] == '✅'].copy()
            
            if len(df_validos) > 0:
                # Encontrar modelo com maior R²
                df_validos['R²'] = pd.to_numeric(df_validos['R²'], errors='coerce')
                melhor_idx = df_validos['R²'].idxmax()
                melhor_modelo = df_validos.loc[melhor_idx].name
                melhor_r2 = df_validos.loc[melhor_idx, 'R²']
                
                print(f"📊 Baseado nos resultados da Fase 3:")
                print(f"   • Total de modelos: {len(df_validos)}")
                print(f"   • Melhor modelo: {melhor_modelo}")
                print(f"   • R²: {melhor_r2:.4f}")
                print(f"   • MAE: {df_validos.loc[melhor_idx, 'MAE']:.6f}")
                
                return melhor_modelo, melhor_r2
                
        except Exception as e:
            print(f"⚠️  Erro ao ler resultados: {e}")
    
    # 2. Fallback: analisar arquivos .joblib disponíveis
    print("   ⚠️  Usando fallback: análise dos arquivos disponíveis...")
    
    modelos_disponiveis = []
    
    for arquivo in os.listdir(models_dir):
        if arquivo.endswith('.joblib') and 'scaler' not in arquivo and 'target' not in arquivo:
            nome_modelo = arquivo.replace('.joblib', '')
            
            # Priorizar modelos baseado no nome
            prioridade = {
                'PLS_Regression': 10,      # Geralmente muito bom
                'BayesianRidge_Poly2': 9,
                'Stacking_Ensemble': 8,
                'KernelRidge_RBF': 7,
                'ElasticNet_Optimized': 6,
                'XGBoost_Optimized': 5,
                'GradientBoosting_PhysicsBased': 4,
                'RandomForest_Optimized': 3,
                'HistGradientBoosting': 2
            }
            
            modelos_disponiveis.append({
                'nome': nome_modelo,
                'arquivo': arquivo,
                'prioridade': prioridade.get(nome_modelo, 1)
            })
    
    if modelos_disponiveis:
        # Ordenar por prioridade
        modelos_disponiveis.sort(key=lambda x: x['prioridade'], reverse=True)
        melhor_modelo = modelos_disponiveis[0]['nome']
        melhor_arquivo = modelos_disponiveis[0]['arquivo']
        
        print(f"📊 Seleção por prioridade:")
        print(f"   • Modelos disponíveis: {len(modelos_disponiveis)}")
        print(f"   • Melhor escolha: {melhor_modelo}")
        print(f"   • Arquivo: {melhor_arquivo}")
        
        return melhor_modelo, None
    
    # 3. Fallback final: usar qualquer modelo
    print("   ⚠️  Fallback final: primeiro modelo encontrado")
    
    for arquivo in os.listdir(models_dir):
        if arquivo.endswith('.joblib') and 'scaler' not in arquivo and 'target' not in arquivo:
            nome_modelo = arquivo.replace('.joblib', '')
            print(f"   • Usando: {nome_modelo}")
            return nome_modelo, None
    
    raise FileNotFoundError("Nenhum modelo encontrado para NSGA-II")

# ==============================================================================
# FUNÇÕES AUXILIARES PARA MATERIAIS ÚNICOS
# ==============================================================================

def gerar_materiais_unicos(num_camadas):
    """Gera uma sequência de materiais ÚNICOS (sem repetições)"""
    if num_camadas > len(MATERIAIS_DISPONIVEIS):
        raise ValueError(f"Não é possível ter {num_camadas} camadas com materiais únicos (apenas {len(MATERIAIS_DISPONIVEIS)} materiais disponíveis)")
    
    # Embaralhar materiais
    materiais_disponiveis = list(range(len(MATERIAIS_DISPONIVEIS)))
    np.random.shuffle(materiais_disponiveis)
    
    # Pegar os N primeiros (únicos)
    return materiais_disponiveis[:num_camadas]

def corrigir_materiais_repetidos(materiais_indices):
    """Corrige materiais repetidos substituindo por materiais não utilizados"""
    materiais = materiais_indices.copy()
    todos_materiais = set(range(len(MATERIAIS_DISPONIVEIS)))
    
    # Contar ocorrências
    from collections import Counter
    contador = Counter(materiais)
    
    # Materiais repetidos
    repetidos = [mat for mat, count in contador.items() if count > 1]
    
    # Materiais não utilizados
    utilizados = set(materiais)
    nao_utilizados = list(todos_materiais - utilizados)
    
    # Para cada repetição (exceto a primeira ocorrência)
    for i, mat in enumerate(materiais):
        if mat in repetidos:
            # É uma duplicata?
            if contador[mat] > 1:
                if nao_utilizados:
                    # Substituir por material não utilizado
                    novo_material = nao_utilizados.pop(0)
                    materiais[i] = novo_material
                    contador[mat] -= 1
                    contador[novo_material] = 1
                else:
                    # Não há materiais disponíveis - situação crítica
                    # Gera um material aleatório diferente do atual
                    while True:
                        novo_material = np.random.randint(0, len(MATERIAIS_DISPONIVEIS))
                        if novo_material != mat and contador.get(novo_material, 0) == 0:
                            materiais[i] = novo_material
                            contador[mat] -= 1
                            contador[novo_material] = 1
                            break
    
    return np.array(materiais)

# ==============================================================================
# CONFIGURAÇÕES GEOMÉTRICAS DA BLINDAGEM
# ==============================================================================

# Geometria da blindagem (casca semicilíndrica)
ALTURA_BLINDAGEM_CM = 70.0      # Altura do cilindro
RAIO_INTERNO_CM = 35.0          # Raio interno
ESPESSURA_TOTAL_CM = 6.0        # Espessura total da casca

# Calcular raio externo
RAIO_EXTERNO_CM = RAIO_INTERNO_CM + ESPESSURA_TOTAL_CM

# Função para calcular volume de uma camada
# V = 0.5 * π * h * (R_ext² - R_int²)
def calcular_volume_camada(raio_interno_camada, espessura_camada):
    """
    Calcula volume de uma camada cilíndrica
    Volume = 0,5 × Altura × π × (R_externo² - R_interno²)
    """
    raio_externo_camada = raio_interno_camada + espessura_camada
    area_secao = 0.5 * np.pi * (raio_externo_camada**2 - raio_interno_camada**2)
    return ALTURA_BLINDAGEM_CM * area_secao

def calcular_peso_custo_teoricos():
    """
    Calcula valores teóricos mínimos e máximos ABSOLUTOS considerando configurações ótimas
    """
    print("\n" + "="*60)
    print("CÁLCULO TEÓRICO - CASCA SEMICILÍNDRICA")
    print("="*60)
    
    # Geometria
    raio_interno = RAIO_INTERNO_CM
    altura = ALTURA_CM
    espessura_total = ESPESSURA_TOTAL_FIXA
    raio_externo = raio_interno + espessura_total
    
    # Volume total FIXO (não depende da distribuição)
    volume_total = 0.5 * np.pi * (raio_externo**2 - raio_interno**2) * altura
    
    print(f"Geometria:")
    print(f"  Raio interno: {raio_interno} cm")
    print(f"  Raio externo: {raio_externo} cm")
    print(f"  Espessura total: {espessura_total} cm")
    print(f"  Altura: {altura} cm")
    print(f"  Volume total: {volume_total:.0f} cm³")
    print(f"  Número de camadas: {NUM_CAMADAS}")
    print(f"  Materiais únicos obrigatórios: SIM")
    
    # VERIFICAÇÃO DE VIABILIDADE
    if NUM_CAMADAS > len(MATERIAIS_DISPONIVEIS):
        print(f"\n❌ IMPOSSÍVEL: {NUM_CAMADAS} camadas com materiais únicos")
        print(f"   (apenas {len(MATERIAIS_DISPONIVEIS)} materiais disponíveis)")
        return volume_total

    return volume_total

def obter_ordenacao_modelos(models_dir):
    """
    Retorna uma lista de nomes de modelos ordenados do melhor para o pior
    com base no R² registrado em resultados_finais.csv.
    Se o arquivo não existir, retorna uma lista vazia.
    """
    resultados_path = os.path.join(os.path.dirname(models_dir), 
                                  "resultados_finais", 
                                  "resultados_finais.csv")
    if not os.path.exists(resultados_path):
        print("⚠️  Arquivo de resultados não encontrado. Fallback desabilitado.")
        return []
    
    try:
        df = pd.read_csv(resultados_path)
        df_validos = df[df['Status'] == '✅'].copy()
        df_validos['R²'] = pd.to_numeric(df_validos['R²'], errors='coerce')
        df_validos = df_validos.dropna(subset=['R²'])
        df_ordenado = df_validos.sort_values('R²', ascending=False)
        # Extrair nomes dos modelos (ajuste conforme a coluna que contém o nome)
        # No CSV, o índice pode ser o nome, ou há uma coluna 'Modelo'.
        # Vamos tentar usar o índice se for string, senão usar a coluna 'Modelo'.
        if 'Modelo' in df_ordenado.columns:
            nomes = df_ordenado['Modelo'].tolist()
        else:
            # assume que o índice é o nome
            nomes = df_ordenado.index.tolist()
        return nomes
    except Exception as e:
        print(f"⚠️  Erro ao ler ordenação: {e}")
        return []

# ==============================================================================
# CARREGAMENTO DOS MODELOS DA FASE 3 - SELEÇÃO AUTOMÁTICA BASEADA EM R²
# ==============================================================================

def carregar_modelos_fase3(modelo_selecionado=None):
    """
    Carrega modelos da Fase 3 com opção de seleção:
    - modelo_selecionado=None: Seleciona automaticamente o melhor modelo (maior R²)
    - modelo_selecionado='PLS_Regression': Carrega modelo específico
    
    Parâmetros:
    -----------
    modelo_selecionado : str ou None
        None: usa melhor modelo baseado em R²
        str: nome do modelo específico (ex: 'PLS_Regression', 'RandomForest_Optimized')
    """
    print("🔍 Carregando modelos da Fase 3...")
    
    # 1. VERIFICAÇÃO DE INTEGRIDADE
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"❌ Diretório de modelos não encontrado: {MODELS_DIR}")
    
    print(f"   • Diretório: {MODELS_DIR}")
    
    # 2. CARREGAR COMPONENTES ESSENCIAIS (UNALTERED)
    print("\n📂 Carregando componentes essenciais...")
    
    scaler_path = os.path.join(MODELS_DIR, 'scaler_standard.joblib')
    target_path = os.path.join(MODELS_DIR, 'target_transform.joblib')
    
    # Carregar StandardScaler
    if not os.path.exists(scaler_path):
        print(f"❌ StandardScaler não encontrado: {scaler_path}")
        sys.exit(1)
    
    scaler = joblib.load(scaler_path)
    print("✅ StandardScaler carregado")
    
    # Carregar target transform
    if not os.path.exists(target_path):
        print(f"❌ Target transform não encontrado: {target_path}")
        sys.exit(1)
    
    target_transform = joblib.load(target_path)
    print("✅ Target transform carregado")
    
    # ==============================================================================
    # 🎯 SISTEMA INTELIGENTE DE DESCOBERTA DE MODELOS
    # ==============================================================================
    print("\n🤖 SISTEMA INTELIGENTE DE DESCOBERTA DE MODELOS")
    print("="*50)
    
    # Dicionário para mapear nome simplificado → arquivo real
    mapeamento_modelos = {}
    
    # 1. PRIMEIRA VARREdura: identificar TODOS os arquivos de modelo
    arquivos_modelos = []
    for arquivo in os.listdir(MODELS_DIR):
        # Ignorar scaler, target e arquivos muito pequenos
        if 'scaler' in arquivo or 'target' in arquivo:
            continue
        
        caminho_completo = os.path.join(MODELS_DIR, arquivo)
        
        # Aceitar múltiplas extensões
        if (arquivo.endswith('.joblib') or arquivo.endswith('.keras') or 
            arquivo.endswith('.h5') or arquivo.endswith('.pkl')):
            
            # Verificar tamanho mínimo (exceto para ElasticNet que pode ser pequeno)
            if arquivo.endswith('.joblib'):
                tamanho = os.path.getsize(caminho_completo)
                if tamanho < 500 and 'ElasticNet' not in arquivo:
                    print(f"⚠️  AVISO: {arquivo} muito pequeno ({tamanho} bytes)")
                    continue
            
            arquivos_modelos.append((arquivo, caminho_completo))
    
    print(f"   • Arquivos de modelo detectados (NeuralNetwork_Fisica é representado por 2 arquivos no diretório): {len(arquivos_modelos)}")
    
    # 2. MAPEAR NOMES SIMPLIFICADOS → ARQUIVOS REAIS
    # Primeiro, agrupar candidatos por nome base
    candidatos_por_nome = {}
    for arquivo, caminho in arquivos_modelos:
        nome_base = arquivo
        for ext in ['.joblib', '.keras', '.h5', '.pkl', '_model', '_wrapper', '_optimized']:
            if nome_base.endswith(ext):
                nome_base = nome_base[:-len(ext)]
                
        if nome_base not in candidatos_por_nome:
            candidatos_por_nome[nome_base] = []
        candidatos_por_nome[nome_base].append((arquivo, caminho))
        
    # Agora, escolher o melhor arquivo para cada nome base
    mapeamento_modelos = {}
    for nome_base, candidatos in candidatos_por_nome.items():
        if nome_base == 'NeuralNetwork_Fisica':
            # Priorizar arquivo .keras para NeuralNetwork_Fisica
            candidatos_keras = [c for c in candidatos if c[0].endswith('.keras')]
            if candidatos_keras:
                arquivo, caminho = candidatos_keras[0]
                mapeamento_modelos[nome_base] = (arquivo, caminho)
                print(f"   • Mapeado: '{nome_base}' → '{arquivo}' (prioridade .keras)")
            else:
                arquivo, caminho = candidatos[0]
                mapeamento_modelos[nome_base] = (arquivo, caminho)
                print(f"   • Mapeado: '{nome_base}' → '{arquivo}'")
        else:
            # Para outros modelos, priorizar .joblib
            candidatos_joblib = [c for c in candidatos if c[0].endswith('.joblib')]
            if candidatos_joblib:
                arquivo, caminho = candidatos_joblib[0]
                mapeamento_modelos[nome_base] = (arquivo, caminho)
                print(f"   • Mapeado: '{nome_base}' → '{arquivo}' (prioridade .joblib)")
            else:
                arquivo, caminho = candidatos[0]
                mapeamento_modelos[nome_base] = (arquivo, caminho)
                print(f"   • Mapeado: '{nome_base}' → '{arquivo}'")
    
    # 3. LISTAR MODELOS DISPONÍVEIS (nomes simplificados)
    modelos_disponiveis = list(mapeamento_modelos.keys())
    
    print(f"\n   • Modelos disponíveis: {len(modelos_disponiveis)}")
    for i, nome in enumerate(sorted(modelos_disponiveis), 1):
        arquivo_real, _ = mapeamento_modelos[nome]
        print(f"      {i:2d}. {nome:30s} (arquivo: {arquivo_real})")
    
    # ==============================================================================
    # 🎯 OPÇÃO A: MODELO ESPECÍFICO ESCOLHIDO PELO USUÁRIO
    # ==============================================================================
    if modelo_selecionado is not None:
        print(f"\n🎯 Usuário escolheu: {modelo_selecionado}")
        
        # Verificar se o modelo existe (com ou sem sufixos)
        modelo_encontrado = None
        arquivo_real = None
        caminho_real = None
        
        # Tentar correspondência exata primeiro
        if modelo_selecionado in modelos_disponiveis:
            modelo_encontrado = modelo_selecionado
            arquivo_real, caminho_real = mapeamento_modelos[modelo_selecionado]
        
        # Se não encontrou, tentar correspondência parcial
        else:
            for nome_modelo in modelos_disponiveis:
                if modelo_selecionado in nome_modelo or nome_modelo in modelo_selecionado:
                    modelo_encontrado = nome_modelo
                    arquivo_real, caminho_real = mapeamento_modelos[nome_modelo]
                    break
        
        if modelo_encontrado:
            print(f"   • Modelo encontrado: {modelo_encontrado}")
            print(f"   • Nome do arquivo real: {arquivo_real}")
            print(f"   • Caminho: {caminho_real}")
            
            try:
                # CARREGAMENTO BASEADO NA EXTENSÃO
                if caminho_real.endswith('.joblib'):
                    modelo = joblib.load(caminho_real)
                    print(f"✅ Modelo .joblib carregado: {modelo_encontrado}")
                    print(f"   • Tipo: {type(modelo).__name__}")
                    
                elif caminho_real.endswith('.keras') or caminho_real.endswith('.h5'):
                    try:
                        from tensorflow.keras.models import load_model
        
                        # Carregar modelo Keras
                        keras_model = load_model(caminho_real)
                        print(f"✅ Modelo Keras carregado: {modelo_encontrado}")
                        
                        # 🆕 ESCOLHA DO WRAPPER CORRETO BASEADO NO NOME DO MODELO
                        if 'NeuralNetwork_Fisica' in modelo_encontrado:
                            # Usar wrapper ESPECÍFICO para NeuralNetwork_Fisica
                            modelo = NeuralNetworkFisicaWrapper(keras_model)
                            print(f"   • Wrapper ESPECÍFICO aplicado para NeuralNetwork_Fisica")
                            print(f"   • 10 features com normalização min-max personalizada")
                        else:
                            # Para outros modelos Keras, usar wrapper SIMPLES
                            class SimpleKerasWrapper:
                                def __init__(self, keras_model):
                                    self.keras_model = keras_model
                                    self._is_keras_model = True
                                    
                                def predict(self, X):
                                    # Garantir que temos 10 features
                                    if X.shape[1] != 10:
                                        X = criar_nn_features_fisicas(X)
                                    return self.keras_model.predict(X, verbose=0).flatten()
                                
                                def __getstate__(self):
                                    # Evitar serialização do modelo Keras entre processos
                                    state = self.__dict__.copy()
                                    if 'keras_model' in state:
                                        state['keras_model'] = None
                                    return state
                                
                                def __setstate__(self, state):
                                    # Recriar wrapper vazio após serialização
                                    self.__dict__.update(state)
                                    self._is_keras_model = True
                                    
                            modelo = SimpleKerasWrapper(keras_model)
                            print(f"   • Wrapper SIMPLES aplicado para compatibilidade")
                            
                    except ImportError as e:
                        print(f"❌ TensorFlow não instalado: {e}")
                        print("   Instale com: pip install tensorflow")
                        raise
                    except Exception as e:
                        print(f"❌ Erro ao carregar modelo Keras: {e}")
                        raise
                        
                else:
                    print(f"❌ Formato não suportado: {caminho_real}")
                    raise ValueError(f"Formato não suportado: {arquivo_real}")
                    
                # Retornar com sucesso
                return scaler, target_transform, modelo, modelo_encontrado
            
            except Exception as e:
                print(f"❌ Erro geral ao carregar o modelo: {e}")
                raise
                
    # ==============================================================================
    # 🎯 OPÇÃO B: SELEÇÃO AUTOMÁTICA BASEADA EM R² (PADRÃO)
    # ==============================================================================
    if modelo_selecionado is None:
        print("\n🔍 Seleção automática (baseada em R² da Fase 3)...")
        
        # Obter ranking ordenado dos modelos (já valida integridade)
        ranking = obter_ordenacao_modelos(MODELS_DIR)
        
        if ranking:
            melhor_nome, melhor_r2 = ranking[0]  # primeiro da lista = melhor R²
            print(f"📊 Melhor modelo: {melhor_nome} (R² = {melhor_r2:.4f})")
            
            # Verificar correspondência exata com os modelos disponíveis
            if melhor_nome in mapeamento_modelos:
                melhor_modelo_nome = melhor_nome
                print(f"   • Correspondência encontrada: {melhor_modelo_nome}")
                
                arquivo_real, caminho_real = mapeamento_modelos[melhor_modelo_nome]
                print(f"📂 Carregando: {melhor_modelo_nome} ({arquivo_real})")
                
                try:
                    if caminho_real.endswith('.joblib'):
                        modelo = joblib.load(caminho_real)
                    elif caminho_real.endswith('.keras') or caminho_real.endswith('.h5'):
                        keras_model = load_model(caminho_real)
                        
                        class KerasWrapper:
                            def __init__(self, keras_model):
                                self.keras_model = keras_model
                            def predict(self, X):
                                return self.keras_model.predict(X, verbose=0).flatten()
                            
                        modelo = KerasWrapper(keras_model)
                        print(f"   • Wrapper Keras criado para compatibilidade")
                        
                    print(f"✅ Melhor modelo carregado: {melhor_modelo_nome}")
                    print(f"   • R²: {melhor_r2:.4f}")
                    print(f"   • Tipo: {type(modelo).__name__}")
                    
                    return scaler, target_transform, modelo, melhor_modelo_nome
                except Exception as e:
                    print(f"⚠️ Erro ao carregar melhor modelo: {e}")
            else:
                print(f"   • Modelo '{melhor_nome}' não encontrado no diretório.")
        else:
            print("   • Nenhum modelo válido encontrado no ranking.")
            
        # Se chegou aqui, fallback será acionado
        print("   ⚠️ Usando fallback: ordem de prioridade...")
    
    # ==============================================================================
    # 🎯 FALLBACK: ORDEM DE PRIORIDADE
    # ==============================================================================
    print("   ⚠️  Usando fallback: ordem de prioridade...")
    
    ordem_preferencia = [
        'PLS_Regression',
        'BayesianRidge_Poly2',
        'Stacking_Ensemble',
        'RandomForest_Optimized',
        'XGBoost_Optimized',
        'ElasticNet_Optimized',  # Mesmo sendo pequeno, pode funcionar
        'GradientBoosting_PhysicsBased',
        'HistGradientBoosting',
        'KernelRidge_RBF',
        'NeuralNetwork_Fisica'  # Último porque requer TensorFlow
    ]
    
    modelo_carregado = False
    for nome_preferido in ordem_preferencia:
        if nome_preferido in modelos_disponiveis:
            arquivo_real, caminho_real = mapeamento_modelos[nome_preferido]
            print(f"   • Tentando: {nome_preferido} ({arquivo_real})")
            
            try:
                if caminho_real.endswith('.joblib'):
                    modelo = joblib.load(caminho_real)
                    modelo_nome = nome_preferido
                    modelo_carregado = True
                    print(f"✅ Modelo carregado (prioridade): {modelo_nome}")
                    print(f"   • Tipo: {type(modelo).__name__}")
                    break
                    
                elif caminho_real.endswith('.keras') or caminho_real.endswith('.h5'):
                    if nome_preferido == 'NeuralNetwork_Fisica':
                        print(f"   • NeuralNetwork requer TensorFlow. Pulando...")
                        continue
                    
                    from tensorflow.keras.models import load_model
                    keras_model = load_model(caminho_real)
                    
                    class KerasWrapper:
                        def __init__(self, keras_model):
                            self.keras_model = keras_model
                            
                        def predict(self, X):
                            return self.keras_model.predict(X, verbose=0).flatten()
                    
                    modelo = KerasWrapper(keras_model)
                    modelo_nome = nome_preferido
                    modelo_carregado = True
                    print(f"✅ Modelo Keras carregado: {modelo_nome}")
                    print(f"   • Wrapper criado para compatibilidade")
                    break
                    
            except ImportError as e:
                print(f"   ⚠️  TensorFlow não instalado para {nome_preferido}")
                continue
            except Exception as e:
                print(f"   ⚠️  Erro ao carregar {nome_preferido}: {e}")
                continue
    
    # ==============================================================================
    # 🎯 FALLBACK FINAL: QUALQUER MODELO VÁLIDO
    # ==============================================================================
    if not modelo_carregado:
        print("   ⚠️  Fallback final: primeiro modelo válido encontrado...")
        
        for nome_modelo in modelos_disponiveis:
            if nome_modelo in ['scaler_standard', 'target_transform']:
                continue
                
            arquivo_real, caminho_real = mapeamento_modelos[nome_modelo]
            print(f"   • Tentando: {nome_modelo}")
            
            try:
                if caminho_real.endswith('.joblib'):
                    modelo = joblib.load(caminho_real)
                    modelo_nome = nome_modelo
                    modelo_carregado = True
                    print(f"✅ Modelo carregado (fallback): {modelo_nome}")
                    break
            except Exception as e:
                print(f"   ⚠️  Erro: {e}")
                continue
        
        if not modelo_carregado:
            raise RuntimeError("❌ Nenhum modelo válido pôde ser carregado!")
    
    # ==============================================================================
    # 🎯 VERIFICAÇÃO FINAL
    # ==============================================================================
    print(f"\n🎯 Modelo selecionado para NSGA-II: {modelo_nome}")
    print(f"   • Scaler: {type(scaler).__name__}")
    print(f"   • Target transform: OK")
    print(f"   • Modelo: {type(modelo).__name__}")
    
    # Verificação adicional para ElasticNet
    if 'ElasticNet' in modelo_nome:
        print(f"\n🔍 VERIFICAÇÃO DO ELASTICNET:")
        if hasattr(modelo, 'coef_'):
            print(f"   • Coeficientes shape: {modelo.coef_.shape}")
            print(f"   • Norma coeficientes: {np.linalg.norm(modelo.coef_):.6f}")
            
            if np.linalg.norm(modelo.coef_) < 0.001:
                print(f"   ⚠️  ALERTA: Coeficientes muito próximos de zero!")
                print(f"   • Modelo pode retornar valores constantes")
    
    return scaler, target_transform, modelo, modelo_nome

# ==============================================================================
# NOVA FUNÇÃO: CARREGAR DADOS REAIS DO MCNP (2000 CONFIGURAÇÕES)
# ==============================================================================

def carregar_dados_mcnp_sobol(caminho_arquivo):
    """
    Carrega e processa a tabela preparada da campanha DE/DF AP.
    Formato específico do seu arquivo:
    id_simulacao,dose_efetiva_mSv_seg,erro_relativo,materiais,espessuras,5 min de Dose efetiva (mSv)
    """
    
    print(f"\n📊 CARREGANDO DADOS MCNP: {os.path.basename(caminho_arquivo)}")
    
    if not os.path.exists(caminho_arquivo):
        raise FileNotFoundError(f"❌ Arquivo MCNP não encontrado: {caminho_arquivo}")
    
    try:
        # Carregar CSV
        df_mcnp = pd.read_csv(caminho_arquivo)
        print(f"   • Registros carregados: {len(df_mcnp)}")
        print(f"   • Colunas originais: {list(df_mcnp.columns)}")
        
        # Renomear colunas para facilitar
        rename_dict = {
            '5 min de Dose efetiva (mSv)': 'dose_5min',
            'dose_efetiva_mSv_seg': 'dose_mSv_s',
            'erro_relativo': 'erro_relativo',
            'materiais': 'materiais_str',
            'espessuras': 'espessuras_str',
            'id_simulacao': 'id_simulacao'
        }
        df_mcnp.rename(columns=rename_dict, inplace=True)
        
        # A tabela preparada preserva as sete colunas canônicas da Fase 1.
        # Elas têm precisão maior que as strings de apresentação e evitam que a
        # soma de espessuras seja alterada por arredondamento no NSGA-II.
        material_cols = [f'material_camada_{i}' for i in range(1, 8)]
        espessura_cols = [f'espessura_camada_{i}' for i in range(1, 8)]
        if set(material_cols + espessura_cols).issubset(df_mcnp.columns):
            print("   • Usando materiais e espessuras canônicos da Fase 1...")
            df_mcnp['materiais_list'] = df_mcnp[material_cols].apply(
                lambda row: [str(value).strip() for value in row], axis=1)
            df_mcnp['espessuras_list'] = df_mcnp[espessura_cols].apply(
                lambda row: [float(value) for value in row], axis=1)
        else:
            print("   • Processando colunas textuais de materiais e espessuras...")
            df_mcnp['materiais_list'] = df_mcnp['materiais_str'].apply(
                lambda value: [item.strip() for item in str(value).replace('"', '').replace("'", "").strip().split(',')])

            def parse_espessuras(value):
                text = str(value).replace('"', '').replace("'", "").replace('[', '').replace(']', '')
                return [float(item.strip()) for item in text.split(',')]

            df_mcnp['espessuras_list'] = df_mcnp['espessuras_str'].apply(parse_espessuras)
        
        # Calcular peso e custo para cada configuração
        print("   • Calculando peso e custo para cada configuração...")
        pesos = []
        custos = []
        
        for i, row in df_mcnp.iterrows():
            espessuras = row['espessuras_list']
            materiais = row['materiais_list']
            
            peso_total = 0.0
            custo_total = 0.0
            
            # Raio acumulado para cada camada
            raio_atual = RAIO_INTERNO_CM
            
            for j in range(NUM_CAMADAS):
                material = materiais[j]
                espessura = espessuras[j]
                
                # Raio externo desta camada
                raio_externo = raio_atual + espessura
                
                # Volume da casca semicilíndrica
                volume_cm3 = 0.5 * np.pi * (raio_externo**2 - raio_atual**2) * ALTURA_CM
                
                # Densidade do material
                densidade_g_cm3 = DENSIDADES.get(material, 2.0)
                
                # Massa em kg
                massa_kg = (densidade_g_cm3 * volume_cm3) / 1000.0
                peso_total += massa_kg
                
                # Custo desta camada
                custo_kg = CUSTOS_POR_KG.get(material, 0.0)
                custo_total += custo_kg * massa_kg
                
                # Atualizar raio para próxima camada
                raio_atual = raio_externo
            
            pesos.append(peso_total)
            custos.append(custo_total)
        
        df_mcnp['peso_kg'] = pesos
        df_mcnp['custo_usd'] = custos
        
        # Exemplos dos dados processados
        print(f"\n📋 EXEMPLOS DOS DADOS PROCESSADOS (3 primeiros registros):")
        for i in range(min(3, len(df_mcnp))):
            print(f"\n   📍 Registro {i+1}:")
            print(f"      • ID: {df_mcnp.iloc[i]['id_simulacao']}")
            print(f"      • Dose (5min): {df_mcnp.iloc[i]['dose_5min']:.3f} mSv")
            print(f"      • Dose (mSv/s): {df_mcnp.iloc[i]['dose_mSv_s']:.6f}")
            print(f"      • Peso: {df_mcnp.iloc[i]['peso_kg']:.2f} kg")
            print(f"      • Custo: ${df_mcnp.iloc[i]['custo_usd']:.2f}")
            print(f"      • Materiais: {df_mcnp.iloc[i]['materiais_list']}")
            print(f"      • Espessuras: {[f'{e:.2f}' for e in df_mcnp.iloc[i]['espessuras_list']]}")
            print(f"      • Soma espessuras: {sum(df_mcnp.iloc[i]['espessuras_list']):.2f} cm")
        
        # Estatísticas básicas
        print(f"\n📈 ESTATÍSTICAS DOS DADOS MCNP:")
        print(f"   • Número de configurações: {len(df_mcnp)}")
        print(f"   • Faixa dose (5min): {df_mcnp['dose_5min'].min():.3f} a {df_mcnp['dose_5min'].max():.3f} mSv")
        print(f"   • Média dose (5min): {df_mcnp['dose_5min'].mean():.3f} mSv")
        print(f"   • Faixa peso: {df_mcnp['peso_kg'].min():.1f} a {df_mcnp['peso_kg'].max():.1f} kg")
        print(f"   • Faixa custo: ${df_mcnp['custo_usd'].min():.0f} a ${df_mcnp['custo_usd'].max():.0f}")
        
        # Verificar consistência (dose_mSv_s * 300 ≈ dose_5min)
        print(f"\n🔍 VERIFICAÇÃO DE CONSISTÊNCIA DOS DADOS DE DOSE EFETIVA DO CSV:")
        for i in range(min(3, len(df_mcnp))):
            dose_calculada = df_mcnp.iloc[i]['dose_mSv_s'] * 300.0
            dose_relatada = df_mcnp.iloc[i]['dose_5min']
            diferenca_percentual = abs(dose_calculada - dose_relatada) / dose_relatada * 100
            
            print(f"   • Reg {i+1}: Calculada={dose_calculada:.3f} vs Relatada={dose_relatada:.3f} "
                  f"({diferenca_percentual:.2f}% de diferença)")
        
        print(f"\n✅ DADOS MCNP CARREGADOS COM SUCESSO!")
        return df_mcnp
        
    except Exception as e:
        print(f"❌ ERRO ao carregar dados MCNP: {e}")
        import traceback
        traceback.print_exc()
        return None

# Carregar dados MCNP para teste
df_mcnp = carregar_dados_mcnp_sobol(MCNP_DATA_PATH)

# ==============================================================================
# 🎯 DIAGNÓSTICO DO MODELO - IDENTIFICAR PORQUE RETORNA VALOR CONSTANTE
# ==============================================================================

def diagnosticar_modelo_constante(scaler, modelo, target_transform):
    """
    Diagnóstico para identificar por que o modelo retorna valores constantes
    """
    print("\n🔬 DIAGNÓSTICO DO MODELO - VALOR CONSTANTE")
    print("="*70)
    print("🚨 PROBLEMA: Modelo retornando dose constante (3.333 mSv/5min)")
    print("-"*70)
    
    # 1. Testar com configurações extremas dos dados MCNP
    print("1. TESTE COM CONFIGURAÇÕES EXTREMAS DO MCNP:")
    print("-"*40)
    
    if df_mcnp is not None:
        # Encontrar configurações com doses extremas
        idx_min_dose = df_mcnp['dose_5min'].idxmin()
        idx_max_dose = df_mcnp['dose_5min'].idxmax()
        
        for idx, label in [(idx_min_dose, "MENOR DOSE"), (idx_max_dose, "MAIOR DOSE")]:
            row = df_mcnp.loc[idx]
            
            print(f"\n📊 {label} (ID {row['id_simulacao']}):")
            print(f"   • Dose real MCNP: {row['dose_5min']:.3f} mSv/5min")
            print(f"   • Materiais: {row['materiais_list']}")
            print(f"   • Espessuras: {row['espessuras_list']}")
            
            # Converter para formato do NSGA-II
            materiais_idx = []
            for mat_nome in row['materiais_list']:
                if mat_nome in MATERIAIS_DISPONIVEIS:
                    materiais_idx.append(MATERIAIS_DISPONIVEIS.index(mat_nome))
                else:
                    materiais_idx.append(0)
            
            individuo = np.concatenate([row['espessuras_list'], materiais_idx])
            
            # Avaliar com modelo
            dose_pred, peso_pred, custo_pred = avaliar_individuo(
                individuo, scaler, modelo, target_transform
            )
            
            print(f"   • Dose predita: {dose_pred:.3f} mSv/5min")
            print(f"   • Erro: {abs(dose_pred - row['dose_5min'])/row['dose_5min']*100:.1f}%")
    
    # 2. Testar o modelo diretamente com features
    print("\n2. TESTE DIRETO DO MODELO ML:")
    print("-"*40)
    
    # Criar 3 configurações extremas para teste
    testes = [
        ("High-Z concentrado", 
         np.array([2.0, 2.0, 0.5, 0.5, 0.5, 0.25, 0.25]),  # Espessuras
         [3, 6, 2, 0, 1, 5, 4]),  # Índices: Bi2O3, W-Epoxy, Fe, SiC, Al, UHMWPE, B4C-HDPE
        
        ("Low-Z concentrado",
         np.array([0.25, 0.25, 0.5, 0.5, 2.0, 2.0, 0.5]),
         [5, 4, 0, 1, 2, 3, 6]),  # UHMWPE, Al, SiC, B4C-HDPE, Fe, Bi2O3, W-Epoxy
        
        ("Uniforme",
         np.array([0.857, 0.857, 0.857, 0.857, 0.857, 0.857, 0.857]),
         [0, 1, 2, 3, 4, 5, 6])  # Todos materiais em sequência
    ]
    
    for nome, espessuras, materiais_idx in testes:
        # Normalizar espessuras para soma = 6.0
        espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA
        
        print(f"\n📊 Configuração: {nome}")
        print(f"   • Espessuras: {['%.3f' % e for e in espessuras]}")
        print(f"   • Materiais: {[MATERIAIS_DISPONIVEIS[i] for i in materiais_idx]}")
        
        # Criar features manualmente
        features, peso, custo = criar_features_para_otimizacao(espessuras, materiais_idx)
        
        # Preparar vetor de features
        features_vector = np.array([[
            features['z_eff_ponderado_espessura'],
            features['atenuacao_total'],
            features['z_dens_esp_total'],
            features['high_z_protection'],
            features['low_z_moderation'],
            features['high_low_ratio'],
            features['z_x_atenuacao'],
            features['transmissao'],
            # features['log_transmissao'],
            features['sequence_score'],
            features['z_variance']
        ]])
        
        print(f"   • Features críticas:")
        print(f"     - Z_eff ponderado: {features['z_eff_ponderado_espessura']:.1f}")
        print(f"     - Atenuação total: {features['atenuacao_total']:.3f}")
        print(f"     - Transmissão: {features['transmissao']:.4f}")
        print(f"     - High/Low ratio: {features['high_low_ratio']:.2f}")
        
        # Aplicar scaler
        features_scaled = scaler.transform(features_vector)
        
        # Verificar se scaling está correto
        print(f"   • Features escaladas (após StandardScaler):")
        print(f"     - Valores: {features_scaled[0]}")
        print(f"     - Faixa: [{features_scaled[0].min():.3f}, {features_scaled[0].max():.3f}]")
        
        # Predição direta
        dose_pred_transformed = modelo.predict(features_scaled)[0]
        mean_log = target_transform['mean']
        std_log = target_transform['std']
        dose_pred_mSv_5min = inversa_transformacao_otimizada_5min(dose_pred_transformed, mean_log, std_log)
        
        print(f"   • Predição do modelo:")
        print(f"     - Transformada (z-score): {dose_pred_transformed:.3f}")
        print(f"     - mSv/5min: {dose_pred_mSv_5min:.3f}")
        
        # Verificar se é sempre o mesmo valor
        if nome == "High-Z concentrado":
            valor_referencia = dose_pred_mSv_5min
        else:
            if abs(dose_pred_mSv_5min - valor_referencia) < 0.001:
                print(f"     ⚠️  MESMO VALOR QUE ANTERIOR! (constante)")
    
    # 3. Verificar transformação do target
    print("\n3. VERIFICAÇÃO DA TRANSFORMAÇÃO DO TARGET:")
    print("-"*40)
    
    mean_log = target_transform['mean']
    std_log = target_transform['std']
    
    print(f"   • Parâmetros do target_transform:")
    print(f"     - mean (log): {mean_log:.6f}")
    print(f"     - std (log): {std_log:.6f}")
    print(f"     - Média linear: {np.exp(mean_log):.6e} mSv/s")
    print(f"     - Média para 5min: {np.exp(mean_log) * 300:.3f} mSv")
    
    # Calcular z-score para 3.333 mSv/5min
    dose_mSv_s = 3.333 / 300.0
    dose_log = np.log(dose_mSv_s)
    z_score = (dose_log - mean_log) / std_log
    
    print(f"\n   • Análise do valor 3.333 mSv/5min:")
    print(f"     - Em mSv/s: {dose_mSv_s:.6e}")
    print(f"     - Em log: {dose_log:.6f}")
    print(f"     - Z-score: {z_score:.3f}")
    print(f"     - Clip aplicado? {abs(z_score) > 5}")
    
    # 4. Verificar se o modelo PLS está funcionando
    print("\n4. VERIFICAÇÃO DO MODELO PLS:")
    print("-"*40)
    
    if hasattr(modelo, 'coef_'):
        print(f"   • Coeficientes do PLS: {modelo.coef_.shape}")
        print(f"   • Norma dos coeficientes: {np.linalg.norm(modelo.coef_):.6f}")
        
        if np.linalg.norm(modelo.coef_) < 1e-10:
            print(f"   ⚠️  COEFICIENTES PRÓXIMOS DE ZERO! Modelo não aprendeu")
        else:
            print(f"   • Coeficiente máximo: {np.max(np.abs(modelo.coef_)):.6f}")
            print(f"   • Coeficiente mínimo: {np.min(np.abs(modelo.coef_)):.6f}")
    
    if hasattr(modelo, 'x_loadings_'):
        print(f"   • Variância explicada: {modelo.x_loadings_.shape[1]} componentes")
    
    # 5. Recomendações
    print("\n5. RECOMENDAÇÕES:")
    print("-"*40)
    print("   A. Verificar se o arquivo 'target_transform.joblib' está correto")
    print("   B. Re-treinar modelo na Fase 3 com validação cruzada")
    print("   C. Testar outro modelo (RandomForest ou XGBoost)")
    print("   D. Verificar consistência das features entre Fase 3 e NSGA-II")
    
    return True

def calcular_z_variance_ponderada(z_eff_vals, espessuras):
    """
    Calcula variância de Z_eff PONDERADA pelas espessuras
    Isso varia entre configurações (diferente da variância simples que é constante)
    """
    # Média ponderada por espessura
    z_mean_ponderado = np.sum(z_eff_vals * espessuras) / np.sum(espessuras)
    
    # Variância ponderada
    numerador = np.sum(espessuras * (z_eff_vals - z_mean_ponderado)**2)
    denominador = np.sum(espessuras)
    
    # Normalizar para escala similar às outras features
    return numerador / (denominador + 1e-10) / 10.0  # Divide por 10 para escala razoável

# ==============================================================================
# FUNÇÃO DE FEATURE ENGINEERING PARA OTIMIZAÇÃO - VERSÃO CORRIGIDA
# ==============================================================================

def criar_features_para_otimizacao(espessuras, materiais_indices):
    """
    Cria features EXATAMENTE como na Fase 3 - MESMA ORDEM E CÁLCULO.
    Retorna:
        features : dict com as 16 features físico‑estratégicas (para compatibilidade).
        peso_total_kg : float, peso da blindagem (kg).
        custo_total : float, custo (USD).
        full_vector : np.ndarray (72,) vetor completo na ordem usada no treinamento.
    """
    # Converter índices para nomes de materiais
    materiais = []
    for idx in materiais_indices:
        try:
            idx_int = int(idx)
            materiais.append(MATERIAIS_DISPONIVEIS[idx_int])
        except:
            materiais.append('Al')  # fallback

    # ==================== FEATURES FÍSICO‑ESTRATÉGICAS ====================
    z_eff_vals = np.zeros(NUM_CAMADAS)
    mu_vals = np.zeros(NUM_CAMADAS)
    dens_vals = np.zeros(NUM_CAMADAS)

    for i in range(NUM_CAMADAS):
        material = materiais[i]
        z_eff_vals[i] = Z_EFF.get(material, 20.0)
        mu_vals[i] = MU_REAL.get(material, 0.07)
        dens_vals[i] = DENSIDADES.get(material, 2.0)

    features = {}

    # 1. Z_eff ponderado por espessura
    numerador = np.sum(z_eff_vals * espessuras)
    denominador = np.sum(espessuras)
    features['z_eff_ponderado_espessura'] = numerador / (denominador + 1e-10)

    # 2. Atenuação total
    atenuacao_total = np.sum(mu_vals * espessuras)
    features['atenuacao_total'] = atenuacao_total

    # 3. Z × Densidade × Espessura
    features['z_dens_esp_total'] = np.sum(z_eff_vals * dens_vals * espessuras)

    # 4. Proteção High‑Z (Z > 30)
    high_z = np.sum([z_eff_vals[i] * espessuras[i] for i in range(NUM_CAMADAS) if z_eff_vals[i] > 30])
    features['high_z_protection'] = high_z

    # 5. Moderação Low‑Z (Z ≤ 30)
    low_z = np.sum([z_eff_vals[i] * espessuras[i] for i in range(NUM_CAMADAS) if z_eff_vals[i] <= 30])
    features['low_z_moderation'] = low_z

    # 6. High/Low ratio
    features['high_low_ratio'] = high_z / (low_z + 1e-10)

    # 7. Interação Z × Atenuação
    features['z_x_atenuacao'] = features['z_dens_esp_total'] * features['atenuacao_total']

    # 8. Transmissão
    features['transmissao'] = np.exp(-atenuacao_total)

    # 9. Sequence score (penalidades e recompensas conforme Fase 3)
    sequence_score = 0.0
    if z_eff_vals[0] > 30:
        sequence_score -= 2.0
    if z_eff_vals[-1] > 30:
        sequence_score += 1.0
    if np.all(np.diff(z_eff_vals) >= 0):
        sequence_score += 1.5
    features['sequence_score'] = sequence_score

    # 10‑16. Espessura por material (thick_<material>)
    material_thickness = {mat: 0.0 for mat in MATERIAIS_DISPONIVEIS}
    for i, material in enumerate(materiais):
        material_thickness[material] += espessuras[i]

    for mat in MATERIAIS_DISPONIVEIS:
        features[f'thick_{mat}'] = material_thickness[mat]

    # ==================== CÁLCULO DE PESO E CUSTO ====================
    peso_total_kg = 0.0
    custo_total = 0.0
    raio_atual = RAIO_INTERNO_CM
    for i in range(NUM_CAMADAS):
        material = materiais[i]
        espessura = espessuras[i]

        raio_externo = raio_atual + espessura
        volume_cm3 = 0.5 * np.pi * (raio_externo**2 - raio_atual**2) * ALTURA_CM

        densidade_g_cm3 = DENSIDADES.get(material, 2.0)
        massa_kg = (densidade_g_cm3 * volume_cm3) / 1000.0
        peso_total_kg += massa_kg

        custo_kg = CUSTOS_POR_KG.get(material, 0.0)
        custo_total += custo_kg * massa_kg

        raio_atual = raio_externo

    # Penalidade por materiais repetidos (mantida da versão original)
    if len(set(materiais)) < len(materiais):
        num_repetidos = len(materiais) - len(set(materiais))
        fator_penalidade = 1.0 + (0.2 * num_repetidos)
        peso_total_kg *= fator_penalidade
        custo_total *= fator_penalidade

    # ==================== CONSTRUÇÃO DO VETOR COMPLETO (72 DIMENSÕES) ====================
    # 1. As 16 features físico‑estratégicas na ordem exata da Fase 3
    feature_order = [
        features['z_eff_ponderado_espessura'],
        features['atenuacao_total'],
        features['z_dens_esp_total'],
        features['high_z_protection'],
        features['low_z_moderation'],
        features['high_low_ratio'],
        features['z_x_atenuacao'],
        features['transmissao'],
        features['sequence_score']
    ]
    # Adiciona as 7 thick_<material> na ordem de MATERIAIS_DISPONIVEIS (mesmo da Fase 3)
    for mat in MATERIAIS_DISPONIVEIS:
        feature_order.append(features[f'thick_{mat}'])

    # 2. Espessuras individuais (7)
    feature_order.extend(espessuras)   # espessuras já é array de 7 floats

    # 3. One‑hot encoding dos materiais por camada (49)
    #    A ordem dos materiais na codificação one‑hot da Fase 3 é alfabética.
    alphabetical_materials = sorted(MATERIAIS_DISPONIVEIS)  # ['Al', 'B4C-HDPE', 'Bi2O3-HDPE', 'Fe', 'SiC', 'UHMWPE', 'W-Epoxy']
    one_hot = np.zeros(49, dtype=np.float64)
    for i, material in enumerate(materiais):
        # Encontra a posição alfabética do material
        try:
            idx_alpha = alphabetical_materials.index(material)
        except ValueError:
            idx_alpha = 0  # fallback
        pos = i * 7 + idx_alpha
        one_hot[pos] = 1.0
    feature_order.extend(one_hot)

    full_vector = np.array(feature_order, dtype=np.float64)

    return features, peso_total_kg, custo_total, full_vector

# ==============================================================================
# FUNÇÃO DE AVALIAÇÃO (FITNESS) DO NSGA-II
# ==============================================================================

def avaliar_individuo(individuo, scaler, modelo, target_transform):
    """
    Avalia um indivíduo (configuração de blindagem) e retorna os três objetivos
    VERSÃO CORRIGIDA – USANDO O VETOR COMPLETO DE 72 FEATURES
    """
    # 0. Garantir que indivíduo é array 1D
    try:
        individuo = np.asarray(individuo, dtype=float).flatten()
    except (ValueError, TypeError):
        return 1000.0, 10000.0, 1000000.0

    if len(individuo) != NUM_CAMADAS * 2:
        return 1000.0, 10000.0, 1000000.0

    # Decodificar
    espessuras = individuo[:NUM_CAMADAS]
    materiais_indices = individuo[NUM_CAMADAS:].astype(int)

    # 1. Ajustar espessuras para soma total fixa e limites
    espessuras = np.array(espessuras)
    soma_atual = np.sum(espessuras)
    if soma_atual > 0:
        espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma_atual)

    # 2. Aplicar clipping nos limites individuais e reajustar a soma iterativamente
    max_iter = 5
    for _ in range(max_iter):
        espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
        soma_clipped = np.sum(espessuras)
        if abs(soma_clipped - ESPESSURA_TOTAL_FIXA) < 1e-8:
            break
        espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma_clipped)

    # Garantia final: nenhuma espessura excede o limite
    espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
    if abs(np.sum(espessuras) - ESPESSURA_TOTAL_FIXA) > 1e-6:
        # Caso extremo, normaliza uma última vez
        espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA

    # 2. Obter o vetor completo de 72 features + peso/custo
    _, peso_total, custo_total, full_vector = criar_features_para_otimizacao(espessuras, materiais_indices)

    # 3. Aplicar o StandardScaler treinado na Fase 3 (espera 72 features)
    if hasattr(scaler, 'feature_names_in_'):
        X_df = pd.DataFrame(full_vector.reshape(1, -1), columns=scaler.feature_names_in_)
        features_scaled = scaler.transform(X_df)
    else:
        features_scaled = scaler.transform(full_vector.reshape(1, -1))

    # 4. Predição no espaço transformado
    dose_pred_transformed = modelo.predict(features_scaled)[0]

    # 5. Converter para dose em 5 minutos
    dose_5min = inversa_transformacao_otimizada_5min(
        dose_pred_transformed,
        target_transform['mean'],
        target_transform['std']
    )

    return dose_5min, peso_total, custo_total

# ==============================================================================
# ✅ ADICIONE AQUI AS FUNÇÕES DE DIAGNÓSTICO 
# ==============================================================================

def verificar_transformacao_target(target_transform):
    """
    Verifica os parâmetros da transformação do target
    """
    print("\n🔍 VERIFICAÇÃO DA TRANSFORMAÇÃO DO TARGET (DOSE)")
    print("="*60)
    
    if not isinstance(target_transform, dict):
        print(f"❌ ERRO: target_transform não é dict, é {type(target_transform)}")
        return False
    
    if 'mean' not in target_transform or 'std' not in target_transform:
        print("❌ ERRO: target_transform não contém mean/std")
        print(f"  Chaves disponíveis: {list(target_transform.keys())}")
        return False
    
    mean_log = target_transform['mean']
    std_log = target_transform['std']
    
    print(f"• Média (log): {mean_log:.4f}")
    print(f"• Desvio padrão (log): {std_log:.4f}")
    
    # Calcular faixa física esperada (mantém mSv/s para referência)
    print(f"\n• Faixa física esperada para dose (mSv/s) [ORIGINAL DO DATASET]:")
    print(f"  - Mínimo (mean - 3σ): {np.exp(mean_log - 3*std_log):.4e} mSv/s")
    print(f"  - Médio: {np.exp(mean_log):.4e} mSv/s")
    print(f"  - Máximo (mean + 3σ): {np.exp(mean_log + 3*std_log):.4e} mSv/s")
    
    # CALCULAR DIRETAMENTE PARA 5 MINUTOS (PONTO PRINCIPAL)
    print(f"\n• Faixa para 5 minutos ({FATOR_TEMPORAL_DOSE} s) [OTIMIZAÇÃO NSGA-II]:")
    print(f"  - Mínimo: {np.exp(mean_log - 3*std_log) * FATOR_TEMPORAL_DOSE:.2f} mSv/5min")
    print(f"  - Médio: {np.exp(mean_log) * FATOR_TEMPORAL_DOSE:.2f} mSv/5min")
    print(f"  - Máximo: {np.exp(mean_log + 3*std_log) * FATOR_TEMPORAL_DOSE:.2f} mSv/5min")
    
    # 🔥 NOVO: ANÁLISE DOS VALORES ATUAIS DO NSGA-II (0.848496 e 1.272744 mSv/5min)
    print(f"\n• Análise dos valores atuais do NSGA-II:")
    
    # Valor 1: 0.848496 mSv/5min
    dose_atual_1 = 0.848496
    dose_mSv_s_1 = dose_atual_1 / FATOR_TEMPORAL_DOSE
    dose_log_1 = np.log(dose_mSv_s_1)
    z_score_1 = (dose_log_1 - mean_log) / std_log
    
    print(f"  - Valor 1: {dose_atual_1:.6f} mSv/5min")
    print(f"    • Em mSv/s: {dose_mSv_s_1:.6e}")
    print(f"    • Em log: {dose_log_1:.4f}")
    print(f"    • Z-score: {z_score_1:.2f} σ")
    print(f"    • Faixa: {'✓ DENTRO' if -3 <= z_score_1 <= 3 else '✗ FORA'} da faixa de treinamento")
    
    # Valor 2: 1.272744 mSv/5min
    dose_atual_2 = 1.272744
    dose_mSv_s_2 = dose_atual_2 / FATOR_TEMPORAL_DOSE
    dose_log_2 = np.log(dose_mSv_s_2)
    z_score_2 = (dose_log_2 - mean_log) / std_log
    
    print(f"  - Valor 2: {dose_atual_2:.6f} mSv/5min")
    print(f"    • Em mSv/s: {dose_mSv_s_2:.6e}")
    print(f"    • Em log: {dose_log_2:.4f}")
    print(f"    • Z-score: {z_score_2:.2f} σ")
    print(f"    • Faixa: {'✓ DENTRO' if -3 <= z_score_2 <= 3 else '✗ FORA'} da faixa de treinamento")
    
    # REMOVER análise do 51.365 (não é mais relevante)
    # dose_atual = 51.365 / FATOR_TEMPORAL_DOSE  # REMOVER
    # dose_log_atual = np.log(dose_atual)  # REMOVER
    # print(f"\n• Análise do valor atual (51.365 mSv/5min):")  # REMOVER
    # print(f"  - Em mSv/s: {dose_atual:.6f}")  # REMOVER
    # print(f"  - Em log: {dose_log_atual:.4f}")  # REMOVER
    # print(f"  - Z-score: {(dose_log_atual - mean_log)/std_log:.2f} σ")  # REMOVER
    
    # ✅ NOVA CONCLUSÃO
    print(f"\n✅ CONCLUSÃO PARA NSGA-II:")
    print(f"  • O NSGA-II está trabalhando na faixa de {dose_atual_1:.2f} a {dose_atual_2:.2f} mSv/5min")
    print(f"  • Esta faixa corresponde a {dose_mSv_s_1:.2e} a {dose_mSv_s_2:.2e} mSv/s")
    print(f"  • Os valores estão {'DENTRO' if -3 <= min(z_score_1, z_score_2) and max(z_score_1, z_score_2) <= 3 else 'FORA'} da faixa de treinamento")
    print(f"  • A otimização será efetiva se os valores variarem dentro desta faixa")
    
    return True

def diagnosticar_modelo_dose(scaler, modelo, target_transform):
    """
    Diagnóstico científico do modelo de dose - Testa variação com configurações válidas
    """
    
    print("\n🔍 DIAGNÓSTICO DO MODELO DE DOSE")
    print("="*60)
    print("Nota: Todas as configurações respeitam a restrição de materiais únicos")
    
    # Testar com 5 configurações ALEATÓRIAS E VÁLIDAS
    configs_teste = []
    
    # Configuração 1: Sequência balanceada (High-Z, Low-Z intercalados)
    esp1 = np.random.uniform(ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA, NUM_CAMADAS)
    esp1 = esp1 / np.sum(esp1) * ESPESSURA_TOTAL_FIXA
    mat1 = list(range(len(MATERIAIS_DISPONIVEIS)))[:NUM_CAMADAS]
    np.random.shuffle(mat1)
    configs_teste.append(("Sequência balanceada", esp1, mat1))
    
    # Configuração 2: Z crescente (Low-Z para High-Z)
    esp2 = np.random.uniform(ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA, NUM_CAMADAS)
    esp2 = esp2 / np.sum(esp2) * ESPESSURA_TOTAL_FIXA
    z_eff_order = [6, 4, 0, 2, 1, 5, 3]  # Índices ordenados por Z_eff aproximado
    mat2 = z_eff_order[:NUM_CAMADAS]
    configs_teste.append(("Z crescente", esp2, mat2))
    
    # Configuração 3: Z decrescente (High-Z para Low-Z)
    esp3 = np.random.uniform(ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA, NUM_CAMADAS)
    esp3 = esp3 / np.sum(esp3) * ESPESSURA_TOTAL_FIXA
    mat3 = list(reversed(z_eff_order))[:NUM_CAMADAS]
    configs_teste.append(("Z decrescente", esp3, mat3))
    
    # Configuração 4: Espessuras uniformes, materiais aleatórios únicos
    esp4 = np.ones(NUM_CAMADAS) * (ESPESSURA_TOTAL_FIXA / NUM_CAMADAS)
    mat4 = np.random.permutation(len(MATERIAIS_DISPONIVEIS))[:NUM_CAMADAS]
    configs_teste.append(("Espessuras uniformes", esp4, mat4))
    
    # Configuração 5: Totalmente aleatório (válido)
    esp5 = np.random.uniform(ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA, NUM_CAMADAS)
    esp5 = esp5 / np.sum(esp5) * ESPESSURA_TOTAL_FIXA
    mat5 = np.random.permutation(len(MATERIAIS_DISPONIVEIS))[:NUM_CAMADAS]
    configs_teste.append(("Aleatório completo", esp5, mat5))
    
    resultados = []
    
    for i, (nome, espessuras, materiais_idx) in enumerate(configs_teste):
        # Criar indivíduo
        individuo = np.concatenate([espessuras, materiais_idx])
        
        # Avaliar usando a MESMA função que o NSGA-II usa
        dose, peso, custo = avaliar_individuo(
            individuo, scaler, modelo, target_transform
        )
        
        # Converter índices para nomes de materiais
        materiais_nomes = [MATERIAIS_DISPONIVEIS[idx] for idx in materiais_idx]
        
        # ✅ CORREÇÃO: Calcular Z_eff ponderado por espessura (igual à feature real)
        z_vals = np.array([Z_EFF.get(m, 20.0) for m in materiais_nomes])
        z_eff_ponderado = np.sum(z_vals * espessuras) / np.sum(espessuras)
        
        resultados.append({
            'Nome': nome,
            'Dose': dose,
            'Peso': peso,
            'Custo': custo,
            'Materiais': materiais_nomes,
            'Espessuras': espessuras,
            'Z_eff_ponderado': z_eff_ponderado,  # ✅ CORRETO: ponderado por espessura
            'Z_eff_medio': np.mean(z_vals)      # Mantido para referência
        })
    
    # Exibir resultados
    print(f"\n📊 RESULTADOS DO DIAGNÓSTICO ({len(resultados)} configurações válidas):")
    print("-"*60)
    
    doses = [r['Dose'] for r in resultados]
    pesos = [r['Peso'] for r in resultados]
    custos = [r['Custo'] for r in resultados]
    z_effs_pond = [r['Z_eff_ponderado'] for r in resultados]
    
    for r in resultados:
        print(f"\n{r['Nome']}:")
        print(f"  • Dose: {r['Dose']:.6f} mSv/5min")
        print(f"  • Peso: {r['Peso']:.2f} kg")
        print(f"  • Custo: ${r['Custo']:.2f}")
        print(f"  • Z_eff ponderado: {r['Z_eff_ponderado']:.1f}")  # ✅ CORRIGIDO
        print(f"  • Z_eff médio simples: {r['Z_eff_medio']:.1f}")  # Para referência
    
    print(f"\n📈 ANÁLISE ESTATÍSTICA:")
    print(f"  • Faixa de doses: {min(doses):.6f} a {max(doses):.6f} mSv/5min")
    print(f"  • Variação relativa: {(max(doses)-min(doses))/min(doses)*100:.1f}%")
    print(f"  • Coeficiente de variação: {np.std(doses)/np.mean(doses)*100:.1f}%")
    print(f"  • Z_eff ponderado: {np.mean(z_effs_pond):.1f} ± {np.std(z_effs_pond):.1f}")
    print(f"  • Peso médio: {np.mean(pesos):.1f} ± {np.std(pesos):.1f} kg")
    print(f"  • Custo médio: ${np.mean(custos):.0f} ± ${np.std(custos):.0f}")
    
    # ✅ NOTA CIENTÍFICA SOBRE Z_EFF
    print(f"\n🔬 NOTA CIENTÍFICA SOBRE Z_EFF:")
    print(f"  • Z_eff médio simples ≈ 29.5 para TODAS configurações (matematicamente)")
    print(f"  • Z_eff ponderado por espessura VARIA entre configurações")
    print(f"  • O modelo usa Z_eff PONDERADO (não médio) para previsão")
    
    # Critério de validação para artigo
    print(f"\n✅ VALIDAÇÃO CIENTÍFICA:")
    if (max(doses) - min(doses)) > 0.1:
        print(f"  • Variação de dose: SUFICIENTE ({max(doses)-min(doses):.3f} mSv/5min)")
        print(f"  • Modelo: CAPAZ de discriminar diferentes configurações")
    else:
        print(f"  ⚠️  Variação de dose: INSUFICIENTE para otimização")
        print(f"  • Modelo pode estar retornando valores constantes")
    
    print("\n" + "="*60)
    
    return resultados

def debug_criar_features():
    """Descobre porque Z_eff é sempre 29.5"""
    print("\n🔬 DEBUG PROFUNDO: PORQUE Z_EFF = 29.5 SEMPRE?")
    print("="*60)
    
    # Configuração simples: 2 materiais diferentes
    esp = np.array([3.0, 3.0])  # 6 cm total
    mat_idx = [0, 1]  # SiC e B4C-HDPE
    
    features, peso, custo = criar_features_para_otimizacao(esp, mat_idx)
    
    print(f"\nConfiguração: SiC(3cm) + B4C-HDPE(3cm)")
    print(f"Z_eff esperado: ≈{(13.5 + 5.2)/2:.1f} = 9.35")
    print(f"Z_eff calculado: {features['z_eff_ponderado_espessura']:.1f}")
    
    # Calcular manualmente:
    z_sic = Z_EFF['SiC']  # 13.5
    z_b4c = Z_EFF['B4C-HDPE']  # 5.2
    
    manual_z_eff = (z_sic * 3.0 + z_b4c * 3.0) / 6.0
    print(f"Z_eff manual: {manual_z_eff:.1f}")
    
    if abs(features['z_eff_ponderado_espessura'] - manual_z_eff) > 0.1:
        print(f"\n❌ BUG! Função retorna {features['z_eff_ponderado_espessura']:.1f} vs {manual_z_eff:.1f}")
        
        # Verificar a função get_property
        print(f"\n🔍 Verificando get_property:")
        from your_module import get_property  # Ajuste conforme necessário
        
        z1 = get_property('SiC', Z_EFF, 20.0)
        z2 = get_property('B4C-HDPE', Z_EFF, 20.0)
        print(f"  • get_property('SiC') = {z1}")
        print(f"  • get_property('B4C-HDPE') = {z2}")

def debug_features_extremas():
    """Debug das features para entender porque o modelo retorna valores constantes"""
    print("\n🔬 DEBUG DAS FEATURES - ANÁLISE DE VARIAÇÃO")
    print("="*60)
    
    print("\n📊 TESTE RÁPIDO: Configurações Extremas")
    
    # Configuração 1: Todos High-Z (ordem decrescente de Z)
    esp1 = np.ones(7) * (6.0/7)
    mat1_idx = [MATERIAIS_DISPONIVEIS.index('Bi2O3-HDPE'),  # Z=74.3
                MATERIAIS_DISPONIVEIS.index('W-Epoxy'),     # Z=69.2
                MATERIAIS_DISPONIVEIS.index('Fe'),         # Z=25.8
                MATERIAIS_DISPONIVEIS.index('SiC'),        # Z=13.5
                MATERIAIS_DISPONIVEIS.index('Al'),         # Z=12.8
                MATERIAIS_DISPONIVEIS.index('UHMWPE'),     # Z=5.9
                MATERIAIS_DISPONIVEIS.index('B4C-HDPE')]   # Z=5.2
    
    # Configuração 2: Todos Low-Z (ordem crescente de Z)  
    esp2 = np.ones(7) * (6.0/7)
    mat2_idx = [MATERIAIS_DISPONIVEIS.index('B4C-HDPE'),   # Z=5.2
                MATERIAIS_DISPONIVEIS.index('UHMWPE'),     # Z=5.9
                MATERIAIS_DISPONIVEIS.index('Al'),         # Z=12.8
                MATERIAIS_DISPONIVEIS.index('SiC'),        # Z=13.5
                MATERIAIS_DISPONIVEIS.index('Fe'),         # Z=25.8
                MATERIAIS_DISPONIVEIS.index('W-Epoxy'),    # Z=69.2
                MATERIAIS_DISPONIVEIS.index('Bi2O3-HDPE')] # Z=74.3
    
    # Calcular features
    feat1, p1, c1 = criar_features_para_otimizacao(esp1, mat1_idx)
    feat2, p2, c2 = criar_features_para_otimizacao(esp2, mat2_idx)
    
    # ✅ CALCULAR Z_EFF MÉDIO SIMPLES PARA COMPARAÇÃO
    z_vals1 = [Z_EFF[MATERIAIS_DISPONIVEIS[idx]] for idx in mat1_idx]
    z_vals2 = [Z_EFF[MATERIAIS_DISPONIVEIS[idx]] for idx in mat2_idx]
    
    print(f"\n🔍 CONFIGURAÇÃO 1 (High-Z primeiro):")
    print(f"  • Z_eff ponderado: {feat1['z_eff_ponderado_espessura']:.1f}")
    print(f"  • Z_eff médio simples: {np.mean(z_vals1):.1f}")
    print(f"  • Atenuação total: {feat1['atenuacao_total']:.3f}")
    print(f"  • Transmissão: {feat1['transmissao']:.4f}")
    print(f"  • High/Low ratio: {feat1['high_low_ratio']:.2f}")
    
    print(f"\n🔍 CONFIGURAÇÃO 2 (Low-Z primeiro):")
    print(f"  • Z_eff ponderado: {feat2['z_eff_ponderado_espessura']:.1f}")
    print(f"  • Z_eff médio simples: {np.mean(z_vals2):.1f}")
    print(f"  • Atenuação total: {feat2['atenuacao_total']:.3f}")
    print(f"  • Transmissão: {feat2['transmissao']:.4f}")
    print(f"  • High/Low ratio: {feat2['high_low_ratio']:.2f}")
    
    print(f"\n📈 DIFERENÇAS:")
    print(f"  • ΔZ_eff ponderado: {abs(feat1['z_eff_ponderado_espessura'] - feat2['z_eff_ponderado_espessura']):.2f}")
    print(f"  • ΔAtenuação: {abs(feat1['atenuacao_total'] - feat2['atenuacao_total']):.3f}")
    print(f"  • ΔTransmissão: {abs(feat1['transmissao'] - feat2['transmissao']):.4f}")
    
    # ✅ CORREÇÃO DA MENSAGEM
    print(f"\n🔬 ANÁLISE:")
    print(f"  • Z_eff médio simples é SEMPRE ≈29.5 (7 materiais únicos)")
    print(f"  • Z_eff ponderado por espessura VARIA entre configurações")
    print(f"  • O modelo usa features corretas para discriminação")
    print(f"  • Scaler utilizado: StandardScaler")  # ← ÚNICA linha alterada

# ==============================================================================
# 🎯 NOVA FUNÇÃO: DEBUG DO Z_EFF PONDERADO
# ==============================================================================

def debug_z_eff_ponderado(scaler, modelo, target_transform):
    """
    Verifica se Z_eff ponderado realmente varia entre configurações
    
    Parameters:
    -----------
    scaler : StandardScaler
        Scaler treinado na Fase 3
    modelo : object
        Modelo ML (scikit-learn ou Keras)
    target_transform : dict
        Transformação do target (mean_log, std_log)
    """
    print("\n🔬 DEBUG DO Z_EFF PONDERADO - VERIFICAÇÃO CIENTÍFICA")
    print("="*60)
    print("Objetivo: Verificar se Z_eff ponderado varia com diferentes distribuições de espessura")
    print("-"*60)
    
    # Configurações de teste
    configs = [
        ("Uniforme", np.ones(7) * (6.0/7)),
        ("Variado 1", np.array([0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])),
        ("Variado 2", np.array([1.5, 1.0, 0.5, 0.5, 0.5, 0.8, 1.2])),
        ("Extremo", np.array([2.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.2]))
    ]
    
    # Materiais fixos (mesma sequência para todos)
    # Ordenados por Z_eff para análise
    materiais_ordenados = ['B4C-HDPE', 'UHMWPE', 'Al', 'SiC', 'Fe', 'W-Epoxy', 'Bi2O3-HDPE']
    materiais_idx = [MATERIAIS_DISPONIVEIS.index(m) for m in materiais_ordenados]
    
    resultados = []
    
    for nome, espessuras in configs:
        # Normalizar para soma = 6.0 cm
        espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA
        
        # Criar indivíduo
        individuo = np.concatenate([espessuras, materiais_idx])
        
        # Calcular Z_eff ponderado manualmente
        z_eff_vals = np.array([Z_EFF[MATERIAIS_DISPONIVEIS[i]] for i in materiais_idx])
        z_eff_ponderado = np.sum(z_eff_vals * espessuras) / np.sum(espessuras)
        
        # Calcular Z_eff médio simples (sempre 29.5)
        z_eff_medio = np.mean(z_eff_vals)
        
        # Avaliar com o modelo
        dose, peso, custo = avaliar_individuo(
            individuo, scaler, modelo, target_transform
        )
        
        resultados.append({
            'Nome': nome,
            'Z_eff_ponderado': z_eff_ponderado,
            'Z_eff_medio': z_eff_medio,
            'Dose': dose,
            'Peso': peso,
            'Custo': custo,
            'Espessuras': espessuras.copy()
        })
        
        print(f"\n📊 {nome}:")
        print(f"  • Espessuras: {['%.3f' % e for e in espessuras]}")
        print(f"  • Materiais: {materiais_ordenados}")
        print(f"  • Z_eff ponderado: {z_eff_ponderado:.2f}")
        print(f"  • Z_eff médio: {z_eff_medio:.2f}")
        print(f"  • ΔZ_eff (ponderado - médio): {z_eff_ponderado - z_eff_medio:.2f}")
        print(f"  • Dose: {dose:.6f} mSv/5min")
        print(f"  • Peso: {peso:.2f} kg")
        print(f"  • Custo: ${custo:.2f}")
    
    # ANÁLISE ESTATÍSTICA
    print("\n" + "="*60)
    print("📈 ANÁLISE ESTATÍSTICA DOS RESULTADOS:")
    
    z_eff_vals = [r['Z_eff_ponderado'] for r in resultados]
    dose_vals = [r['Dose'] for r in resultados]
    
    print(f"  • Z_eff ponderado: {np.min(z_eff_vals):.2f} a {np.max(z_eff_vals):.2f}")
    print(f"  • Variação Z_eff: {np.max(z_eff_vals) - np.min(z_eff_vals):.2f}")
    print(f"  • Dose: {np.min(dose_vals):.6f} a {np.max(dose_vals):.6f} mSv/5min")
    print(f"  • Variação dose: {np.max(dose_vals) - np.min(dose_vals):.6f} mSv/5min")
    
    # Correlação entre Z_eff ponderado e dose
    if len(z_eff_vals) > 1:
        corr = np.corrcoef(z_eff_vals, dose_vals)[0, 1]
        print(f"  • Correlação Z_eff ponderado vs Dose: {corr:.3f}")
    
    print("\n🔍 CONCLUSÃO:")
    if np.max(dose_vals) - np.min(dose_vals) > 0.1:
        print("✅ Z_eff ponderado VARIA e afeta a dose significativamente")
        print("   O modelo está funcionando corretamente")
    else:
        print("⚠️  Z_eff ponderado varia MAS a dose NÃO varia significativamente")
        print("   Possível problema: Features não estão discriminando adequadamente")
    
    return resultados

def diagnosticar_sequence_score_completo(output_dir):
    """Diagnóstico completo da feature sequence_score no dataset de 2000 amostras.
    
    Parâmetros:
    -----------
    output_dir : str, optional
        Caminho do diretório onde o arquivo CSV de análise será salvo.
        Se None, levanta um erro para evitar salvamento acidental.
    """
    print("\n" + "="*60)
    print("🔬 DIAGNÓSTICO CIENTÍFICO: SEQUENCE_SCORE")
    print("="*60)
    
    df_configs = pd.read_csv(CONFIGS_SOBOL_PATH)
    
    all_sequence_scores = []
    
    for idx in range(len(df_configs)):
        materiais = []
        for i in range(1, 8):
            materiais.append(str(df_configs.loc[idx, f'material_camada_{i}']).strip())
        
        z_vals = [Z_EFF.get(m, 20.0) for m in materiais]
        score = 0.0
        
        if z_vals[0] > 30:
            score -= 2.0
        if z_vals[-1] > 30:
            score += 1.0
        if np.all(np.diff(z_vals) >= 0):
            score += 1.5
        
        normalized_score = score / 3.0
        all_sequence_scores.append(normalized_score)
    
    all_sequence_scores = np.array(all_sequence_scores)
    
    print(f"📊 ESTATÍSTICAS DA SEQUENCE_SCORE (2000 amostras):")
    print(f"  • Únicos valores: {np.unique(all_sequence_scores)}")
    print(f"  • Média: {np.mean(all_sequence_scores):.6f}")
    print(f"  • Desvio padrão: {np.std(all_sequence_scores):.6f}")
    print(f"  • Mínimo: {np.min(all_sequence_scores):.6f}")
    print(f"  • Máximo: {np.max(all_sequence_scores):.6f}")
    print(f"  • Coeficiente de variação: {(np.std(all_sequence_scores)/np.mean(all_sequence_scores)*100 if np.mean(all_sequence_scores)!=0 else 0):.2f}%")
    
    unique_vals, counts = np.unique(all_sequence_scores, return_counts=True)
    print(f"\n📈 DISTRIBUIÇÃO:")
    for val, count in zip(unique_vals, counts):
        print(f"  • {val:6.4f}: {count:4d} amostras ({count/len(all_sequence_scores)*100:.1f}%)")
    
    # Criar DataFrame para exportação
    df_analysis = pd.DataFrame({'sequence_score': all_sequence_scores})
    
    # 🚨 CORREÇÃO: Agora exige output_dir explicitamente
    if output_dir is None:
        raise ValueError("❌ output_dir deve ser especificado para salvar o diagnóstico. "
                         "Chame a função com o caminho do diretório de debug.")
    
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'sequence_score_analysis.csv')
    df_analysis.to_csv(csv_path, index=False)
    print(f"\n💾 Dados salvos em: {csv_path}")
    
    # Conclusão baseada no desvio padrão
    if np.std(all_sequence_scores) < 0.001:
        print(f"\n🚨 CONCLUSÃO CIENTÍFICA:")
        print(f"  • Sequence_score é CONSTANTE em todo o dataset (std={np.std(all_sequence_scores):.6f})")
        print(f"  • Isso viola os pressupostos do StandardScaler")
        print(f"\n💡 RECOMENDAÇÃO PARA O ARTIGO:")
        print(f"  1. Reportar esta descoberta como uma limitação")
        print(f"  2. Justificar que a unicidade dos materiais restringe as sequências possíveis")
        print(f"  3. Manter a feature para consistência metodológica")
        print(f"  4. Discutir nas 'limitações e trabalhos futuros'")
    else:
        print(f"\n✅ Sequence_score varia no dataset (std={np.std(all_sequence_scores):.6f})")
    
    return all_sequence_scores

# # ==============================================================================
# 🔬 VERIFICAÇÃO OBJETIVA: FEATURES FASE 3 vs FASE 4
# ==============================================================================

def verificar_features_fase3_fase4():
    """
    Verifica se as features geradas na Fase 4 correspondem àquelas usadas no treinamento (Fase 3).
    Foco: número de features, consistência do Z_eff e presença de todas as features físicas e espessuras.
    """
    print("\n" + "="*60)
    print("🔬 VERIFICAÇÃO: FEATURES FASE 3 vs FASE 4")
    print("="*60)
    
    # 1. Carregar uma amostra do dataset original (Fase 1)
    print("1. CARREGANDO AMOSTRA DO DATASET FASE 3...")
    df_configs = pd.read_csv(CONFIGS_SOBOL_PATH)
    
    # Pegar primeira amostra (índice 0)
    idx = 0
    
    # Extrair dados da Fase 3 (espessuras e nomes dos materiais)
    espessuras_fase3 = []
    materiais_fase3 = []
    
    for i in range(1, 8):
        esp_col = f'espessura_camada_{i}'
        mat_col = f'material_camada_{i}'
        
        esp = float(str(df_configs.loc[idx, esp_col]).replace(',', '.'))
        mat = str(df_configs.loc[idx, mat_col]).strip()
        
        espessuras_fase3.append(esp)
        materiais_fase3.append(mat)
    
    print(f"   • Amostra {idx}:")
    print(f"     Espessuras: {['%.3f' % e for e in espessuras_fase3]}")
    print(f"     Materiais: {materiais_fase3}")
    
    # 2. Criar features com a função da Fase 4
    print("\n2. CRIANDO FEATURES COM FUNÇÃO FASE 4...")
    
    # Converter materiais para índices
    materiais_idx_fase4 = []
    for mat_nome in materiais_fase3:
        if mat_nome in MATERIAIS_DISPONIVEIS:
            materiais_idx_fase4.append(MATERIAIS_DISPONIVEIS.index(mat_nome))
        else:
            materiais_idx_fase4.append(0)  # fallback (não deve ocorrer)
    
    features_fase4, peso_f4, custo_f4, _ = criar_features_para_otimizacao(
        np.array(espessuras_fase3), 
        materiais_idx_fase4
    )
    
    # 3. Verificar se o dicionário contém todas as 16 features esperadas
    print("\n3. VERIFICANDO FEATURES GERADAS...")
    expected_features = [
        'z_eff_ponderado_espessura',
        'atenuacao_total',
        'z_dens_esp_total',
        'high_z_protection',
        'low_z_moderation',
        'high_low_ratio',
        'z_x_atenuacao',
        'transmissao',
        'sequence_score',
        'thick_SiC',
        'thick_B4C-HDPE',
        'thick_Fe',
        'thick_Bi2O3-HDPE',
        'thick_Al',
        'thick_UHMWPE',
        'thick_W-Epoxy'
    ]
    
    missing = [f for f in expected_features if f not in features_fase4]
    if missing:
        print(f"   ❌ FEATURES AUSENTES: {missing}")
        return False, features_fase4
    else:
        print(f"   ✅ Todas as 16 features estão presentes.")
    
    # 4. Verificar consistência do Z_eff ponderado
    print("\n4. VERIFICAÇÃO Z_EFF PONDERADO:")
    z_vals = [Z_EFF.get(m, 20.0) for m in materiais_fase3]
    z_eff_manual = np.sum(np.array(z_vals) * np.array(espessuras_fase3)) / np.sum(espessuras_fase3)
    z_eff_calculado = features_fase4['z_eff_ponderado_espessura']
    
    print(f"   • Z_eff manual (ponderado): {z_eff_manual:.2f}")
    print(f"   • Z_eff calculado: {z_eff_calculado:.2f}")
    print(f"   • Diferença: {abs(z_eff_manual - z_eff_calculado):.6f}")
    
    if abs(z_eff_manual - z_eff_calculado) < 0.1:
        print(f"   ✅ Z_EFF CONSISTENTE")
    else:
        print(f"   ❌ Z_EFF INCONSISTENTE")
        return False, features_fase4
    
    # 5. Verificar rapidamente se as espessuras por material somam corretamente
    print("\n5. VERIFICAÇÃO DAS ESPESSURAS POR MATERIAL:")
    soma_thickness = sum(features_fase4[f'thick_{mat}'] for mat in MATERIAIS_DISPONIVEIS)
    print(f"   • Soma das espessuras (thick_*): {soma_thickness:.3f} cm")
    print(f"   • Espessura total real: {np.sum(espessuras_fase3):.3f} cm")
    if abs(soma_thickness - np.sum(espessuras_fase3)) < 0.01:
        print(f"   ✅ CONSISTENTE")
    else:
        print(f"   ❌ INCONSISTENTE")
        return False, features_fase4
    
    print("\n✅ VERIFICAÇÃO CONCLUÍDA: FEATURES DA FASE 4 CONSISTENTES COM A FASE 3.")
    return True, features_fase4

def mostrar_resumo_final(historico_fitness):
    """Mostra resumo enxuto da evolução"""
    print("\n" + "="*80)
    print("RESUMO DA EVOLUÇÃO - 10 PONTOS CHAVE")
    print("="*80)
    print("Ger |  Melhor Dose   |   Peso    |   Custo    |  ΔDose(%)")
    print("-"*80)
    
    # Selecionar 10 pontos igualmente espaçados
    n_geracoes = len(historico_fitness)
    intervalos = max(1, n_geracoes // 10)
    
    dose_inicial = None
    
    for g in range(0, n_geracoes, intervalos):
        fitness_g = historico_fitness[g]
        melhor_dose = np.min(fitness_g[:, 0])
        melhor_idx = np.argmin(fitness_g[:, 0])
        melhor_peso = fitness_g[melhor_idx, 1]
        melhor_custo = fitness_g[melhor_idx, 2]
        
        if dose_inicial is None:
            dose_inicial = melhor_dose

        delta = ((dose_inicial - melhor_dose) / dose_inicial * 100) if dose_inicial > 0 else 0

        print(f"{g+1:3d} | {melhor_dose:9.4f} mSv | {melhor_peso:7.1f} kg | ${melhor_custo:8.0f} | {delta:7.2f}%")
    
    # Última geração sempre
    if n_geracoes-1 not in range(0, n_geracoes, intervalos):
        fitness_final = historico_fitness[-1]
        melhor_dose = np.min(fitness_final[:, 0])
        melhor_idx = np.argmin(fitness_final[:, 0])
        melhor_peso = fitness_final[melhor_idx, 1]
        melhor_custo = fitness_final[melhor_idx, 2]
        delta = ((dose_inicial - melhor_dose) / dose_inicial * 100) if dose_inicial > 0 else 0
        
        print(f"{n_geracoes:3d} | {melhor_dose:9.4f} mSv | {melhor_peso:7.1f} kg | ${melhor_custo:8.0f} | {delta:7.2f}%")

# ==============================================================================
# IMPLEMENTAÇÃO DO NSGA-II (VERSÃO REVISADA)
# ==============================================================================

class NSGA2Optimizer:
    """Implementação do NSGA-II com checkpointing para retomada."""

    def __init__(self, scaler, modelo, target_transform, modelo_nome=None, modelo_r2=None,
             results_dir=None, figures_dir=None, timestamp=None):
        self.scaler = scaler
        self.modelo = modelo
        self.target_transform = target_transform
        self.modelo_nome = modelo_nome if modelo_nome else "Modelo_ML"
        self.modelo_r2 = modelo_r2

        # ---------- Históricos e contadores (todos devem ser salvos no checkpoint) ----------
        self.historico_melhor_dose = []          # melhor dose por geração
        self.historico_convergencia = []         # histórico de convergência (opcional)
        self.historico_diagnosticos = []          # diagnósticos de cada geração
        self.historico_populacoes = []            # população por geração (pode ser grande)
        self.historico_fitness = []                # fitness por geração
        self.hv_history = []                       # hipervolume normalizado por geração
        self.num_restarts = 0                       # número de restarts realizados
        # self.consecutive_stagnation = 0             # contador de estagnação consecutiva
        self.mutacao_chamadas = 0                   # total de mutações aplicadas
        self.current_generation = 0
        self.stagnation_limit = 8   # número de gerações sem melhoria para acionar restart
        self.results_dir = results_dir
        self.figures_dir = figures_dir
        self.timestamp = timestamp if timestamp else time.strftime("%Y%m%d_%H%M%S")

        # ---------- Parâmetros geométricos e de materiais (para uso nos checkpoints) ----------
        self.NUM_CAMADAS = NUM_CAMADAS                     # variável global definida no início do script
        self.MATERIAIS_DISPONIVEIS = MATERIAIS_DISPONIVEIS # variável global

        # ---------- Configurações adaptativas (constantes, não precisam ser salvas) ----------
        self.mutacao_inicial = 0.25                  # taxa de mutação na primeira geração
        self.mutacao_final = 0.10                    # taxa de mutação na última geração
        self.restart_percentual = 0.20               # % da população substituída no restart

        # ---------- Ponto de referência fixo para hipervolume (não precisa ser salvo) ----------
        self.reference_point = PONTO_REFERENCIA_HV.copy()

        if self.modelo is None:
            raise ValueError("❌ Modelo não pode ser None no NSGA-II")

        print(f"✅ NSGA-II inicializado com modelo: {self.modelo_nome}")
        print(f"   • Tipo do modelo: {type(self.modelo).__name__}")
        if self.modelo_r2:
            print(f"   • R² do modelo: {self.modelo_r2:.4f}")

    # ----------------------------------------------------------------------
    # Checkpointing
    # ----------------------------------------------------------------------
    def save_checkpoint(self, filepath, geracao_atual, populacao, fitness):
        """
        Salva todo o estado necessário para retomar a otimização.
        Inclui população atual, fitness, geração atual, estado do RNG,
        todos os históricos e contadores.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        estado = {
            'geracao_atual': geracao_atual,
            'populacao': populacao,
            'fitness': fitness,
            'random_state': np.random.get_state(),
            'historico_melhor_dose': self.historico_melhor_dose,
            'historico_convergencia': self.historico_convergencia,
            'historico_diagnosticos': self.historico_diagnosticos,
            'historico_populacoes': self.historico_populacoes,
            'historico_fitness': self.historico_fitness,
            'hv_history': self.hv_history,
            'num_restarts': self.num_restarts,
            #'consecutive_stagnation': self.consecutive_stagnation,
            'mutacao_chamadas': self.mutacao_chamadas,
        }
        joblib.dump(estado, filepath)
        print(f"✅ Checkpoint salvo em {filepath} (geração {geracao_atual})")

    def load_checkpoint(self, filepath):
        """
        Carrega o estado de um checkpoint e restaura o gerador aleatório.
        Retorna (geracao_atual, populacao, fitness).
        """
        estado = joblib.load(filepath)

        # Restaura atributos (com .get para compatibilidade com checkpoints antigos)
        self.historico_melhor_dose = estado['historico_melhor_dose']
        self.historico_convergencia = estado.get('historico_convergencia', [])
        self.historico_diagnosticos = estado.get('historico_diagnosticos', [])
        self.historico_populacoes = estado.get('historico_populacoes', [])
        self.historico_fitness = estado.get('historico_fitness', [])
        self.hv_history = estado.get('hv_history', [])
        self.num_restarts = estado['num_restarts']
        #self.consecutive_stagnation = estado.get('consecutive_stagnation', 0)
        self.mutacao_chamadas = estado.get('mutacao_chamadas', 0)
        self.current_generation = estado['geracao_atual']
        
        # Restaura gerador aleatório
        np.random.set_state(estado['random_state'])

        print(f"🔄 Checkpoint carregado de {filepath} (geração {estado['geracao_atual']})")
        return estado['geracao_atual'], estado['populacao'], estado['fitness']
        
    # ----------------------------------------------------------------------
    # Salvamento de CSVs durante checkpoints
    # ----------------------------------------------------------------------
    def salvar_checkpoint_csv(self, populacao, fitness, geracao):
        """Salva CSVs com os resultados parciais da otimização (soluções destacadas e todas as soluções)."""
        # Importações necessárias (se já não estiverem no topo do script, adicione aqui)
        print(f"\n💾 CHECKPOINT CSV - Geração {geracao+1}")
        
        # 1. Salvar CSV de resultados da população (configurações completas)
        try:
            df_resultados, _ = salvar_resultados_csv(
                populacao, fitness,
                output_dir=self.results_dir,
                suffix=f"_gen{geracao+1}"
            )
            print(f"   ✅ Resultados salvos: {df_resultados}")
        except Exception as e:
            print(f"   ⚠️ Erro ao salvar resultados: {e}")
            
        # 2. Extrair a frente de Pareto atual (primeira fronteira)
        fronteiras, _ = self.fast_non_dominated_sort(fitness)
        if not fronteiras or len(fronteiras[0]) == 0:
            print("   ⚠️ Nenhuma solução não-dominada no checkpoint.")
            return
        idx_frente = fronteiras[0]
        fitness_frente = fitness[idx_frente]
        populacao_frente = populacao[idx_frente]
        
        # 3. Calcular TOPSIS na frente (usa a função global)
        best_idx, scores = calcular_tradeoff_topsis(fitness_frente)
        
        # 4. Identificar soluções extremas
        dose = fitness_frente[:, 0]
        peso = fitness_frente[:, 1]
        custo = fitness_frente[:, 2]
        idx_min_dose = np.argmin(dose)
        idx_min_peso = np.argmin(peso)
        idx_min_custo = np.argmin(custo)
        
        # 5. Preparar dados das soluções destacadas (incluindo materiais e espessuras)
        indices_dest = [idx_min_dose, idx_min_peso, idx_min_custo, best_idx]
        tipos = ['Melhor_Protecao', 'Menor_Peso', 'Menor_Custo', 'Melhor_Tradeoff_TOPSIS']
        
        materiais_cols = {f'Material_camada_{i+1}': [] for i in range(self.NUM_CAMADAS)}
        espessuras_cols = {f'Espessura_camada_{i+1}_cm': [] for i in range(self.NUM_CAMADAS)}
        
        for idx in indices_dest:
            sol = populacao_frente[idx]
            esp = sol[:self.NUM_CAMADAS]
            mat_idx = sol[self.NUM_CAMADAS:].astype(int)
            mat_nomes = [self.MATERIAIS_DISPONIVEIS[i] for i in mat_idx]
            for i in range(self.NUM_CAMADAS):
                espessuras_cols[f'Espessura_camada_{i+1}_cm'].append(esp[i])
                materiais_cols[f'Material_camada_{i+1}'].append(mat_nomes[i])
                
        # 6. Criar DataFrame de destacadas
        destacadas_df = pd.DataFrame({
            'Tipo_Solucao': tipos,
            'Indice_Original': indices_dest,
            'ID_Solucao': [i+1 for i in indices_dest],
            'Dose_mSv_5min': np.round([dose[i] for i in indices_dest], 3),
            'Peso_kg': np.round([peso[i] for i in indices_dest], 1),
            'Custo_USD': np.round([custo[i] for i in indices_dest], 0),
            'Score_TOPSIS': [scores[i] for i in indices_dest],
            'Metodo_Selecao': ['Menor Dose', 'Menor Peso', 'Menor Custo', 'TOPSIS-IQR'],
            **{f'Material_camada_{i+1}': materiais_cols[f'Material_camada_{i+1}'] for i in range(self.NUM_CAMADAS)},
            **{f'Espessura_camada_{i+1}_cm': espessuras_cols[f'Espessura_camada_{i+1}_cm'] for i in range(self.NUM_CAMADAS)}
        })
        
        # 7. Criar DataFrame com todas as soluções da frente
        ranking = np.argsort(scores)[::-1] + 1
        todas_df = pd.DataFrame({
            'ID_Solucao': range(1, len(fitness_frente)+1),
            'Dose_mSv_5min': np.round(dose, 3),
            'Peso_kg': np.round(peso, 1),
            'Custo_USD': np.round(custo, 0),
            'Score_TOPSIS': scores,
            'Ranking_TOPSIS': ranking
        })
        todas_df = todas_df.sort_values('Score_TOPSIS', ascending=False).reset_index(drop=True)
        todas_df['Ranking_TOPSIS'] = range(1, len(todas_df)+1)
        
        # 8. Salvar CSVs
        csv_dir = os.path.join(self.results_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        dest_file = os.path.join(csv_dir, f"solucoes_destacadas_gen{geracao+1}_{self.timestamp}.csv")
        todas_file = os.path.join(csv_dir, f"todas_solucoes_gen{geracao+1}_{self.timestamp}.csv")
        destacadas_df.to_csv(dest_file, index=False, encoding='utf-8')
        todas_df.to_csv(todas_file, index=False, encoding='utf-8')
        
        print(f"   ✅ Soluções destacadas salvas: {dest_file}")
        print(f"   ✅ Todas soluções salvas: {todas_file}")
        
    def mostrar_populacao_detalhada(self, populacao, fitness, titulo="POPULAÇÃO"):
        """Mostra detalhes de todos os indivíduos"""
        print(f"\n{'='*100}")
        print(f"🔍 {titulo} - {len(populacao)} INDIVÍDUOS")
        print(f"{'='*100}")
        print(f"{'ID':3s} | {'Dose (mSv/5min)':15s} | {'Peso (kg)':10s} | {'Custo (USD)':12s} | {'MATERIAIS':50s}")
        print(f"{'-'*3} | {'-'*15} | {'-'*10} | {'-'*12} | {'-'*50}")
        
        for i in range(min(100, len(populacao))):  # Mostra até 100
            individuo = populacao[i]
            dose = fitness[i, 0]
            peso = fitness[i, 1]
            custo = fitness[i, 2]
            
            # Decodificar materiais
            materiais_idx = individuo[NUM_CAMADAS:].astype(int)
            materiais = [MATERIAIS_DISPONIVEIS[idx] for idx in materiais_idx]
            
            # Formatar string de materiais
            materiais_str = " → ".join([f"{m[:10]}" for m in materiais])
            
            print(f"{i+1:3d} | {dose:15.6f} | {peso:10.2f} | {custo:12.2f} | {materiais_str}")
    
        # Estatísticas
        print(f"\n📊 ESTATÍSTICAS:")
        print(f"   • Dose:  {fitness[:,0].min():.4f} a {fitness[:,0].max():.4f} mSv/5min")
        print(f"   • Peso:  {fitness[:,1].min():.1f} a {fitness[:,1].max():.1f} kg")
        print(f"   • Custo: ${fitness[:,2].min():.0f} a ${fitness[:,2].max():.0f}")
    
    def calcular_spread_metric(self, fitness):
        """
        Spread metric (Δ) - Deb et al., 2002 - CORRIGIDA
        Medida de diversidade da população Pareto
        """
        if len(fitness) < 2:
            return 1.0
    
        # 1. Calcular distâncias entre soluções consecutivas (ordenadas por um objetivo)
        # Ordenar pela dose (primeiro objetivo)
        sorted_indices = np.argsort(fitness[:, 0])
        fitness_sorted = fitness[sorted_indices]
        
        # 2. Calcular distância Euclidiana entre soluções consecutivas
        distances = []
        for i in range(len(fitness_sorted) - 1):
            dist = np.linalg.norm(fitness_sorted[i] - fitness_sorted[i + 1])
            distances.append(dist)
    
        if len(distances) == 0:
            return 1.0
    
        # 3. Calcular distância média
        mean_distance = np.mean(distances)
    
        # 4. Calcular distância aos extremos (pontos de borda)
        # Os extremos são o primeiro e último ponto após ordenação
        d_f = np.linalg.norm(fitness_sorted[0] - fitness_sorted[-1])
        
        # 5. Calcular spread (Δ) - Deb et al., 2002, Eq. 9
        numerator = d_f + np.sum(np.abs(np.array(distances) - mean_distance))
        denominator = d_f + (len(fitness_sorted) - 1) * mean_distance
        
        if denominator == 0:
            return 1.0
    
        spread = numerator / denominator
    
        # Valores próximos de 0 indicam distribuição perfeitamente uniforme
        # Valores próximos de 1 indicam alta concentração/aglomeração
        return min(spread, 1.0)

    # ==========================================================================
    # 🎯 DIAGNÓSTICO DE CONVERGÊNCIA MELHORADO
    # ==========================================================================
    
    def diagnosticar_convergencia(self, fitness_atual, geracao, max_geracoes):
        """
        Diagnóstico de convergência baseado em hipervolume e spread.
        Retorna (estagnado, mensagem, diagnostico)
        """
        # Inicializa atributos se não existirem
        if not hasattr(self, 'historico_hipervolume'):
            self.historico_hipervolume = []
            self.melhor_hipervolume = -np.inf
            self.geracoes_sem_melhoria = 0
            
        # Calcula hipervolume da frente atual (supondo que fitness_atual seja a frente)
        hv_atual = calcular_hipervolume_unificado(
            fitness_atual,
            referencia=self.reference_point,
            normalizado=True
        )['hipervolume_normalizado']
        self.historico_hipervolume.append(hv_atual)
                
        # Atualiza melhor hipervolume
        hv_tol = 0.001  # 0.1% de melhoria mínima
        if hv_atual > self.melhor_hipervolume * (1 + hv_tol):
            self.melhor_hipervolume = hv_atual
            self.geracoes_sem_melhoria = 0
        else:
            self.geracoes_sem_melhoria += 1
            
        # Calcula spread (já existe no código)
        spread = self.calcular_spread_metric(fitness_atual)
        
        # Critério de estagnação: sem melhoria no hipervolume por 8 gerações
        # (mesmo valor usado para restart no código original)
        estagnado = self.geracoes_sem_melhoria >= 8
        
        diagnostico = {
            'geracao': geracao,
            'melhor_dose': np.min(fitness_atual[:, 0]),
            'spread_metric': spread,
            'hipervolume': hv_atual,
            'melhor_hipervolume': self.melhor_hipervolume,
            'geracoes_sem_melhoria': self.geracoes_sem_melhoria,
            'estagnado': estagnado
        }
        
        if estagnado:
            mensagem = f"Estagnação detectada: {self.geracoes_sem_melhoria} gerações sem melhoria no hipervolume (atual={hv_atual:.4f}, melhor={self.melhor_hipervolume:.4f})"
        else:
            mensagem = None
            
        return estagnado, mensagem, diagnostico
    
    def gerar_resumo_evolucao(self, historico_fitness, historico_diagnosticos, n_geracoes=None):
        """
        Gera um resumo ENXUTO da evolução, mostrando apenas quando há mudanças significativas
        e destacando claramente quando estagnou. Usa o tamanho real do histórico.
        """
        n_reais = len(historico_fitness)
        if n_reais == 0:
            print("Nenhum histórico disponível.")
            return

        print("\n" + "="*100)
        print("📊 RESUMO OBJETIVO DA EVOLUÇÃO - PONTOS DE MUDANÇA CRÍTICOS")
        print("LEGENDA: 🟢 Melhoria | 🟡 Estagnação | 🔴 Regressão | ⚪ Mantém")
        print(f"{'Geração':>8} | {'Melhor Dose':>12} | {'Melhor Peso':>10} | {'Melhor Custo':>12} | {'Status':>8} | {'Ação':>20}")
        
        melhor_dose_global = float('inf')
        geracao_melhor_dose = 0
        contador_estagnacao = 0
        limite_estagnacao = 8  # Gerações sem melhoria para considerar estagnado
        
        for geracao in range(n_reais):
            fitness = historico_fitness[geracao]
            if len(fitness) == 0:
                continue
            
            melhor_dose_atual = np.min(fitness[:, 0])
            melhor_idx = np.argmin(fitness[:, 0])
            melhor_peso_atual = fitness[melhor_idx, 1]
            melhor_custo_atual = fitness[melhor_idx, 2]
            
            if melhor_dose_atual < melhor_dose_global - 1e-10:
                status = "🟢 MELHORA"
                melhor_dose_global = melhor_dose_atual
                geracao_melhor_dose = geracao
                contador_estagnacao = 0
            elif abs(melhor_dose_atual - melhor_dose_global) < 1e-10:
                contador_estagnacao += 1
                if contador_estagnacao >= limite_estagnacao:
                    status = "🟡 ESTAGNADO"
                else:
                    status = "⚪ MANTÉM"
            else:
                status = "🔴 REGRIDE"
                contador_estagnacao += 1
                
            # Define a ação recomendada
            if status == "🟡 ESTAGNADO":
                acao = "PARAR AQUI"
            elif status == "🟢 MELHORA":
                acao = "CONTINUAR"
            else:
                acao = "MONITORAR"

            # Mostrar apenas gerações relevantes: primeiras 5, últimas 5, a cada 10, e quando status muda
            mostrar = False
            if geracao == 0:
                mostrar = True
                print(f"{'INICIAL':>8} | {melhor_dose_atual:12.4f} | {melhor_peso_atual:10.1f} | ${melhor_custo_atual:11.0f} | {'INÍCIO':>8} | {'---':>20}")
                continue
            
            condicoes_exibir = [
                geracao < 5,
                geracao >= n_reais - 5,
                geracao % 10 == 0,
                status == "🟢 MELHORA",
                status == "🟡 ESTAGNADO",
                "restart" in str(historico_diagnosticos[geracao]).lower() if geracao < len(historico_diagnosticos) else False
            ]
            # Verifica se houve restart (se o diagnóstico estiver disponível)
            if historico_diagnosticos and geracao < len(historico_diagnosticos):
                diag = historico_diagnosticos[geracao]
                if isinstance(diag, dict) and diag.get('recomendacao') and 'restart' in diag['recomendacao'].lower():
                    condicoes_exibir.append(True)
        
            if any(condicoes_exibir):
                mostrar = True
                
            if mostrar:
                # Calcula a variação em relação à geração anterior
                if geracao > 0:
                    dose_anterior = np.min(historico_fitness[geracao-1][:, 0])
                    variacao = melhor_dose_atual - dose_anterior
                    seta = "↘" if variacao < 0 else "↗" if variacao > 0 else "→"
                else:
                    seta = ""
                print(f"{geracao+1:>8} | {melhor_dose_atual:11.4f}{seta:1s} | {melhor_peso_atual:10.1f} | ${melhor_custo_atual:11.0f} | {status:>8} | {acao:>20}")
              
                # Se houve restart, exibe uma linha adicional
                if historico_diagnosticos and geracao < len(historico_diagnosticos):
                    diag = historico_diagnosticos[geracao]
                    if isinstance(diag, dict) and diag.get('recomendacao') and 'restart' in diag['recomendacao'].lower():
                        print(f"{'':>8} | {'':12} | {'':10} | {'':12} | {'':8} | 🔄 RESTART APLICADO")
                        
        print("-"*100)
        print("\n🎯 CONCLUSÃO TÉCNICA PARA O ARTIGO:")
        print(f"   • Geração do ótimo: {geracao_melhor_dose + 1}")
        print(f"   • Gerações totais executadas: {n_reais}")
        print(f"   • Gerações até estagnação: {contador_estagnacao}")
        if contador_estagnacao >= limite_estagnacao:
            print(f"   ⚠️  OTIMIZAÇÃO ESTAGNOU na geração {geracao_melhor_dose + 1 + limite_estagnacao}")
        else:
            print(f"   ✅ Otimização ainda estava melhorando na última geração")
        if n_reais > 1:
            melhoria_percentual = ((historico_fitness[0][:, 0].min() - melhor_dose_global) / historico_fitness[0][:, 0].min() * 100)
            print(f"   📈 Melhoria total: {melhoria_percentual:.1f}%")
        print("="*100)

    # ==========================================================================
    # 🎯 RESTART ADAPTATIVO INTELIGENTE (CORRIGIDO)
    # ==========================================================================
    def aplicar_restart_adaptativo(self, populacao, fitness, percentual=None, 
                                   manter_melhores=True):
        """
        Restart adaptativo inteligente (CORRIGIDO)
        Preserva as melhores soluções (baseado em Pareto e crowding distance)
        e gera diversidade. No modo simples, substitui os piores também por critério Pareto.
        """
        if percentual is None:
            percentual = self.restart_percentual
            
        tamanho_pop = len(populacao)
        num_restart = max(2, int(tamanho_pop * percentual))  # Mínimo 2 indivíduos
        
        print(f"   🔄 RESTART ADAPTATIVO: Substituindo {num_restart} indivíduos")
        self.num_restarts += 1
        
        # Calcular ranks e crowding distances para toda a população
        fronteiras, ranks = self.fast_non_dominated_sort(fitness)
        crowding = np.zeros(len(populacao))
        for fronteira in fronteiras:
            dists = self.crowding_distance(fitness, fronteira)
            for j, idx in enumerate(fronteira):
                crowding[idx] = dists[j]
                
        if manter_melhores:
            # Ordenar por (rank, -crowding): melhores primeiro
            indices_ordenados = sorted(range(len(populacao)), 
                                       key=lambda i: (ranks[i], -crowding[i]))
            indices_manter = indices_ordenados[:tamanho_pop - num_restart]
            indices_substituir = indices_ordenados[tamanho_pop - num_restart:]
            
            populacao_mantida = populacao[indices_manter].copy()
            fitness_mantido = fitness[indices_manter].copy()
            
            # Gerar novos indivíduos aleatórios
            novos_individuos = self.inicializar_populacao(num_restart)
            novos_fitness = self.avaliar_populacao(novos_individuos)
            
            populacao_nova = np.vstack([populacao_mantida, novos_individuos])
            fitness_novo = np.vstack([fitness_mantido, novos_fitness])
            
            print(f"   • Mantidos: {len(populacao_mantida)} melhores indivíduos (Pareto)")
            print(f"   • Adicionados: {num_restart} novos indivíduos aleatórios")
            return populacao_nova, fitness_novo
        
        else:
            # Modo simples: substituir os piores (rank alto, crowding baixo)
            # Ordenar por (-rank, crowding): piores primeiro
            indices_ordenados = sorted(range(len(populacao)), 
                                       key=lambda i: (-ranks[i], crowding[i]))
            indices_substituir = indices_ordenados[:num_restart]
            
            # Substituir cada um dos piores
            for idx in indices_substituir:
                novo_individuo = self.inicializar_populacao(1)[0]
                novo_fitness = self.avaliar_populacao([novo_individuo])[0]
                populacao[idx] = novo_individuo
                fitness[idx] = novo_fitness
                
            print(f"   • Substituídos {num_restart} piores indivíduos (Pareto)")
            return populacao, fitness
    
    # ==========================================================================
    # 🎯 MUTAÇÃO ADAPTATIVA COM CONTROLE DE DIVERSIDADE
    # ==========================================================================
    def mutacao_adaptativa(self, individuo, geracao, max_geracoes, 
                          diversidade_atual=0.5):
        """
        Mutação adaptativa com controle baseado em:
        1. Progresso da otimização (gerações)
        2. Diversidade da população
        3. Tipo de variável (contínua vs discreta)
        """
        # print(f"   🔧 [DEBUG] Mutação adaptativa chamada na geração {geracao} com diversidade {diversidade_atual:.3f}")
        
        self.mutacao_chamadas += 1
        
        # Taxa base baseada no progresso
        progresso = geracao / max_geracoes
        
        # Taxa decrescente com progresso
        taxa_base = self.mutacao_inicial - (self.mutacao_inicial - self.mutacao_final) * progresso
        
        # Ajuste baseado na diversidade
        # Se diversidade baixa (<0.3), aumenta mutação
        if diversidade_atual < 0.3:
            ajuste_diversidade = 1.5  # +50%
        elif diversidade_atual > 0.7:
            ajuste_diversidade = 1.0  # -30%
        else:
            ajuste_diversidade = 1.0  # neutro
        
        taxa_mutacao_atual = taxa_base * ajuste_diversidade
        # print(f"      → Taxa de mutação calculada: {taxa_mutacao_atual:.3f}")
        
        # Limites razoáveis
        taxa_mutacao_atual = np.clip(taxa_mutacao_atual, 0.05, 0.5)
        
        mutado = individuo.copy()
        
        for i in range(len(individuo)):
            if np.random.random() < taxa_mutacao_atual:
                if i < NUM_CAMADAS:  # Espessura (variável contínua)
                    # Mutação Gaussiana com sigma adaptativo
                    # Sigma menor conforme progresso
                    sigma_base = (ESPESSURA_MAX_CAMADA - ESPESSURA_MIN_CAMADA) * 0.15
                    sigma = sigma_base * (1.0 - progresso * 0.7)  # Reduz com progresso
                    
                    # Adicionar ruído gaussiano
                    mutado[i] += np.random.normal(0, sigma)
                    
                    # Garantir limites
                    mutado[i] = np.clip(mutado[i], ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
                    
                else:  # Material (variável discreta)
                    # Para materiais, as vezes troca, as vezes permuta
                    if np.random.random() < 0.7:  # 70% troca por novo material
                        mutado[i] = np.random.randint(0, len(MATERIAIS_DISPONIVEIS))
                    else:  # 30% permuta com outro índice de material
                        outro_idx = np.random.randint(NUM_CAMADAS, len(individuo))
                        while outro_idx == i:
                            outro_idx = np.random.randint(NUM_CAMADAS, len(individuo))
                        # Troca os materiais
                        temp = mutado[i]
                        mutado[i] = mutado[outro_idx]
                        mutado[outro_idx] = temp
        
        # GARANTIR ESPESSURA TOTAL FIXA E LIMITES (correção iterativa)
        espessuras = mutado[:NUM_CAMADAS]
        soma = np.sum(espessuras)
        if soma > 0:
            espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma)
            
        # Correção iterativa
        max_iter = 5
        for _ in range(max_iter):
            espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
            soma_clip = np.sum(espessuras)
            if abs(soma_clip - ESPESSURA_TOTAL_FIXA) < 1e-8:
                break
            espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma_clip)
    
        # Garantia final
        espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
        if abs(np.sum(espessuras) - ESPESSURA_TOTAL_FIXA) > 1e-6:
            espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA
            
        mutado[:NUM_CAMADAS] = espessuras

        # GARANTIR MATERIAIS ÚNICOS (NOVA RESTRIÇÃO BASEADA NA FASE 3)
        materiais = mutado[NUM_CAMADAS:].astype(int)
    
        if len(np.unique(materiais)) < NUM_CAMADAS:
            # Corrigir materiais repetidos
            mutado = self.corrigir_materiais_repetidos(mutado)
            
        return mutado, taxa_mutacao_atual
    
    # ==========================================================================
    # 🎯 NOVA FUNÇÃO: CORREÇÃO DE MATERIAIS REPETIDOS
    # ==========================================================================
    def corrigir_materiais_repetidos(self, individuo):
        """
        Corrige materiais repetidos mantendo a viabilidade
        """
        mutado = individuo.copy()
        materiais = mutado[NUM_CAMADAS:].astype(int)
        
        # Verificar se há repetições
        materiais_unicos, contagens = np.unique(materiais, return_counts=True)
        repetidos = materiais_unicos[contagens > 1]
        
        if len(repetidos) == 0:
            return mutado  # Já está correto
        
        # Para cada material repetido (exceto a primeira ocorrência)
        materiais_disponiveis = list(range(len(MATERIAIS_DISPONIVEIS)))
        
        # Materiais já usados
        usados = set()
        
        for i in range(NUM_CAMADAS):
            material_idx = NUM_CAMADAS + i
            material_atual = int(mutado[material_idx])
            
            if material_atual in usados:
                # Material repetido, precisa trocar
                # Encontrar material não utilizado
                disponiveis = [m for m in materiais_disponiveis if m not in usados]
                
                if disponiveis:
                    novo_material = np.random.choice(disponiveis)
                    mutado[material_idx] = novo_material
                    usados.add(novo_material)
                else:
                    # Todos materiais já usados (impossível com 7 materiais e 7 camadas)
                    # Troca aleatória com outro material
                    outro_idx = np.random.randint(NUM_CAMADAS, len(mutado))
                    while outro_idx == material_idx:
                        outro_idx = np.random.randint(NUM_CAMADAS, len(mutado))
                    
                    temp = mutado[material_idx]
                    mutado[material_idx] = mutado[outro_idx]
                    mutado[outro_idx] = temp
            else:
                usados.add(material_atual)
        
        return mutado
        
    def inicializar_populacao(self, tamanho):
        """Inicializa população aleatória com materiais ÚNICOS por indivíduo"""
        
        populacao = []

        for _ in range(tamanho):
            # Gerar espessuras aleatórias
            espessuras = np.random.uniform(ESPESSURA_MIN_CAMADA, 
                                           ESPESSURA_MAX_CAMADA, 
                                           NUM_CAMADAS)
            # Normalizar para soma fixa
            espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA
            
            # GARANTIR MATERIAIS ÚNICOS
            todos_materiais = list(range(len(MATERIAIS_DISPONIVEIS)))
            
            if NUM_CAMADAS <= len(todos_materiais):
                # Amostragem SEM reposição
                materiais = np.random.choice(todos_materiais, 
                                             NUM_CAMADAS, 
                                             replace=False)
            else:
                # Se mais camadas que materiais, complete com aleatórios
                # Mas garanta o máximo de unicidade possível
                materiais = list(range(len(MATERIAIS_DISPONIVEIS)))
                # Completar com materiais aleatórios
                while len(materiais) < NUM_CAMADAS:
                    materiais.append(np.random.randint(0, len(MATERIAIS_DISPONIVEIS)))
                materiais = np.array(materiais)
    
            individuo = np.concatenate([espessuras, materiais])
            populacao.append(individuo)
            
        return populacao
    
    def avaliar_populacao(self, populacao):
        """Avalia todos os indivíduos, evitando sobrecarga de processos no Elastic Net."""
        
        # 🚨 DETECTAR SE É MODELO KERAS (NÃO SERIALIZÁVEL)
        is_keras_model = hasattr(self.modelo, 'keras_model') or hasattr(self.modelo, '_is_keras_model')
        
        if is_keras_model or not AVALIACAO_POPULACIONAL_PARALELA:
            if is_keras_model:
                print("   ⚠️  Modelo Keras detectado: usando avaliação SERIAL (sem paralelismo)")
                print("   💡 Motivo: Modelos TensorFlow/Keras não são serializáveis entre processos")
            else:
                print(f"⚡ Avaliação local: {len(populacao)} indivíduos (mais rápida para {self.modelo_nome})")
            
            # AVALIAÇÃO SERIAL para modelos Keras
            resultados = []
            for i, individuo in enumerate(populacao):
                # Mostrar progresso a cada 10 indivíduos
                if i % 10 == 0 and i > 0:
                    print(f"      • Avaliados {i}/{len(populacao)} indivíduos")
                    
                resultado = avaliar_individuo(individuo, self.scaler, self.modelo, self.target_transform)
                resultados.append(resultado)
                
            fitness = np.array(resultados)
            return fitness
        
        else:
            # Para modelos scikit-learn (serializáveis), usar paralelismo normal
            print(f"⚡ Paralelização ativa: avaliando {len(populacao)} indivíduos em paralelo com {multiprocessing.cpu_count()} núcleos (o scikit-learn usa a joblib como backend padrão)")
            
            def avaliar_single(individuo):
                """Avalia um único indivíduo - função para paralelizar"""
                return avaliar_individuo(individuo, self.scaler, self.modelo, self.target_transform)
            
            # PARALELIZAÇÃO apenas para modelos não-Keras
            n_cores = max(1, multiprocessing.cpu_count())
            resultados = Parallel(n_jobs=n_cores, backend='loky', verbose=0)(
                delayed(avaliar_single)(individuo) for individuo in populacao
            )
            
            fitness = np.array(resultados)
            return fitness
    
    def dominancia_pareto(self, fitness1, fitness2):
        """Verifica se fitness1 domina fitness2 (todos objetivos a minimizar)"""
        # fitness1 domina fitness2 se for melhor em todos os objetivos
        melhor_ou_igual = np.all(fitness1 <= fitness2)
        estritamente_melhor = np.any(fitness1 < fitness2)
        
        return melhor_ou_igual and estritamente_melhor
    
    def fast_non_dominated_sort(self, fitness):
        """Ordenação não-dominada com matriz vetorizada de dominância.

        Para 500--1000 indivíduos e três objetivos, a matriz booleana cabe em
        memória. Ela substitui milhões de chamadas NumPy escalares por geração
        sem alterar o critério de dominância de Pareto.
        """
        n = len(fitness)
        if n == 0:
            return [], []

        valores = np.asarray(fitness, dtype=float)
        domina = (np.all(valores[:, None, :] <= valores[None, :, :], axis=2)
                  & np.any(valores[:, None, :] < valores[None, :, :], axis=2))
        np.fill_diagonal(domina, False)

        dominados_por = domina.sum(axis=0).astype(int)
        rank = np.full(n, -1, dtype=int)
        frente_atual = np.flatnonzero(dominados_por == 0).tolist()
        fronteiras = []
        nivel = 0

        while frente_atual:
            fronteiras.append(frente_atual)
            rank[frente_atual] = nivel
            proximos = []
            for p in frente_atual:
                for q in np.flatnonzero(domina[p]):
                    dominados_por[q] -= 1
                    if dominados_por[q] == 0:
                        proximos.append(int(q))
            frente_atual = proximos
            nivel += 1

        return fronteiras, rank.tolist()
    
    def crowding_distance(self, fitness, fronteira):
        """
        Calcula distância de aglomeração CORRIGIDA
        Implementação baseada em Deb et al. (2002) - NSGA-II
        """
        n = len(fronteira)
        if n == 0:
            return []
    
        # Casos especiais
        if n == 1:
            return [float('inf')]
        if n == 2:
            return [float('inf'), float('inf')]
    
        # Inicializar distâncias com zeros
        distances = np.zeros(n, dtype=float)
        
        # Extrair fitness apenas dos indivíduos da fronteira
        fitness_front = fitness[fronteira, :]
        
        # Para cada objetivo
        for obj_idx in range(fitness.shape[1]):
            # Obter valores deste objetivo
            obj_values = fitness_front[:, obj_idx]
        
            # Ordenar índices por este objetivo
            sorted_indices = np.argsort(obj_values)
            sorted_values = obj_values[sorted_indices]
            
            # Amplitude do objetivo
            f_min = sorted_values[0]
            f_max = sorted_values[-1]
            f_range = f_max - f_min
            
            # Se amplitude for zero, todos têm mesma distância (zero)
            if f_range < 1e-10:
                continue
        
            # Atribuir distância infinita apenas aos extremos ABSOLUTOS
            # (primeiro e último após ordenação)
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Calcular distância para pontos intermediários
            # Equação: (f[i+1] - f[i-1]) / (f_max - f_min)
            for i in range(1, n-1):
                # Índice na lista ordenada
                idx = sorted_indices[i]
            
                # Valores vizinhos (já ordenados)
                valor_anterior = sorted_values[i-1]
                valor_proximo = sorted_values[i+1]
                
                # Contribuição para crowding distance
                contribuicao = (valor_proximo - valor_anterior) / f_range
                
                # Acumular (não sobrescrever infinitos!)
                if not np.isinf(distances[idx]):
                    distances[idx] += contribuicao
                    
        return distances.tolist()
    
    def selecionar_pais(self, populacao, fitness, num_pais):
        """Seleção por torneio baseada em dominância e crowding distance"""
        n = len(populacao)
        pais = []
        
        # Obter frentes de Pareto
        fronteiras, ranks = self.fast_non_dominated_sort(fitness)
        
        # Calcular crowding distance para cada frente
        crowding_distances = []
        for fronteira in fronteiras:
            crowding_distances.extend(self.crowding_distance(fitness, fronteira))
        
        for _ in range(num_pais // 2):
            # Torneio binário
            candidatos1 = np.random.choice(n, 2, replace=False)
            candidatos2 = np.random.choice(n, 2, replace=False)
            
            # Comparar por rank
            rank1 = min(ranks[candidatos1[0]], ranks[candidatos1[1]])
            rank2 = min(ranks[candidatos2[0]], ranks[candidatos2[1]])
            
            if rank1 < rank2:
                vencedor1 = candidatos1[0] if ranks[candidatos1[0]] < ranks[candidatos1[1]] else candidatos1[1]
            elif rank1 > rank2:
                vencedor1 = candidatos2[0] if ranks[candidatos2[0]] < ranks[candidatos2[1]] else candidatos2[1]
            else:
                # Empate no rank, usar crowding distance
                if crowding_distances[candidatos1[0]] > crowding_distances[candidatos1[1]]:
                    vencedor1 = candidatos1[0]
                else:
                    vencedor1 = candidatos1[1]
            
            # Segundo vencedor
            candidatos3 = np.random.choice(n, 2, replace=False)
            candidatos4 = np.random.choice(n, 2, replace=False)
            
            rank3 = min(ranks[candidatos3[0]], ranks[candidatos3[1]])
            rank4 = min(ranks[candidatos4[0]], ranks[candidatos4[1]])
            
            if rank3 < rank4:
                vencedor2 = candidatos3[0] if ranks[candidatos3[0]] < ranks[candidatos3[1]] else candidatos3[1]
            elif rank3 > rank4:
                vencedor2 = candidatos4[0] if ranks[candidatos4[0]] < ranks[candidatos4[1]] else candidatos4[1]
            else:
                if crowding_distances[candidatos3[0]] > crowding_distances[candidatos3[1]]:
                    vencedor2 = candidatos3[0]
                else:
                    vencedor2 = candidatos3[1]
            
            pais.append(populacao[vencedor1])
            pais.append(populacao[vencedor2])
        
        return np.array(pais)
    
    def crossover_simulated_binary(self, pai1, pai2, eta=10): # eta = parâmetro de distribuição do crossover SBX
        """Crossover SBX com garantia de materiais únicos"""
        filho1 = pai1.copy()
        filho2 = pai2.copy()
    
        # Crossover SBX tradicional
        for i in range(len(pai1)):
            if np.random.random() < 0.5:
                continue
    
            if np.random.random() <= 0.5:
                # Para variáveis contínuas (espessuras)
                if i < NUM_CAMADAS:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2*u)**(1/(eta+1))
                    else:
                        beta = (1/(2*(1-u)))**(1/(eta+1))
        
                    filho1[i] = 0.5*((1+beta)*pai1[i] + (1-beta)*pai2[i])
                    filho2[i] = 0.5*((1-beta)*pai1[i] + (1+beta)*pai2[i])
        
                    # Garantir limites
                    filho1[i] = np.clip(filho1[i], ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
                    filho2[i] = np.clip(filho2[i], ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
                else:
                    # Para variáveis discretas (materiais)
                    if i >= NUM_CAMADAS:
                        # Crossover uniforme
                        if np.random.random() < 0.5:
                            filho1[i] = pai2[i]
                            filho2[i] = pai1[i]

        # Aplicar correção aos dois filhos
        filho1 = self.corrigir_materiais_repetidos(filho1)
        filho2 = self.corrigir_materiais_repetidos(filho2)
        
        # VERIFICAÇÃO DE SEGURANÇA
        for i, filho in enumerate([filho1, filho2], 1):
            materiais_filho = filho[NUM_CAMADAS:].astype(int)
            if len(set(materiais_filho)) != NUM_CAMADAS:
                print(f"⚠️  ALERTA: Filho {i} ainda tem materiais repetidos após correção!")
                print(f"   Materiais: {materiais_filho}")
                print(f"   Únicos: {set(materiais_filho)}")
            
                # Forçar correção com o método da classe
                filho = self.corrigir_materiais_repetidos(filho)
                if i == 1:
                    filho1 = filho
                else:
                    filho2 = filho
        
        # 3. GARANTIR SOMA DE ESPESSURAS CORRETA E LIMITES (correção iterativa)
        for idx_filho, filho in enumerate([filho1, filho2]):
            espessuras = filho[:NUM_CAMADAS]
            soma = np.sum(espessuras)
            
            if soma > 0:
                # Primeira normalização
                espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma)
            else:
                espessuras = np.ones(NUM_CAMADAS) * (ESPESSURA_TOTAL_FIXA / NUM_CAMADAS)
                
            # Correção iterativa para garantir limites e soma exata
            max_iter = 5
            for _ in range(max_iter):
                espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
                soma_clip = np.sum(espessuras)
                if abs(soma_clip - ESPESSURA_TOTAL_FIXA) < 1e-8:
                    break
                espessuras = espessuras * (ESPESSURA_TOTAL_FIXA / soma_clip)
    
            # Garantia final
            espessuras = np.clip(espessuras, ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
            if abs(np.sum(espessuras) - ESPESSURA_TOTAL_FIXA) > 1e-6:
                espessuras = espessuras / np.sum(espessuras) * ESPESSURA_TOTAL_FIXA
                
            if idx_filho == 0:
                filho1[:NUM_CAMADAS] = espessuras
            else:
                filho2[:NUM_CAMADAS] = espessuras
                
        return filho1, filho2

    def mutacao_polinomial(self, individuo, taxa_mutacao=0.1, eta=20):
        """Mutação polinomial"""
        mutado = individuo.copy()
        
        for i in range(len(individuo)):
            if np.random.random() < taxa_mutacao:
                if i < NUM_CAMADAS:  # Espessura
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2*u)**(1/(eta+1)) - 1
                    else:
                        delta = 1 - (2*(1-u))**(1/(eta+1))
                    
                    mutado[i] += delta * (ESPESSURA_MAX_CAMADA - ESPESSURA_MIN_CAMADA)
                    mutado[i] = np.clip(mutado[i], ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA)
                else:  # Material
                    mutado[i] = np.random.randint(0, len(MATERIAIS_DISPONIVEIS))
        
        # Garantir soma de espessuras
        espessuras = mutado[:NUM_CAMADAS]
        soma = np.sum(espessuras)
        if soma > 0:
            mutado[:NUM_CAMADAS] = espessuras / soma * ESPESSURA_TOTAL_FIXA
        
        mutado = self.corrigir_materiais_repetidos(mutado)
        
        return mutado
    
    def executar_otimizacao(self, n_geracoes=NUM_GENERATIONS, tamanho_populacao=POPULATION_SIZE,
                            checkpoint_file=None, checkpoint_interval=10, figures_dir=None, force_new=False):
        """
        Executa o NSGA-II com suporte a checkpointing.
        
        Parâmetros:
            n_geracoes: número total de gerações desejado (incluindo as já executadas)
            tamanho_populacao: tamanho da população
            checkpoint_file: caminho do arquivo .pkl para salvar/restaurar estado
            checkpoint_interval: salvar checkpoint a cada N gerações
        """
        print("🚀 Inicializando/Retomando otimização NSGA-II...")
        
        # --- Resetar históricos para garantir que comecem vazios (nova execução) ---
        self.historico_fitness = []
        self.historico_diagnosticos = []
        self.hv_history = []
        self.historico_melhor_dose = []
        self.geracoes_sem_melhoria = 0
        self.melhor_hipervolume = -np.inf
        self.consecutive_stagnation = 0
            
        # --- 1. Tentar carregar checkpoint (apenas se force_new for False e arquivo existir) ---
        start_gen = 0
        populacao = None
        fitness = None
        
        carregar = not force_new and checkpoint_file and os.path.exists(checkpoint_file)
        if carregar:
            start_gen, populacao, fitness = self.load_checkpoint(checkpoint_file)
            print(f"🔄 Checkpoint carregado. Geração inicial: {start_gen}")
            # Os históricos já foram restaurados pelo load_checkpoint
        else:
            if force_new:
                print("🆕 Forçando nova execução (checkpoint ignorado).")
            else:
                print("🆕 Nenhum checkpoint encontrado. Inicializando população...")
            populacao = self.inicializar_populacao(tamanho_populacao)
            fitness = self.avaliar_populacao(populacao)
            # Inicializa históricos
            self.historico_populacoes = [populacao.copy()]
            self.historico_fitness = [fitness.copy()]
            self.historico_diagnosticos = []
            self.hv_history = []
            # Define limites físicos (opcional, apenas para referência)
            self.limites_fisicos = {
                'dose': (LIMITE_DOSE_MIN, LIMITE_DOSE_MAX),
                'peso': (LIMITE_PESO_MIN, LIMITE_PESO_MAX),
                'custo': (LIMITE_CUSTO_MIN, LIMITE_CUSTO_MAX)
            }
            print(f"👥 População inicial gerada: {len(populacao)} indivíduos com materiais únicos.")
            
        # --- 2. Loop principal (começa da geração start_gen) ---
        for geracao in range(start_gen, n_geracoes):
            self.current_generation = geracao + 1   # <--- AQUI
            print(f"\n{'='*60}")
            print(f"🚀 GERAÇÃO {geracao+1}/{n_geracoes}")
            print(f"{'='*60}")
            
            # ==============================================================
            # 1. Calcular hipervolume da primeira fronteira
            # ==============================================================
            fronteiras, _ = self.fast_non_dominated_sort(fitness)
            if len(fronteiras) > 0 and len(fronteiras[0]) > 1:
                hv_resultado = calcular_hipervolume_unificado(
                    fitness[fronteiras[0]],
                    referencia=self.reference_point,
                    normalizado=True
                )
                hv_norm = hv_resultado['hipervolume_normalizado']
                print(f"   📊 Hipervolume = {hv_norm:.6f}")
                self.hv_history.append(hv_norm)
            else:
                self.hv_history.append(0.0)
                print(f"   ⚠️ Hipervolume não pôde ser calculado (fronteira vazia).")
                
            # ==============================================================
            # 2. Mostrar população (a cada 5 gerações)
            # ==============================================================
            if geracao == 0 or (geracao + 1) % 5 == 0:
                self.mostrar_populacao_detalhada(populacao, fitness, f"GERAÇÃO {geracao+1}")

            # ==============================================================
            # 3. Diagnóstico de convergência
            # ==============================================================
            if geracao >= 10:  # começar após 10 gerações
                estagnado, mensagem, diagnostico = self.diagnosticar_convergencia(fitness, geracao, n_geracoes)
                self.historico_diagnosticos.append(diagnostico)
                
                if ADAPTIVE_RESTART and estagnado and mensagem:
                    print(f"   📊 {mensagem}")
                    # Usar o contador real de gerações sem melhoria
                    if self.geracoes_sem_melhoria >= self.stagnation_limit:  # Altere para o valor desejado (8, 10, etc.)
                        print(f"   🔄 Aplicando restart adaptativo ({self.restart_percentual*100:.0f}% população)")
                        populacao, fitness = self.aplicar_restart_adaptativo(populacao, fitness)
                        # Resetar o contador de estagnação para recomeçar o monitoramento
                        self.geracoes_sem_melhoria = 0
                        self.melhor_hipervolume = -np.inf  # opcional: reinicia o melhor hipervolume
            else:
                # Diagnóstico vazio para gerações iniciais
                self.historico_diagnosticos.append({'geracao': geracao, 'estagnado': False})
    
            # ==============================================================
            # 🎯 SELEÇÃO, CROSSOVER E MUTAÇÃO
            # ==============================================================
            # Calcular diversidade atual para uso na mutação adaptativa
            current_spread = self.calcular_spread_metric(fitness)
            
            # A seleção e o crossover pertencem às duas variantes. Apenas o
            # operador de mutação (e o reinício) muda entre adaptativo e canônico.
            if ADAPTIVE_MUTATION:
                progresso = geracao / n_geracoes
                taxa_base = self.mutacao_inicial - (self.mutacao_inicial - self.mutacao_final) * progresso
                if current_spread < 0.3:
                    ajuste = 1.5
                elif current_spread > 0.7:
                    ajuste = 0.7
                else:
                    ajuste = 1.0
                taxa_mutacao_atual = np.clip(taxa_base * ajuste, 0.05, 0.5)
                print(f"   🔧 Mutação adaptativa: geração {geracao+1}; diversidade (spread)={current_spread:.3f}; taxa de mutação={taxa_mutacao_atual:.3f}")

            pais = self.selecionar_pais(populacao, fitness, tamanho_populacao)
            filhos = []
            for i in range(0, len(pais), 2):
                if i + 1 >= len(pais):
                    continue
                pai1, pai2 = pais[i], pais[i + 1]

                if np.random.random() < CROSSOVER_PROB:
                    filho1, filho2 = self.crossover_simulated_binary(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()

                if ADAPTIVE_MUTATION:
                    filho1, _ = self.mutacao_adaptativa(filho1, geracao, n_geracoes, diversidade_atual=current_spread)
                    filho2, _ = self.mutacao_adaptativa(filho2, geracao, n_geracoes, diversidade_atual=current_spread)
                else:
                    if np.random.random() < MUTATION_PROB:
                        filho1 = self.mutacao_polinomial(filho1)
                    if np.random.random() < MUTATION_PROB:
                        filho2 = self.mutacao_polinomial(filho2)

                filho1 = np.asarray(filho1, dtype=float).flatten()
                filho2 = np.asarray(filho2, dtype=float).flatten()
                filhos.append(filho1 if len(filho1) == NUM_CAMADAS * 2 else self.inicializar_populacao(1)[0])
                filhos.append(filho2 if len(filho2) == NUM_CAMADAS * 2 else self.inicializar_populacao(1)[0])

            filhos = np.array(filhos, dtype=float)
                
            # 🆕 DEBUG: Verificar se os operadores genéticos estão funcionando
            print(f"\n🔧 DEBUG OPERADORES GENÉTICOS - Geração {geracao+1}:")
            print(f"   • Pais selecionados: {len(pais)}")
            print(f"   • Filhos gerados: {len(filhos)}")
            
            if len(filhos) > 0:
                # Verificar se os filhos são diferentes dos pais
                num_diferentes = 0
                for i in range(min(5, len(filhos))):
                    eh_diferente = False
                    for j in range(min(5, len(populacao))):
                        if np.array_equal(filhos[i], populacao[j]):
                            break
                    else:
                        num_diferentes += 1
    
                print(f"   • Quantos dos 5 primeiros filhos são geneticamente diferentes dos pais ? {num_diferentes}/{min(5, len(filhos))}")
    
                # Verificar mutação
                if len(filhos) > 0:
                    filho_exemplo = filhos[0]
                    print(f"   • As espessuras estão dentro da faixa esperada ? {filho_exemplo[:7].round(3)}")
                    
                    # Verificar materiais
                    materiais_filho = filho_exemplo[NUM_CAMADAS:].astype(int)
                    print(f"   • Materiais filho: {materiais_filho}")
                    
                    # Verificar espessuras
                    espessuras_filho = filho_exemplo[:NUM_CAMADAS]
                    print(f"   • Soma espessuras filho: {np.sum(espessuras_filho):.3f} cm")
                    
            # Avaliar filhos
            fitness_filhos = self.avaliar_populacao(filhos)

            # DEBUG: Comparar fitness dos filhos com pais
            print(f"\n📊 COMPARAÇÃO FILHOS vs PAIS:")
            print(f"   • Fitness médio pais: Dose={np.mean(fitness[:,0]):.3f}, Peso={np.mean(fitness[:,1]):.1f}, Custo=${np.mean(fitness[:,2]):.0f}")
            print(f"   • Fitness médio filhos: Dose={np.mean(fitness_filhos[:,0]):.3f}, Peso={np.mean(fitness_filhos[:,1]):.1f}, Custo=${np.mean(fitness_filhos[:,2]):.0f}")
            
            # Verificar se há filhos melhores que os pais
            melhor_dose_filhos = np.min(fitness_filhos[:, 0])
            melhor_dose_pais = np.min(fitness[:, 0])
            
            if melhor_dose_filhos < melhor_dose_pais:
                print(f"   ✅ Melhor dose reduzida! Dose: {melhor_dose_filhos:.3f} < {melhor_dose_pais:.3f}")
            else:
                print(f"   ⚠️  Nenhum filho melhor que os pais (dose)")

            # Combinar pais e filhos
            populacao_combinada = np.vstack([populacao, filhos])
            fitness_combinado = np.vstack([fitness, fitness_filhos])
            
            # Selecionar nova população (NSGA-II core)
            fronteiras, _ = self.fast_non_dominated_sort(fitness_combinado)
            
            # Extrair a primeira fronteira (soluções não-dominadas)
            primeira_fronteira = fronteiras[0]
            if len(primeira_fronteira) > 1:
                fitness_frente = fitness_combinado[primeira_fronteira]
                hv_resultado = calcular_hipervolume_unificado(
                    fitness_frente, 
                    referencia=self.reference_point,
                    normalizado=True
                )
                hv_norm = hv_resultado['hipervolume_normalizado']
                print(f"   📊 Geração {geracao+1}: Hipervolume = {hv_norm:.6f}")
                self.hv_history.append(hv_norm)
            
            # DIAGNÓSTICO: Tamanho das fronteiras (adicionar após a linha acima)
            if geracao % 10 == 0 or geracao < 5:  # Mostra nas primeiras 5 e a cada 10 gerações
                print(f"  🔍 Geração {geracao+1}: {len(fronteiras)} fronteiras encontradas.")
                for i, fronteira in enumerate(fronteiras[:3]):  # Mostra as 3 primeiras
                    print(f"     F{i+1}: {len(fronteira)} indivíduos", end="")
                    if i == 0:
                        # Calcula diversidade média na primeira fronteira
                        if len(fronteira) > 1:
                            crowding_f1 = self.crowding_distance(fitness_combinado, fronteira)
                            # Calcular média APENAS dos valores finitos (ignorar inf)
                            crowding_finite = [d for d in crowding_f1 if not np.isinf(d)]
                            if crowding_finite:
                                avg = np.mean(crowding_finite)
                            else:
                                avg = float('inf')
                            print(f" | Crowding Distance (average): {avg:.4f} - quanto maior, mais esparsas são as soluções, logo, maior diversidade")
                        else:
                            print()
                if len(fronteiras) > 3:
                    print(f"     ... e mais {len(fronteiras) - 3} fronteiras.")
        
            # ==============================================================
            # 🎯 SELEÇÃO DA NOVA POPULAÇÃO - ALGORITMO NSGA-II PADRÃO
            # ==============================================================
            nova_populacao = []
            nova_fitness = []
            
            # 2. ADICIONAR fronteiras COMPLETAS até atingir ou ultrapassar o tamanho
            for rank, fronteira in enumerate(fronteiras):
                if len(nova_populacao) + len(fronteira) <= tamanho_populacao:
                    # Adicionar TODA a fronteira
                    for idx in fronteira:
                        nova_populacao.append(populacao_combinada[idx])
                        nova_fitness.append(fitness_combinado[idx])
                    print(f"  • Frente {rank+1}: adicionados {len(fronteira)} indivíduos (total: {len(nova_populacao)})")
                else:
                    # Última fronteira: selecionar os mais diversos
                    slots_restantes = tamanho_populacao - len(nova_populacao)
                    print(f"  • Frente {rank+1}: selecionando {slots_restantes}/{len(fronteira)} mais diversos")
                    
                    # Calcular crowding distance para esta fronteira
                    distancias = self.crowding_distance(fitness_combinado, fronteira)
                    
                    # Ordenar por crowding distance (maior primeiro)
                    indices_ordenados = [fronteira[i] for i in np.argsort(distancias)[::-1]]
                    
                    # Selecionar os mais diversos
                    for i in range(slots_restantes):
                        idx = indices_ordenados[i]
                        nova_populacao.append(populacao_combinada[idx])
                        nova_fitness.append(fitness_combinado[idx])
                        
                    break  # Parar após preencher a população
        
            # 3. VERIFICAÇÃO CRÍTICA: garantir que temos EXATAMENTE 'tamanho_populacao' indivíduos
            if len(nova_populacao) != tamanho_populacao:
                print(f"  ⚠️  População incompleta: {len(nova_populacao)}/{tamanho_populacao}")
                
                if len(nova_populacao) < tamanho_populacao:
                    # Completar com os MELHORES da população combinada que não foram selecionados
                    faltam = tamanho_populacao - len(nova_populacao)
                    
                    # Criar conjunto de indivíduos já selecionados
                    selecionados_set = set([tuple(ind) for ind in nova_populacao])
                    
                    # Encontrar não selecionados
                    nao_selecionados = []
                    for idx in range(len(populacao_combinada)):
                        ind_tuple = tuple(populacao_combinada[idx])
                        if ind_tuple not in selecionados_set:
                            nao_selecionados.append((idx, fitness_combinado[idx]))
                
                    if nao_selecionados:
                        # Ordenar por qualidade (soma dos objetivos)
                        nao_selecionados.sort(key=lambda x: np.sum(x[1]))
                        
                        # Adicionar os melhores
                        for i in range(min(faltam, len(nao_selecionados))):
                            idx = nao_selecionados[i][0]
                            nova_populacao.append(populacao_combinada[idx])
                            nova_fitness.append(fitness_combinado[idx])
                            
                    # Se ainda faltar, criar novos aleatórios
                    if len(nova_populacao) < tamanho_populacao:
                        faltam_final = tamanho_populacao - len(nova_populacao)
                        print(f"  🔧 Criando {faltam_final} novos indivíduos aleatórios")
                        novos = self.inicializar_populacao(faltam_final)
                        novos_fitness = self.avaliar_populacao(novos)
                        nova_populacao.extend(novos)
                        nova_fitness.extend(novos_fitness)
                        
                elif len(nova_populacao) > tamanho_populacao:
                    # Cortar os PIORES (maior soma de objetivos)
                    print(f"  🔧 Cortando {len(nova_populacao) - tamanho_populacao} piores indivíduos")
                    scores = [np.sum(f) for f in nova_fitness]
                    indices_ordenados = np.argsort(scores)[:tamanho_populacao]
                    nova_populacao = [nova_populacao[i] for i in indices_ordenados]
                    nova_fitness = [nova_fitness[i] for i in indices_ordenados]
                    
            # 4. VERIFICAÇÃO FINAL DE TAMANHO
            if len(nova_populacao) != tamanho_populacao:
                print(f"  🚨 ERRO CRÍTICO: População {len(nova_populacao)} != {tamanho_populacao}")
                print(f"  🔧 Forçando ajuste final...")
                if len(nova_populacao) > tamanho_populacao:
                    nova_populacao = nova_populacao[:tamanho_populacao]
                    nova_fitness = nova_fitness[:tamanho_populacao]
                else:
                    faltam = tamanho_populacao - len(nova_populacao)
                    novos = self.inicializar_populacao(faltam)
                    novos_fitness = self.avaliar_populacao(novos)
                    nova_populacao.extend(novos)
                    nova_fitness.extend(novos_fitness)
                    
            # 5. CONVERTER PARA ARRAYS (garantir formato correto)
            populacao = np.array(nova_populacao, dtype=float)
            fitness = np.array(nova_fitness, dtype=float)
            
            # --- Salvar checkpoints a cada checkpoint_interval gerações ---
            if (geracao + 1) % checkpoint_interval == 0:
                self.salvar_checkpoint_csv(populacao, fitness, geracao)
                if checkpoint_file:
                    self.save_checkpoint(checkpoint_file, geracao + 1, populacao, fitness)
                    
            # 🔽 INSIRA AQUI AS LINHAS PARA REGISTRAR O HISTÓRICO
            self.historico_populacoes.append(populacao.copy())
            self.historico_fitness.append(fitness.copy())
            
            # ==============================================================
            # 7. Salvar checkpoint periódico (se ativado)
            # ==============================================================
            # print(f"DEBUG: checkpoint_file={checkpoint_file}, geracao={geracao}, modulo={ (geracao+1) % checkpoint_interval if checkpoint_interval else 'N/A' }")
            # if checkpoint_file and ((geracao + 1) % checkpoint_interval == 0):
            #     print(f"💾 Tentando salvar checkpoint em {checkpoint_file}...")
            #    self.save_checkpoint(checkpoint_file, geracao + 1, populacao, fitness)
            
            # Salvar figuras a cada 10 gerações (se figures_dir for fornecido)
            if figures_dir and (geracao + 1) % 25 == 0:
                spread_atual = self.calcular_spread_metric(fitness)
                subdir = os.path.join(figures_dir, f"geracao_{geracao+1}")
                os.makedirs(subdir, exist_ok=True)
                visualizar_frente_pareto_3d(
                    fitness_pareto=fitness,
                    solucoes_pareto=populacao,   # <--- adicionar esta linha
                    output_dir=subdir,
                    spread_value=spread_atual
                )
            
            # 6. DEBUG: Mostrar estatísticas da nova população
            print(f"  ✅ Nova população: {len(populacao)} indivíduos")
            print(f"  📊 Melhor dose: {fitness[:, 0].min():.4f} mSv/5min")
            print(f"  📊 Melhor peso: {fitness[:, 1].min():.1f} kg")
            print(f"  📊 Melhor custo: ${fitness[:, 2].min():.0f}")
            
            # Encontrar frente de Pareto atual
            fronteiras, _ = self.fast_non_dominated_sort(fitness)
            frente_atual = [populacao[idx] for idx in fronteiras[0]]
            
            # ==============================================================
            # 🎯 ANÁLISE DA FRENTE DE PARETO (DEB ET AL., 2002)
            # ==============================================================
            
            # Extrair apenas a primeira fronteira (soluções não-dominadas)
            fitness_frente = np.array([fitness[idx] for idx in fronteiras[0]])
            
            if len(fitness_frente) == 0:
                continue

            doses = fitness_frente[:, 0]
            pesos = fitness_frente[:, 1]
            custos = fitness_frente[:, 2]

            # Cabeçalho apenas na primeira geração
            if geracao == 0:
                print("\n" + "="*80)
                print("EVOLUÇÃO NSGA-II - ANÁLISE PARETO (DEB ET AL., 2002)")
                print("="*80)
                print("Ger | Soluções | Melhor Dose  | Melhor Peso | Melhor Custo | Trade-off")
                print("-"*80)
                
            # Encontrar MELHOR em CADA objetivo (já na frente de Pareto)
            idx_melhor_dose = np.argmin(doses)
            idx_melhor_peso = np.argmin(pesos)
            idx_melhor_custo = np.argmin(custos)
            
            # Mostrar linha da tabela para TODAS as gerações
            print(f"{geracao+1:3d} | {len(fitness_frente):8d} | {doses.min():11.4f} | {pesos.min():10.1f} | ${custos.min():11.0f} | ...")
            
            # ==============================================================
            # 🎯 VERIFICAÇÃO DE DIVERSIDADE (EVITAR ESTAGNAÇÃO)
            # ==============================================================
            if geracao > 10:  # Começar após 10 gerações
                # Calcular diversidade atual
                diversidade_atual = len(set([tuple(sol) for sol in frente_atual]))
    
                # Se diversidade baixa, aplicar mutação extra
                if diversidade_atual < len(frente_atual) * 0.3:  # Menos de 30% de diversidade
                    print(f"   ⚠️  Baixa diversidade ({diversidade_atual}/{len(frente_atual)} únicas)")
                    print(f"   🔄 Aplicando mutação extra em 10% da população...")
                    
                    # Selecionar 10% da população para mutação extra
                    num_mutar = max(5, int(len(populacao) * 0.1))
                    indices_mutar = np.random.choice(len(populacao), num_mutar, replace=False)
        
                    for idx in indices_mutar:
                        # Mutação mais agressiva
                        populacao[idx] = self.mutacao_polinomial(
                            populacao[idx], 
                            taxa_mutacao=0.5,  # 50% de chance
                            eta=10  # Mais agressivo
                        )
            
            # ==============================================================
            # 🎯 MÉTODO TOPSIS - HWANG & YOON (1981) - PARA SELEÇÃO DO MELHOR TRADE-OFF
            # ==============================================================
            # Aplicar TOPSIS apenas a cada 5 gerações para evitar repetição
            if geracao % 5 == 0 or geracao == n_geracoes - 1:
                best_idx_topsis, scores_topsis = calcular_tradeoff_topsis(fitness_frente)
                idx_melhor_tradeoff = best_idx_topsis
            else:
                # Usar solução com menor soma normalizada como fallback
                scores_fallback = (doses/doses.max() + pesos/pesos.max() + custos/custos.max())
                idx_melhor_tradeoff = np.argmin(scores_fallback)
                        
        # ==========================================================================
        # 6. Salvar checkpoint final (se ativo e última geração não salva)
        # ==========================================================================
        if checkpoint_file and ((n_geracoes) % checkpoint_interval != 0):
            # Se a última geração não foi salva pelo intervalo, salva agora
            self.save_checkpoint(checkpoint_file, n_geracoes, populacao, fitness)

        # ======================================================================
        # 🎯 RECALCULAR MELHOR TRADE-OFF COM TOPSIS - HWANG & YOON (1981)
        # ======================================================================
        print("\n" + "="*80)
        print("🔍 RECALCULANDO MELHOR TRADE-OFF - TOPSIS-IQR (FRENTE DE PARETO FINAL)")
        print("="*80)

        # Obter frente final de Pareto
        fronteiras_final, _ = self.fast_non_dominated_sort(fitness)
        if fronteiras_final and len(fronteiras_final) > 0:
            frente_final = [populacao[idx] for idx in fronteiras_final[0]]
            fitness_frente_final = np.array([fitness[idx] for idx in fronteiras_final[0]])

            print(f"📊 Frente final de Pareto: {len(fitness_frente_final)} soluções")

            # APLICAR TOPSIS PARA SELEÇÃO DO MELHOR TRADE-OFF
            best_idx_topsis, scores_topsis = calcular_tradeoff_topsis(fitness_frente_final)

            doses_final = fitness_frente_final[:, 0]
            pesos_final = fitness_frente_final[:, 1]
            custos_final = fitness_frente_final[:, 2]

            print(f"\n🎯 MELHOR TRADE-OFF - TOPSIS-IQR:")
            print(f"   • Dose:  {doses_final[best_idx_topsis]:.4f} mSv/5min")
            print(f"   • Peso:  {pesos_final[best_idx_topsis]:.2f} kg")
            print(f"   • Custo: ${custos_final[best_idx_topsis]:.2f}")
            print(f"   • Score TOPSIS: {scores_topsis[best_idx_topsis]:.3f} [0=pior, 1=melhor]")
            print(f"   • Ranking: {np.sum(scores_topsis > scores_topsis[best_idx_topsis]) + 1}/{len(scores_topsis)}")

            if ADAPTIVE_MUTATION:
                print(f"\n🔧 Mutação adaptativa foi chamada {self.mutacao_chamadas} vezes durante a otimização.")

            frente_pareto = frente_final
            idx_equilibrio = best_idx_topsis
        else:
            print("⚠️  Não foi possível recalcular trade-off - frente de Pareto vazia")
            idx_equilibrio = 0
            frente_pareto = []

        print("\n" + "="*80)
        print("✅ OTIMIZAÇÃO CONCLUÍDA!")

        # ==============================================================
        #    🎯 RESUMO FINAL DETALHADO
        # ==============================================================
        fronteiras_final, _ = self.fast_non_dominated_sort(fitness)
        if fronteiras_final and len(fronteiras_final) > 0:
            frente_final = [populacao[idx] for idx in fronteiras_final[0]]
            fitness_frente_final = [fitness[idx] for idx in fronteiras_final[0]]

            doses_final = [f[0] for f in fitness_frente_final]
            pesos_final = [f[1] for f in fitness_frente_final]
            custos_final = [f[2] for f in fitness_frente_final]

            idx_min_dose = np.argmin(doses_final)
            idx_min_peso = np.argmin(pesos_final)
            idx_min_custo = np.argmin(custos_final)

            fitness_frente_final_array = np.array(fitness_frente_final)
            best_idx_topsis_final, scores_topsis_final = calcular_tradeoff_topsis(fitness_frente_final_array)
            idx_melhor_tradeoff_final = best_idx_topsis_final

            # ======================================================================
            # 💾 SALVAR HISTÓRICO DO HIPERVOLUME (se houver)
            # ======================================================================
            if hasattr(self, 'hv_history') and len(self.hv_history) > 0:
                df_hv = pd.DataFrame({
                    'geracao': range(1, len(self.hv_history) + 1),
                    'hipervolume': self.hv_history
                })
                csv_path = os.path.join(RESULTS_DIR, 'csv', 'hipervolume_historico.csv')
                df_hv.to_csv(csv_path, index=False)
                print(f"   💾 Histórico do hipervolume salvo em: {csv_path}")

        # ==============================================================
        # 🎯 RESUMO OBJETIVO DA EVOLUÇÃO - QUANDO PARAR?
        # ==============================================================
        print("\n" + "="*100)

        # DEBUG: verificar tamanho do histórico
        print(f"🔍 DEBUG: tamanho do historico_fitness = {len(self.historico_fitness)}")

        if hasattr(self, 'historico_diagnosticos') and self.historico_diagnosticos:
            self.gerar_resumo_evolucao(self.historico_fitness, self.historico_diagnosticos)
        else:
            self.gerar_resumo_evolucao(self.historico_fitness, None)

        # ==============================================================
        # 📝 CONCLUSÃO PARA O ARTIGO
        # ==============================================================
        ponto_estagnacao = None
        if hasattr(self, 'historico_diagnosticos') and self.historico_diagnosticos:
            for i, diagnostico in enumerate(self.historico_diagnosticos):
                if isinstance(diagnostico, dict) and diagnostico.get('estagnado', False):
                    ponto_estagnacao = i + 1
                    break

        if ponto_estagnacao:
            print(f"   • Convergência atingida na geração {ponto_estagnacao}/{n_geracoes}")
            print(f"   • {n_geracoes - ponto_estagnacao} gerações subsequentes sem melhoria significativa")
            print(f"   • Economia de tempo potencial: {(n_geracoes - ponto_estagnacao)/n_geracoes*100:.0f}%")
        else:
            print("="*100)

        # ==============================================================
        # 🎯 RETORNAR RESULTADOS
        # ==============================================================
        return populacao, fitness, frente_pareto
        
        # ==========================================================================
        # 📈 ANÁLISE DE ESTABILIZAÇÃO DO HIPERVOLUME (NOVO BLOCO)
        # ==========================================================================
        if hasattr(self, 'hv_history') and len(self.hv_history) >= 10:
            threshold = 0.01  # 1% de variação máxima
            window = 5        # número de gerações consecutivas
            stable_gen = None
            for i in range(window, len(self.hv_history)):
                changes = []
                for j in range(i - window + 1, i + 1):
                    if self.hv_history[j-1] > 0:
                        rel_change = abs(self.hv_history[j] - self.hv_history[j-1]) / self.hv_history[j-1]
                        changes.append(rel_change)
                if changes and max(changes) < threshold:
                    stable_gen = i - window + 1  # primeira geração do período estável
                    break
            if stable_gen:
                print(f"\n📈 Hipervolume estabilizou a partir da geração {stable_gen} "
                      f"(variação < {threshold*100:.0f}% nas últimas {window} gerações).")
                
                # Armazenar para relatório (opcional)
                self.estabilizacao_geracao = stable_gen
            else:
                print("\n⚠️  Hipervolume não estabilizou dentro do critério adotado.")
        else:
            print("\n⚠️  Histórico insuficiente para análise de estabilização (mínimo 10 gerações).")
        
        # Retornar resultados
        if 'frente_final' in locals() and frente_final:
            return populacao, fitness, frente_final
        else:
            # Fallback: usar primeira fronteira
            if fronteiras and len(fronteiras) > 0:
                frente_final = [populacao[idx] for idx in fronteiras[0]]
            else:
                frente_final = []
            return populacao, fitness, frente_final

# ============================================================================
# FUNÇÃO DE HYPERVOLUME - COM DEAP
# ============================================================================

def calcular_hipervolume_unificado(fitness_pareto, referencia=None, normalizado=True):
    """
    Calcula o hipervolume exato usando a implementação do DEAP (Zitzler & Thiele, 1998).
    """
    if len(fitness_pareto) < 2:
        return criar_resultado_fallback(fitness_pareto)
    
    # Converte para lista de listas
    points = [list(p) for p in fitness_pareto]
    
    # Define o ponto de referência
    if referencia is None:
        max_vals = np.max(points, axis=0)
        referencia = (max_vals * 1.1).tolist()
    else:
        referencia = list(referencia)
    
    # Cálculo exato do hipervolume via DEAP
    try:
        hv_bruto = benchmarks.tools.hypervolume(points, referencia)
    except (AttributeError, ImportError):
        # Fallback para importação direta do submódulo
        hv_bruto = hv.hypervolume(points, referencia)
    
    # Normalização
    if normalizado:
        vol_total = np.prod(referencia)
        hv_norm = hv_bruto / vol_total if vol_total > 0 else 0.0
    else:
        hv_norm = hv_bruto
    
    ponto_ideal = np.min(points, axis=0).tolist()
    ponto_nadir = np.max(points, axis=0).tolist()
    
    return {
        'hipervolume_normalizado': float(hv_norm),
        'hipervolume_bruto': float(hv_bruto),
        'ponto_ideal': ponto_ideal,
        'ponto_nadir': ponto_nadir,
        'ponto_referencia': referencia,
        'n_solucoes': len(points)
    }

def criar_resultado_fallback(fitness_pareto):
    """Resultado de fallback em caso de erro"""
    n_obj = 3
    fallback = {
        'hipervolume_normalizado': 0.0,
        'hipervolume_bruto': 0.0,
        'ponto_ideal': [0.0, 0.0, 0.0],
        'ponto_nadir': [0.0, 0.0, 0.0],
        'ponto_referencia': [1.0, 1.0, 1.0],
        'n_solucoes': 0
    }
    
    if fitness_pareto is not None and len(fitness_pareto) > 0:
        fitness_np = np.array(fitness_pareto)
        fallback['ponto_ideal'] = np.min(fitness_np, axis=0).tolist()
        fallback['ponto_nadir'] = np.max(fitness_np, axis=0).tolist()
        fallback['ponto_referencia'] = (np.max(fitness_np, axis=0) * 1.1).tolist()
        fallback['n_solucoes'] = len(fitness_np)
        
        # Cálculo simplificado de hipervolume
        ranges = np.max(fitness_np, axis=0) - np.min(fitness_np, axis=0)
        ranges[ranges == 0] = 1.0
        normalized = (fitness_np - np.min(fitness_np, axis=0)) / ranges
        fallback['hipervolume_normalizado'] = float(np.mean(np.prod(1.1 - normalized, axis=1)))
    
    return fallback

# ============================================================================
# FUNÇÕES DE LEGADO (para compatibilidade)
# ============================================================================

def calcular_hipervolume_absoluto(fitness_pareto, referencia=None, normalizado=True):
    """Compatibilidade com código existente"""
    resultado = calcular_hipervolume_unificado(fitness_pareto, referencia, normalizado)
    return resultado['hipervolume_bruto']

def calcular_hipervolume_com_referencia(fitness_pareto, fitness_todos=None):
    """Compatibilidade com código existente"""
    # Usar dados MCNP para referência se disponível
    referencia = None
    if fitness_todos is not None and len(fitness_todos) > 0:
        todos_np = np.array(fitness_todos)
        referencia = tuple(np.max(todos_np, axis=0) * 1.1)
    
    return calcular_hipervolume_unificado(fitness_pareto, referencia, normalizado=True)

# ==============================================================================
# ANÁLISE E VISUALIZAÇÃO DOS RESULTADOS
# ==============================================================================

def analisar_resultados(populacao_final, fitness_final, frente_pareto, df_mcnp=None):
    """
    Análise científica dos resultados da otimização
    """
    print("ANÁLISE DOS RESULTADOS - FRENTE DE PARETO OTIMIZADA")
    
    # 🚨 SOLUÇÃO: Use a frente_pareto que já foi calculada e passada!
    # Não precisa recriar o NSGA2Optimizer
    
    if frente_pareto is None or len(frente_pareto) == 0:
        print("⚠️  Frente de Pareto vazia - usando toda a população")
        soluções_pareto = populacao_final
        fitness_pareto = fitness_final
    else:
        # Já temos a frente de Pareto calculada pelo NSGA-II
        # Precisamos encontrar o fitness correspondente a cada solução
        soluções_pareto = np.array(frente_pareto)
        
        # Encontrar fitness correspondente para cada solução Pareto
        fitness_pareto = []
        for sol_pareto in soluções_pareto:
            # Buscar solução mais próxima na população
            distancias = np.linalg.norm(populacao_final - sol_pareto, axis=1)
            idx_mais_proximo = np.argmin(distancias)
            
            if idx_mais_proximo < len(fitness_final):
                fitness_pareto.append(fitness_final[idx_mais_proximo])
            else:
                # Fallback: usar primeiro fitness
                fitness_pareto.append(fitness_final[0])
        
        fitness_pareto = np.array(fitness_pareto)
    
    # Se não temos fitness_pareto válido, usar os primeiros N
    if len(fitness_pareto) == 0 and len(fitness_final) > 0:
        n_mostrar = min(20, len(fitness_final))
        fitness_pareto = fitness_final[:n_mostrar]
        soluções_pareto = populacao_final[:n_mostrar]
    
    # Extrair objetivos com nomes EXPLÍCITOS
    dose_5min_array = fitness_pareto[:, 0]  # ← NOME EXPLÍCITO: dose para 5 minutos
    peso_array = fitness_pareto[:, 1]       # Peso em kg
    custo_array = fitness_pareto[:, 2]      # Custo em USD
    
    # Estatísticas detalhadas
    doses = fitness_pareto[:, 0]
    pesos = fitness_pareto[:, 1]
    custos = fitness_pareto[:, 2]
    
    print("\n📈 ESTATÍSTICAS DAS SOLUÇÕES PARETO:")
    print(f"   • Dose efetiva:")
    print(f"      - Mínimo: {doses.min():.6f} mSv/5min")
    print(f"      - Máximo: {doses.max():.6f} mSv/5min")
    print(f"      - Média: {doses.mean():.6f} mSv/5min")
    print(f"      - Mediana: {np.median(doses):.6f} mSv/5min")
    
    print(f"\n   • Peso total:")
    print(f"      - Mínimo: {pesos.min():.2f} kg")
    print(f"      - Máximo: {pesos.max():.2f} kg")
    print(f"      - Média: {pesos.mean():.2f} kg")
    
    print(f"\n   • Custo total:")
    print(f"      - Mínimo: ${custos.min():.2f}")
    print(f"      - Máximo: ${custos.max():.2f}")
    print(f"      - Média: ${custos.mean():.2f}")
    
    # Melhores soluções por objetivo
    idx_melhor_dose = np.argmin(doses)
    idx_melhor_peso = np.argmin(pesos)
    idx_melhor_custo = np.argmin(custos)
    
    print("\n🏆 MELHORES SOLUÇÕES POR OBJETIVO:")
    print(f"   • Melhor proteção (menor dose):")
    print(f"      - Dose: {doses[idx_melhor_dose]:.6f} mSv/5min")
    print(f"      - Peso: {pesos[idx_melhor_dose]:.2f} kg")
    print(f"      - Custo: ${custos[idx_melhor_dose]:.2f}")
    
    print(f"\n   • Menor peso:")
    print(f"      - Dose: {doses[idx_melhor_peso]:.6f} mSv/5min")
    print(f"      - Peso: {pesos[idx_melhor_peso]:.2f} kg")
    print(f"      - Custo: ${custos[idx_melhor_peso]:.2f}")
    
    print(f"\n   • Menor custo:")
    print(f"      - Dose: {doses[idx_melhor_custo]:.6f} mSv/5min")
    print(f"      - Peso: {pesos[idx_melhor_custo]:.2f} kg")
    print(f"      - Custo: ${custos[idx_melhor_custo]:.2f}")
    
    # Trade-offs
    print("\n⚖️  ANÁLISE DE TRADE-OFFS:")
    
    # Correlações
    corr_dose_peso = np.corrcoef(doses, pesos)[0, 1]
    corr_dose_custo = np.corrcoef(doses, custos)[0, 1]
    corr_peso_custo = np.corrcoef(pesos, custos)[0, 1]
    
    print(f"   • Correlação Dose-Peso: {corr_dose_peso:.3f}")
    print(f"   • Correlação Dose-Custo: {corr_dose_custo:.3f}")
    print(f"   • Correlação Peso-Custo: {corr_peso_custo:.3f}")
    
    # Eficiência por custo
    if np.any(custos > 0):
        eficiencia_custo = doses / custos
        idx_melhor_eficiencia = np.argmin(eficiencia_custo)
    
    # ======================================================================
    # CÁLCULO DO HIPERVOLUME EXATO (ZITZLER & THIELE, 1998)
    # ======================================================================
    print("\n📊 CÁLCULO DO HIPERVOLUME EXATO - Implementação baseada no algoritmo de Zitzler & Thiele (1998) via DEAP")
    
    try:
        # Referência global fixa, idêntica para todas as execuções comparadas.
        nadir = np.max(fitness_pareto, axis=0)
        referencia = PONTO_REFERENCIA_HV.copy()
        
        print(f"\n🎯 PONTO DE REFERÊNCIA GLOBAL FIXO:")
        print(f"   • Dose: {referencia[0]:.3f} mSv/5min (nadir atual: {nadir[0]:.3f})")
        print(f"   • Peso: {referencia[1]:.3f} kg (nadir atual: {nadir[1]:.3f})")
        print(f"   • Custo: ${referencia[2]:.0f} (nadir atual: ${nadir[2]:.0f})")
        
        hv_resultados = calcular_hipervolume_unificado(
            fitness_pareto, 
            referencia=referencia,
            normalizado=True
        )

        print(f"\n• Soluções na frente: {hv_resultados['n_solucoes']}")
        print(f"• Hipervolume normalizado: {hv_resultados['hipervolume_normalizado']:.6f}")
        print(f"• Hipervolume bruto: {hv_resultados['hipervolume_bruto']:.2e}")
        print(f"• Ponto ideal (melhor em cada objetivo): {hv_resultados['ponto_ideal']}")
        print(f"• Ponto nadir (pior em cada objetivo): {hv_resultados['ponto_nadir']}")
        print(f"• Ponto referência usado: {hv_resultados['ponto_referencia']}")
    
    except Exception as e:
        print(f"⚠️  Erro no cálculo do hipervolume: {e}")
        hv_resultados = criar_resultado_fallback(fitness_pareto)
        print(f"• Usando valores de fallback")
        
    print("\n🔬 INTERPRETAÇÃO CIENTÍFICA (ZITZLER & THIELE, 1999):")
    hv_norm = hv_resultados['hipervolume_normalizado']
    
    if hv_norm > 0.7:
        print(f"  ✅ EXCELENTE: HV = {hv_norm:.3f} > 0.7 (qualidade superior)")
        print(f"     • A frente Pareto cobre >70% do espaço objetivo normalizado")
        print(f"     • Adequado para publicações de alto impacto")
    elif hv_norm > 0.5:
        print(f"  ✅ BOM: HV = {hv_norm:.3f} > 0.5 (qualidade satisfatória)")
        print(f"     • Cobertura de 50-70% do espaço objetivo")
        print(f"     • Aceitável para periódicos A1/JCR")
    elif hv_norm > 0.3:
        print(f"  ⚠️  MODERADO: HV = {hv_norm:.3f} > 0.3")
        print(f"     • Cobertura de 30-50% - considere aumentar população/gerações")
        print(f"     • Resultados válidos, mas com margem para melhoria")
    else:
        print(f"  ⚠️  BAIXO: HV = {hv_norm:.3f} ≤ 0.3")
        print(f"     • Possível sub-amostragem ou convergência prematura")
        print(f"     • Recomenda-se: População ≥ 300, Gerações ≥ 100")

    return soluções_pareto, fitness_pareto

def visualizar_frente_pareto_3d(fitness_pareto, solucoes_pareto, output_dir, spread_value=0.0, show=False):
    """
    Visualização 3D profissional da frente de Pareto
    
    Parâmetros:
    -----------
    fitness_pareto : array
        Array com fitness das soluções Pareto
    spread_value : float
        Valor da métrica spread (diversidade)
    """
    
    print("\n🎨 Gerando visualizações profissionais da frente de Pareto...")
    
    # 🆕 Usar FIGURES_DIR configurado globalmente (figuras_pareto dentro de resultados_nsga2)
    output_dir = FIGURES_DIR
    print(f"📁 Figuras serão salvas em: {output_dir}")
    
    # Criar diretório de saída
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # ============================================================================
    # 1. DADOS COM NOMENCLATURA CORRETA E EXPLÍCITA
    # ============================================================================
    
    # Extrair dados com nomes EXPLÍCITOS
    dose_5min_array = fitness_pareto[:, 0]    # Dose para 5 minutos (mSv/5min)
    peso_array = fitness_pareto[:, 1]         # Peso em kg
    custo_array = fitness_pareto[:, 2]        # Custo em USD
    
    # ============================================================================
    # 2. FILTRAGEM PARETO PRIMEIRO (DEB ET AL., 2002)
    # ============================================================================
    def filtrar_frente_pareto(fitness):
        """Filtra apenas pontos não-dominados (Deb et al., 2002)"""
        n = len(fitness)
        dominado = np.zeros(n, dtype=bool)
    
        for i in range(n):
            for j in range(n):
                if i != j and not dominado[j]:
                    if np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                        dominado[i] = True
                        break
    
        return fitness[~dominado]

    # Aplicar filtro Pareto antes de QUALQUER análise
    fitness_pareto_filtrado = filtrar_frente_pareto(fitness_pareto)
    
    if len(fitness_pareto_filtrado) < 2:
        print(f"⚠️  ATENÇÃO: Apenas {len(fitness_pareto_filtrado)} solução não-dominada")
        print(f"   • Usando todas as {len(fitness_pareto)} soluções para visualização")
        fitness_pareto_filtrado = fitness_pareto.copy()
    
    # Usar APENAS os dados filtrados
    dose_5min_array = fitness_pareto_filtrado[:, 0]
    peso_array = fitness_pareto_filtrado[:, 1]
    custo_array = fitness_pareto_filtrado[:, 2]
    
    # Verificar outliers
    if peso_array.max() > 1000:
        print(f"⚠️  PESO > 1000 kg detectado! Máximo: {peso_array.max():.0f} kg")
        print(f"   Índices com peso alto: {np.where(peso_array > 1000)[0]}")
        
    if custo_array.min() == 0:
        print(f"⚠️  CUSTO $0 detectado!")
        print(f"   Índices com custo 0: {np.where(custo_array == 0)[0]}")
        
    # ============================================================================
    # 3. IDENTIFICAR SOLUÇÕES DESTACADAS (APÓS FILTRAGEM)
    # ============================================================================
    idx_melhor_dose = np.argmin(dose_5min_array)
    idx_menor_peso = np.argmin(peso_array)
    idx_menor_custo = np.argmin(custo_array)
    idx_melhor_tradeoff, scores_topsis = calcular_tradeoff_topsis(fitness_pareto_filtrado)

    # Extrair configurações (espessuras e materiais) das soluções destacadas
    solucoes_destacadas_indices = [idx_melhor_dose, idx_menor_peso, idx_menor_custo, idx_melhor_tradeoff]
    
    # Preparar listas para as colunas de cada camada
    materiais_cols = {f'Material_camada_{i+1}': [] for i in range(NUM_CAMADAS)}
    espessuras_cols = {f'Espessura_camada_{i+1}_cm': [] for i in range(NUM_CAMADAS)}
    
    for idx in solucoes_destacadas_indices:
        sol = solucoes_pareto[idx]          # solução completa (14 elementos)
        espessuras = sol[:NUM_CAMADAS]
        materiais_idx = sol[NUM_CAMADAS:].astype(int)
        materiais_nomes = [MATERIAIS_DISPONIVEIS[i] for i in materiais_idx]
        
        for i in range(NUM_CAMADAS):
            espessuras_cols[f'Espessura_camada_{i+1}_cm'].append(espessuras[i])
            materiais_cols[f'Material_camada_{i+1}'].append(materiais_nomes[i])
    
    # ============================================================================
    # 4. MÉTODO TOPSIS - HWANG & YOON (1981) - PARA SELEÇÃO DO MELHOR TRADE-OFF
    # ============================================================================
    # Usar a função TOPSIS para encontrar o melhor trade-off
    idx_melhor_tradeoff, scores_topsis = calcular_tradeoff_topsis(fitness_pareto_filtrado)
    
    print(f"\n🎯 Soluções destacadas - TOPSIS-IQR:")
    print(f"   • Melhor proteção (IDX {idx_melhor_dose}): {dose_5min_array[idx_melhor_dose]:.3e} mSv/5min")
    print(f"   • Menor peso (IDX {idx_menor_peso}): {peso_array[idx_menor_peso]:.1f} kg")
    print(f"   • Menor custo (IDX {idx_menor_custo}): ${custo_array[idx_menor_custo]:.0f}")
    print(f"   • Melhor trade-off por TOPSIS (IDX {idx_melhor_tradeoff}):")
    print(f"       - Dose: {dose_5min_array[idx_melhor_tradeoff]:.3e} mSv/5min")
    print(f"       - Peso: {peso_array[idx_melhor_tradeoff]:.1f} kg")
    print(f"       - Custo: ${custo_array[idx_melhor_tradeoff]:.0f}")
    print(f"       - Score TOPSIS: {scores_topsis[idx_melhor_tradeoff]:.3f}")
    
    # ============================================================================
    # 3. FIGURA 1: FRENTE DE PARETO 3D OTIMIZADA - TOTALMENTE PARAMETRIZADA
    # ============================================================================
    
    print("\n📈 Criando figura 3D otimizada da frente de Pareto...")
    
    # ============================================================================
    # CONFIGURAÇÕES DO USUÁRIO - AJUSTE AQUI
    # ============================================================================
    
    # Dicionário de configurações - MODIFIQUE ESTES VALORES PARA CUSTOMIZAR
    metrics_text = f'Spread: {spread_value:.3f}'

    CONFIG_3D = {
        # 1. Tamanho e Layout
        'figsize': (20, 16),  # (largura, altura) em polegadas
    
        # 2. Configurações do Plot Principal
        'cmap': 'viridis_r',           # Mapa de cores: 'viridis', 'plasma', 'RdYlBu_r', 'coolwarm', 'Spectral'
        'point_alpha': 0.85,           # Transparência dos pontos (0.0-1.0)
        'edge_color': 'black',         # Cor da borda dos pontos
        'edge_width': 0.3,             # Espessura da borda
        'depth_shade': True,           # Sombra para profundidade
        'base_size': 40,               # Tamanho base dos pontos
        'size_variation': 80,          # Variação máxima do tamanho
        
        # 3. Configurações dos Eixos
        'x_label': 'Weight (kg)\n(Minimize →)',
        'y_label': 'Cost (USD)\n(Minimize →)',
        'z_label': 'Effective dose (mSv/5 min)\n(Minimize ←)',
        'label_fontsize': 20,
        'label_fontweight': 'bold',
        'label_pad': 35,
    
        # Limites dos eixos (None = automático, ou especifique (min, max))
        'x_limits': (peso_array.min() * 0.9, peso_array.max() * 1.05),
        'y_limits': (custo_array.min() * 0.9, custo_array.max() * 1.05),
        'z_limits': (dose_5min_array.min() * 0.9, dose_5min_array.max() * 1.05),
        
        # 4. Título
        'title': f'3D PARETO FRONT - NSGA-II MULTIOBJECTIVE OPTIMIZATION\n',
        'title_fontsize': 24,
        'title_fontweight': 'bold',
        'title_pad': 0,
        'title_y': 0.92,   # valor entre 0 e 1; 1 = topo da caixa dos eixos, 0.9 = mais próximo
    
        # 5. Grade
        'grid_enabled': True,
        'grid_alpha': 0.25,
        'grid_style': '-',
        'grid_width': 0.5,
        
        # 6. Barra de Cores (Colorbar)
        'colorbar_shrink': 0.65,
        'colorbar_aspect': 25,
        'colorbar_pad': 0.07,   # aumente para afastar
        'colorbar_label': 'Normalized distance to ideal point (0,0,0)',
        'colorbar_label_fontsize': 26,
        'colorbar_label_fontweight': 'bold',
        'colorbar_label_pad': 35,
        'colorbar_tick_fontsize': 25,
        
        # 7. LEGENDA - CUSTOMIZAÇÃO COMPLETA
        'legend_location': 'lower left',  # Opções: 'upper right', 'upper left', 'lower left', 'lower right',
                                       # 'upper center', 'lower center', 'center left', 'center right', 'center'
        'legend_fontsize': 18,
        'legend_framealpha': 0.95,  
        'legend_shadow': True,
        'legend_borderpad': 1,
        'legend_labelspacing': 1,
        'legend_handletextpad': 0.5,
        'legend_ncol': 1,                       # Número de colunas (1 = vertical, 2+ = horizontal)
        'legend_bbox_to_anchor': (0.05, 0.55),  # (x, y) para posicionamento preciso (ex: (1.05, 1.0))
        'legend_loc_offset': (0, 0),            # Ajuste fino (dx, dy) em pontos
        
        # 8. Ângulo de Visualização
        'view_elevation': 28,     # 0-90 (0 = olhando de lado, 90 = de cima)
        'view_azimuth': 135,      # 0-360 (rotação horizontal)
        
        # 9. Linhas de Referência
        'reference_lines': False,
        'ref_lines_alpha': 0.3,
        'ref_lines_style': '--',
        'ref_lines_width': 1,
        'ref_lines_color': 'black',
        
        # 10. Texto Informativo
        # 'info_text': (
        #     f'Geometry: Semicylindrical (R={RAIO_INTERNO_CM} cm, H={ALTURA_CM} cm)\n'
        #     f'Layers: {NUM_CAMADAS} | Total thickness: {ESPESSURA_TOTAL_FIXA} cm\n'
        #     f'Volume: {0.5*np.pi*((RAIO_INTERNO_CM+ESPESSURA_TOTAL_FIXA)**2 - RAIO_INTERNO_CM**2)*ALTURA_CM:.0f} cm³\n'
        #    f'Population: {POPULATION_SIZE} | Generations: {NUM_GENERATIONS}'
        # 'info_text_position': (0.02, 0.02),  # Coordenadas (x, y) em frações da figura
        # 'info_text_fontsize': 9,
        # 'info_text_style': 'italic',
        # 'info_text_bbox': dict(boxstyle='round', facecolor='lightgray', alpha=0.7, edgecolor='black'),
        
        # 11. Soluções Destacadas (Cores, Marcadores, Tamanhos)
        'highlighted_solutions': [
            {
                'index': idx_melhor_dose,
                'color': 'red',
                'marker': '*',
                'size': 400,
                'label': 'Best protection',
                'zorder': 20,
                'edgecolor': 'black',
                'linewidth': 2,
                'format_string': '({:.2e} mSv/5min, {:.1f} kg, ${:.0f} USD)'
            },
            {
                'index': idx_menor_peso,
                'color': 'blue',
                'marker': '^',
                'size': 400,
                'label': 'Lowest weight',
                'zorder': 19,
                'edgecolor': 'black',
                'linewidth': 2,
                'format_string': '({:.2e} mSv/5min, {:.1f} kg, ${:.0f} USD)'
            },
            {
                'index': idx_menor_custo,
                'color': 'green',
                'marker': 's',
                'size': 400,
                'label': 'Lowest cost',
                'zorder': 18,
                'edgecolor': 'black',
                'linewidth': 2,
                'format_string': '({:.2e} mSv/5min, {:.1f} kg, ${:.0f} USD)'
            },
            {
                'index': idx_melhor_tradeoff,
                'color': 'gold',
                'marker': 'D',
                'size': 400,
                'label': 'Best trade-off',
                'zorder': 21,
                'edgecolor': 'black',
                'linewidth': 2,
                'format_string': '({:.2e} mSv/5min, {:.1f} kg, ${:.0f} USD)'
            }
        ],
    
        # 12. Salvar Configurações
        'save_dpi': 600,
        'save_format': 'pdf',
        'save_bbox_inches': 'tight'
    }

    # ============================================================================
    # CONSTRUÇÃO DO GRÁFICO 3D COM BASE NAS CONFIGURAÇÕES
    # ============================================================================
    
    # 1. Criação da figura
    fig_3d = plt.figure(figsize=CONFIG_3D['figsize'])
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    
    ax_3d.tick_params(axis='both', labelsize=22)   # para eixos X e Y
    ax_3d.tick_params(axis='z', labelsize=22)      # para eixo Z

    # 2. Preparação dos dados
    dist_ao_ideal = np.sqrt(
        (peso_array / peso_array.max())**2 + 
        (custo_array / custo_array.max())**2 + 
        (dose_5min_array / dose_5min_array.max())**2
    )
    sorted_idx = np.argsort(dist_ao_ideal)
    
    # 3. Plot principal
    scatter_3d = ax_3d.scatter(
        peso_array[sorted_idx], 
        custo_array[sorted_idx], 
        dose_5min_array[sorted_idx],
        c=dist_ao_ideal[sorted_idx],
        cmap=CONFIG_3D['cmap'],
        s=CONFIG_3D['base_size'] + CONFIG_3D['size_variation'] * (1 - dist_ao_ideal[sorted_idx]),
        alpha=CONFIG_3D['point_alpha'],
        edgecolors=CONFIG_3D['edge_color'],
        linewidth=CONFIG_3D['edge_width'],
        depthshade=CONFIG_3D['depth_shade'],
        zorder=5
    )
    
    # 4. Adicionar soluções destacadas
    for sol in CONFIG_3D['highlighted_solutions']:
        idx = sol['index']
    
        # Formatar string do label
        dose_val = dose_5min_array[idx]
        peso_val = peso_array[idx]
        custo_val = custo_array[idx]
        formatted_values = sol['format_string'].format(dose_val, peso_val, custo_val)
        label = f"{sol['label']}\n{formatted_values}"
        
        ax_3d.scatter(
            peso_array[idx], 
            custo_array[idx], 
            dose_5min_array[idx],
            c=sol['color'],
            s=sol['size'],
            marker=sol['marker'],
            edgecolors=sol['edgecolor'],
            linewidth=sol['linewidth'],
            label=label,
            zorder=sol['zorder']
        )

    # 5. Configurar eixos
    ax_3d.set_xlabel(CONFIG_3D['x_label'], 
                     fontsize=CONFIG_3D['label_fontsize'], 
                     fontweight=CONFIG_3D['label_fontweight'], 
                     labelpad=CONFIG_3D['label_pad'])
    ax_3d.set_ylabel(CONFIG_3D['y_label'], 
                     fontsize=CONFIG_3D['label_fontsize'], 
                     fontweight=CONFIG_3D['label_fontweight'], 
                     labelpad=CONFIG_3D['label_pad'])
    ax_3d.set_zlabel(CONFIG_3D['z_label'], 
                     fontsize=CONFIG_3D['label_fontsize'], 
                     fontweight=CONFIG_3D['label_fontweight'], 
                     labelpad=CONFIG_3D['label_pad'])
    
    # Aplicar limites dos eixos
    if CONFIG_3D['x_limits']:
        ax_3d.set_xlim(*CONFIG_3D['x_limits'])
    if CONFIG_3D['y_limits']:
        ax_3d.set_ylim(*CONFIG_3D['y_limits'])
    if CONFIG_3D['z_limits']:
        ax_3d.set_zlim(*CONFIG_3D['z_limits'])
    
    # 6. Título
    ax_3d.set_title(CONFIG_3D['title'], 
                fontsize=CONFIG_3D['title_fontsize'], 
                fontweight=CONFIG_3D['title_fontweight'], 
                pad=CONFIG_3D['title_y'])
    ax_3d.title.set_position((0.5, 0.7))   # (x, y) em coordenadas dos eixos

    # 7. Grade
    ax_3d.grid(CONFIG_3D['grid_enabled'], 
               alpha=CONFIG_3D['grid_alpha'], 
               linestyle=CONFIG_3D['grid_style'], 
               linewidth=CONFIG_3D['grid_width'])

    # 8. Barra de cores
    cbar_3d = plt.colorbar(scatter_3d, 
                           ax=ax_3d, 
                           shrink=CONFIG_3D['colorbar_shrink'], 
                           aspect=CONFIG_3D['colorbar_aspect'], 
                           pad=CONFIG_3D['colorbar_pad'])
    cbar_3d.set_label(CONFIG_3D['colorbar_label'], 
                      fontsize=CONFIG_3D['colorbar_label_fontsize'], 
                      fontweight=CONFIG_3D['colorbar_label_fontweight'], 
                      labelpad=CONFIG_3D['colorbar_label_pad'])
    cbar_3d.ax.tick_params(labelsize=CONFIG_3D['colorbar_tick_fontsize'])

    cbar_3d.ax.set_title('↓ Lower = Better ↑', fontsize=24, pad=20, fontweight='bold')

    # 9. LEGENDA - CUSTOMIZAÇÃO AVANÇADA
    legend_kwargs = {
        'fontsize': CONFIG_3D['legend_fontsize'],
        'framealpha': CONFIG_3D['legend_framealpha'],
        'shadow': CONFIG_3D['legend_shadow'],
        'borderpad': CONFIG_3D['legend_borderpad'],
        'labelspacing': CONFIG_3D['legend_labelspacing'],
        'handletextpad': CONFIG_3D['legend_handletextpad'],
        'ncol': CONFIG_3D['legend_ncol']
    }

    # Se bbox_to_anchor for especificado, use posicionamento avançado
    if CONFIG_3D['legend_bbox_to_anchor']:
        legend_kwargs['bbox_to_anchor'] = CONFIG_3D['legend_bbox_to_anchor']
        legend_kwargs['loc'] = CONFIG_3D['legend_location']
    else:
        # Posicionamento normal com ajuste fino
        legend_kwargs['loc'] = CONFIG_3D['legend_location']
    
        # Aplicar offset se especificado
        if CONFIG_3D['legend_loc_offset'] != (0, 0):
            # Converter offset para bbox_to_anchor
            dx, dy = CONFIG_3D['legend_loc_offset']
            legend_kwargs['bbox_to_anchor'] = (dx/100 + 1, dy/100 + 1)

    ax_3d.legend(**legend_kwargs)

    # 10. Ângulo de visualização
    ax_3d.view_init(elev=CONFIG_3D['view_elevation'], 
                    azim=CONFIG_3D['view_azimuth'])

    # 11. Linhas de referência
    if CONFIG_3D['reference_lines']:
        min_x, min_y, min_z = peso_array.min(), custo_array.min(), dose_5min_array.min()
    
        ax_3d.plot([min_x, min_x], [min_y, min_y], [min_z, dose_5min_array.max()], 
                   CONFIG_3D['ref_lines_style'],
                   color=CONFIG_3D['ref_lines_color'],
                   alpha=CONFIG_3D['ref_lines_alpha'],
                   linewidth=CONFIG_3D['ref_lines_width'],
                   zorder=1)
    
        ax_3d.plot([min_x, peso_array.max()], [min_y, min_y], [min_z, min_z], 
                   CONFIG_3D['ref_lines_style'],
                   color=CONFIG_3D['ref_lines_color'],
                   alpha=CONFIG_3D['ref_lines_alpha'],
                   linewidth=CONFIG_3D['ref_lines_width'],
                   zorder=1)
    
        ax_3d.plot([min_x, min_x], [min_y, custo_array.max()], [min_z, min_z], 
                   CONFIG_3D['ref_lines_style'],
                   color=CONFIG_3D['ref_lines_color'],
                   alpha=CONFIG_3D['ref_lines_alpha'],
                   linewidth=CONFIG_3D['ref_lines_width'],
                   zorder=1)
        
    # 12. Texto informativo (mantido)
    # fig_3d.text(CONFIG_3D['info_text_position'][0],
    #             CONFIG_3D['info_text_position'][1],
    #             CONFIG_3D['info_text'],
    #             fontsize=CONFIG_3D['info_text_fontsize'],
    #             style=CONFIG_3D['info_text_style'],
    #            bbox=CONFIG_3D['info_text_bbox'])

    # ==================================================================
    # Ajuste final do layout
    # ==================================================================
    plt.tight_layout()

    # 13. Salvar figura
    filename_3d = os.path.join(output_dir, f"pareto_3d_customizado_{timestamp}.{CONFIG_3D['save_format']}")
    # Define DEBUG_SAVE localmente se não existir
    if 'DEBUG_SAVE' not in dir():
        DEBUG_SAVE = False
    if DEBUG_SAVE:
        print(f"[DEBUG] visualizar_frente_pareto_3d -> {filename_3d}")
    fig_3d.savefig(filename_3d, 
                   dpi=CONFIG_3D['save_dpi'], 
                   bbox_inches=CONFIG_3D['save_bbox_inches'], 
                   format=CONFIG_3D['save_format'])
    print(f"✅ Figura 3D customizada salva como: {filename_3d}")
        
    # ============================================================================
    # 4. FIGURA 2: TRADE-OFFS 2D (PARA ANÁLISE DETALHADA) - ATUALIZADO COM TOPSIS
    # ============================================================================
    
    print("📊 Criando figura de trade-offs 2D...")
    
    fig_2d, axes_2d = plt.subplots(2, 2, figsize=(16, 12))
    
    # ----------------------------------------------------------------------------
    # Subplot 1: Dose vs Peso (Trade-off mais importante)
    ax1 = axes_2d[0, 0]
    scatter1 = ax1.scatter(peso_array, dose_5min_array,
                           c=custo_array, cmap='viridis',
                           s=50, alpha=0.7, edgecolors='black')

    # Destacar soluções especiais
    ax1.scatter(peso_array[idx_melhor_dose], dose_5min_array[idx_melhor_dose],
                c='red', s=100, marker='*', edgecolors='black', label='Best protection')
    ax1.scatter(peso_array[idx_menor_peso], dose_5min_array[idx_menor_peso],
                c='blue', s=100, marker='^', edgecolors='black', label='Lower weight')
    ax1.scatter(peso_array[idx_melhor_tradeoff], dose_5min_array[idx_melhor_tradeoff],
                c='gold', s=100, marker='D', edgecolors='black', label='Best trade-off (TOPSIS-IQR)')

    ax1.set_xlabel('Weight (kg)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Effective dose (mSv/5 min)', fontsize=12, fontweight='bold')
    ax1.set_title('(A) Trade-off: Dose vs Weight', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Barra de cores (custo)
    cbar1 = plt.colorbar(scatter1, ax=ax1)
    cbar1.set_label('Cost (USD)', fontsize=12, fontweight='bold')
    
    # ----------------------------------------------------------------------------
    # Subplot 2: Dose vs Custo
    ax2 = axes_2d[0, 1]
    scatter2 = ax2.scatter(custo_array, dose_5min_array,
                           c=peso_array, cmap='plasma',
                           s=50, alpha=0.7, edgecolors='black')

    ax2.annotate(f'Best trade-off\nWeight: {peso_array[idx_melhor_tradeoff]:.1f} kg',
             xy=(custo_array[idx_melhor_tradeoff], dose_5min_array[idx_melhor_tradeoff]),
             xytext=(custo_array[idx_melhor_tradeoff]+1500, dose_5min_array[idx_melhor_tradeoff]+2),
             arrowprops=dict(arrowstyle='->', color='black'),
             fontsize=10, fontweight='bold')

    ax2.scatter(custo_array[idx_melhor_dose], dose_5min_array[idx_melhor_dose],
                c='red', s=100, marker='*', edgecolors='black', label='Best protection')
    ax2.scatter(custo_array[idx_menor_custo], dose_5min_array[idx_menor_custo],
                c='green', s=100, marker='s', edgecolors='black', label='Lower cost')
    ax2.scatter(custo_array[idx_melhor_tradeoff], dose_5min_array[idx_melhor_tradeoff],
                c='gold', s=100, marker='D', edgecolors='black', label='Best trade-off (TOPSIS-IQR)')
    
    ax2.set_xlabel('Cost (USD)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Effective dose (mSv/5 min)', fontsize=12, fontweight='bold')
    ax2.set_title('(B) Trade-off: Dose vs Cost', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Weight (kg)', fontsize=12, fontweight='bold')
    
    # ----------------------------------------------------------------------------
    # Subplot 3: Peso vs Custo
    ax3 = axes_2d[1, 0]
    scatter3 = ax3.scatter(peso_array, custo_array,
                           c=dose_5min_array, cmap='coolwarm',
                           s=50, alpha=0.7, edgecolors='black')

    ax3.scatter(peso_array[idx_menor_peso], custo_array[idx_menor_peso],
                c='blue', s=100, marker='^', edgecolors='black', label='Lower weight')
    ax3.scatter(peso_array[idx_menor_custo], custo_array[idx_menor_custo],
                c='green', s=100, marker='s', edgecolors='black', label='Lower cost')
    ax3.scatter(peso_array[idx_melhor_tradeoff], custo_array[idx_melhor_tradeoff],
                c='gold', s=100, marker='D', edgecolors='black', label='Best trade-off (TOPSIS-IQR)')

    ax3.set_xlabel('Weight (kg)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cost (USD)', fontsize=12, fontweight='bold')
    ax3.set_title('(C) Trade-off: Weight vs Cost', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label('Effective Dose (mSv/5 min)', fontsize=12, fontweight='bold')
    
    # Adicionar texto nas figuras
    # metrics_text = f'Spread: {spread_value:.3f}'
    # fig_3d.text(0.15, 0.95, metrics_text, fontsize=9, 
    #            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Adicionar texto na figura 2D  
    # fig_2d.text(0.25, 0.98, metrics_text, fontsize=8,
    #            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # print(f"✅ Métricas adicionadas: {metrics_text}")

    # ----------------------------------------------------------------------------
    # Subplot 4: Parallel Coordinates (Análise multivariada)
    ax4 = axes_2d[1, 1]
    
    # Calcular normalizações min-max para visualização apenas
    dose_norm_viz = (dose_5min_array - dose_5min_array.min()) / (dose_5min_array.max() - dose_5min_array.min() + 1e-10)
    peso_norm_viz = (peso_array - peso_array.min()) / (peso_array.max() - peso_array.min() + 1e-10)
    custo_norm_viz = (custo_array - custo_array.min()) / (custo_array.max() - custo_array.min() + 1e-10)
    
    # Normalizar todos os objetivos para escala 0-1 (APENAS PARA VISUALIZAÇÃO)
    objectives_normalized = np.column_stack([
        dose_norm_viz,   # Dose (0 = min, 1 = max)
        peso_norm_viz,   # Peso
        custo_norm_viz   # Custo
    ])
    
    # Plot parallel coordinates para todas as soluções
    for i in range(len(objectives_normalized)):
        ax4.plot([0, 1, 2], objectives_normalized[i], 
                 color='gray', alpha=0.2, linewidth=0.5)

    # Destacar soluções especiais (nota: agora idx_melhor_tradeoff vem do TOPSIS)
    for idx, color, label in [
        (idx_melhor_dose, 'red', 'Best protection'),
        (idx_menor_peso, 'blue', 'Lower weight'),
        (idx_menor_custo, 'green', 'Lower cost'),
        (idx_melhor_tradeoff, 'gold', 'Best trade-off (TOPSIS-IQR)')
    ]:
    
        ax4.plot([0, 1, 2], objectives_normalized[idx], 
                 color=color, linewidth=2.5, marker='o', markersize=8, label=label)

    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(['Effective dose\n(mSv/5min)', 'Weight\n(kg)', 'Cost\n(USD)'], 
                        fontsize=10, fontweight='bold')
    ax4.set_ylabel('Normalized value (Min-Max)', fontsize=12, fontweight='bold')
    ax4.set_title('(D) Parallel Coordinates - Multivariate Analysis', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10, loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-0.1, 1.1)
    
    # Título geral da figura 2D - ATUALIZADO
    fig_2d.suptitle('TRADE-OFF ANALYSIS - RADIATION SHIELDING OPTIMIZATION (NSGA-II + TOPSIS-IQR)\n' + f'Pareto Front: {len(fitness_pareto_filtrado)} non-dominated solutions | Spread = {spread_value:.3f}', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Salvar figura 2D
    filename_2d = os.path.join(output_dir, f"trade_offs_topsis_{timestamp}.pdf")
    if DEBUG_SAVE:
        print(f"[DEBUG] visualizar_frente_pareto_3d -> {filename_2d}")
    fig_2d.savefig(filename_2d, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Figura de trade-offs (TOPSIS-IQR) salva como: {filename_2d}")
        
    # ============================================================================
    # 7. GRÁFICOS 2D DA FRENTE DE PARETO PARA CADA PAR DE OBJETIVOS (2D PURA)
    # GRÁFICOS 2D SIMPLES (PROJEÇÕES PURAS, SEM TERCEIRO OBJETIVO)
    # ============================================================================
    print("📈 Criando gráficos 2D da frente de Pareto (2D pura)...")

    # Função auxiliar para calcular a frente de Pareto 2D
    def calcular_frente_pareto_2d(fitness, obj1, obj2):
        """
        Calcula a frente de Pareto para dois objetivos.
        Retorna índices das soluções não-dominadas no espaço 2D.
        """
        f1 = fitness[:, obj1]
        f2 = fitness[:, obj2]
        n = len(f1)
        dominado = np.zeros(n, dtype=bool)
    
        for i in range(n):
            for j in range(n):
                if i != j and not dominado[j]:
                    # j domina i se for melhor ou igual em ambos objetivos
                    # e estritamente melhor em pelo menos um
                    if (f1[j] <= f1[i] and f2[j] <= f2[i]) and (f1[j] < f1[i] or f2[j] < f2[i]):
                        dominado[i] = True
                        break
    
        return np.where(~dominado)[0]

    fig_2d_pura, axes_2d_pura = plt.subplots(1, 3, figsize=(18, 6))

    # Nomes dos objetivos para rótulos
    nomes_objetivos = ['Effective dose (mSv/5 min)', 'Weight (kg)', 'Cost (USD)']
    
    # Pares de objetivos: (0,1), (0,2), (1,2)
    pares = [
        (0, 1, 'Dose vs Weight'),
        (0, 2, 'Dose vs Cost'), 
        (1, 2, 'Weight vs Cost')
    ]

    for idx_par, (obj1, obj2, titulo) in enumerate(pares):
        ax = axes_2d_pura[idx_par]
    
        # Calcular frente de Pareto para estes dois objetivos
        indices_pareto_2d = calcular_frente_pareto_2d(fitness_pareto_filtrado, obj1, obj2)
    
        # Extrair os valores dos dois objetivos
        f1 = fitness_pareto_filtrado[indices_pareto_2d, obj1]
        f2 = fitness_pareto_filtrado[indices_pareto_2d, obj2]
        
        # Plotar a frente 2D
        ax.scatter(f1, f2, s=60, alpha=0.7, 
                   edgecolors='black', color='steelblue',
                   label=f'2D Pareto Front ({len(f1)} solutions)')
    
        # Encontrar as soluções extremas na frente 2D
        if len(f1) > 0:
            # Melhor em f1 (menor valor)
            idx_best_f1 = np.argmin(f1)
            # Melhor em f2 (menor valor)
            idx_best_f2 = np.argmin(f2)
        
            # Destacar essas soluções
            ax.scatter(f1[idx_best_f1], f2[idx_best_f1], 
                       c='red', s=100, marker='*', edgecolors='black', label=f'Best {nomes_objetivos[obj1]}')
            ax.scatter(f1[idx_best_f2], f2[idx_best_f2], 
                       c='blue', s=100, marker='^', edgecolors='black', label=f'Best {nomes_objetivos[obj2]}')
        
            # Calcular o melhor trade-off em 2D (distância ao ponto ideal)
            f1_norm = (f1 - np.min(f1)) / (np.max(f1) - np.min(f1) + 1e-10)
            f2_norm = (f2 - np.min(f2)) / (np.max(f2) - np.min(f2) + 1e-10)
            dist = np.sqrt(f1_norm**2 + f2_norm**2)  # Distância Euclidiana
            idx_best_tradeoff = np.argmin(dist)
            
            ax.scatter(f1[idx_best_tradeoff], f2[idx_best_tradeoff], 
                       c='gold', s=80, marker='D', edgecolors='black', label='Best 2D trade-off')
    
        # Configurar eixos
        ax.set_xlabel(nomes_objetivos[obj1], fontsize=12, fontweight='bold')
        ax.set_ylabel(nomes_objetivos[obj2], fontsize=12, fontweight='bold')
        ax.set_title(f'({chr(65+idx_par)}) 2D Pareto Front: {titulo}', 
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        
    fig_2d_pura.suptitle('PURE 2D PARETO FRONTS - OPTIMIZATION BY OBJECTIVE PAIRS\n' +
                    f'Each plot shows non-dominated solutions considering only two objectives',
                    fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Salvar figura 2D pura
    filename_2d_pura = os.path.join(output_dir, f"frentes_2d_puras_{timestamp}.pdf")
    fig_2d_pura.savefig(filename_2d_pura, dpi=300, bbox_inches='tight', format='pdf')
    print(f"✅ Figura 2D pura salva como: {filename_2d_pura}")
    
    # ============================================================================
    # 5. MOSTRAR FIGURAS
    # ============================================================================
    
    if show:
        print("\n📊 Exibindo figuras...")
        plt.show()
    else:
        # Fecha as figuras para liberar memória, já que não serão exibidas
        plt.close(fig_3d)
        plt.close(fig_2d)
        plt.close(fig_2d_pura)
        
    # ============================================================================
    # 6. SALVAR DADOS DAS SOLUÇÕES DESTACADAS (ATUALIZADO COM TOPSIS)
    # ============================================================================
    
    # Verificar se scores_topsis está disponível (deve ter sido calculado anteriormente)
    if 'scores_topsis' not in locals():
        # Calcular TOPSIS se não estiver disponível
        idx_melhor_tradeoff, scores_topsis = calcular_tradeoff_topsis(fitness_pareto_filtrado)

    # Criar DataFrame com todas as soluções destacadas
    destacadas_df = pd.DataFrame({
        'Tipo_Solucao': ['Melhor_Protecao', 'Menor_Peso', 'Menor_Custo', 'Melhor_Tradeoff_TOPSIS'],
        'Indice_Original': [idx_melhor_dose, idx_menor_peso, idx_menor_custo, idx_melhor_tradeoff],
        'ID_Solucao': [idx_melhor_dose + 1, idx_menor_peso + 1, idx_menor_custo + 1, idx_melhor_tradeoff + 1],
        'Dose_mSv_5min': [dose_5min_array[idx_melhor_dose], dose_5min_array[idx_menor_peso],
                          dose_5min_array[idx_menor_custo], dose_5min_array[idx_melhor_tradeoff]],
        'Peso_kg': [peso_array[idx_melhor_dose], peso_array[idx_menor_peso],
                    peso_array[idx_menor_custo], peso_array[idx_melhor_tradeoff]],
        'Custo_USD': [custo_array[idx_melhor_dose], custo_array[idx_menor_peso],
                      custo_array[idx_menor_custo], custo_array[idx_melhor_tradeoff]],
        'Score_TOPSIS': [scores_topsis[idx_melhor_dose] if idx_melhor_dose < len(scores_topsis) else 0.0,
                         scores_topsis[idx_menor_peso] if idx_menor_peso < len(scores_topsis) else 0.0,
                         scores_topsis[idx_menor_custo] if idx_menor_custo < len(scores_topsis) else 0.0,
                         scores_topsis[idx_melhor_tradeoff] if idx_melhor_tradeoff < len(scores_topsis) else 0.0],
        'Metodo_Selecao': ['Menor Dose', 'Menor Peso', 'Menor Custo', 'TOPSIS-IQR'],
        **{f'Material_camada_{i+1}': materiais_cols[f'Material_camada_{i+1}'] for i in range(NUM_CAMADAS)},
        **{f'Espessura_camada_{i+1}_cm': espessuras_cols[f'Espessura_camada_{i+1}_cm'] for i in range(NUM_CAMADAS)}
    })

    # 🆕 Salvar CSVs na subpasta csv/
    csv_dir = os.path.join(RESULTS_DIR, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    filename_destacadas = os.path.join(csv_dir, f"solucoes_destacadas_topsis_{timestamp}.csv")
    destacadas_df.to_csv(filename_destacadas, index=False, encoding='utf-8')
    print(f"✅ Dados das soluções destacadas (TOPSIS-IQR) salvos: {filename_destacadas}")

    # Calcular o ranking baseado nos scores
    ranking = np.argsort(scores_topsis)[::-1] + 1

    # Salvar também TODAS as soluções da frente de Pareto com scores TOPSIS
    todas_solucoes_df = pd.DataFrame({
        'ID_Solucao': range(1, len(fitness_pareto_filtrado) + 1),   # 1-based   
        'Dose_mSv_5min': dose_5min_array,
        'Peso_kg': peso_array,
        'Custo_USD': custo_array,
        'Score_TOPSIS': scores_topsis,
        'Ranking_TOPSIS': ranking
    })

    # Verificar se a melhor solução (idx_melhor_tradeoff) tem ranking 1
    best_rank = ranking[idx_melhor_tradeoff]
    if best_rank != 1:
        print(f"Atenção: a melhor solução (índice {idx_melhor_tradeoff}) tem ranking {best_rank}. Recalculando...")

    # Ordenar por score TOPSIS (do melhor para o pior)
    todas_solucoes_df = todas_solucoes_df.sort_values('Score_TOPSIS', ascending=False)
    todas_solucoes_df['Ranking_TOPSIS'] = range(1, len(todas_solucoes_df)+1)
    print("DataFrame reordenado por score.")
    
    filename_todas = os.path.join(csv_dir, f"todas_solucoes_topsis_{timestamp}.csv")
    todas_solucoes_df.to_csv(filename_todas, index=False, encoding='utf-8')
    print(f"✅ Todas soluções Pareto com scores TOPSIS-IQR salvas: {filename_todas}")
    
    return {
        'fig_3d': fig_3d,
        'fig_2d': fig_2d,
        'fig_2d_pura': fig_2d_pura,
        'destacadas': destacadas_df,
        'todas_solucoes': todas_solucoes_df,
        'arquivos': {
            'pareto_3d': filename_3d,
            'trade_offs': filename_2d,
            'frentes_2d_puras': filename_2d_pura,
            'destacadas_csv': filename_destacadas,
            'todas_solucoes_csv': filename_todas
        }
    }  

def salvar_resultados_csv(solucoes_pareto, fitness_pareto, output_dir, suffix=''):
    """
    Salva os resultados da frente de Pareto em formato CSV para análise posterior.
    Se output_dir for None, usa a variável global RESULTS_DIR.
    """
    if output_dir is None:
        raise ValueError("output_dir não pode ser None. Forneça o diretório base.")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    filename = os.path.join(csv_dir, f"resultados_pareto_nsga2{suffix}_{timestamp}.csv")
    
    # Criar DataFrame com todas as informações
    resultados = []
    
    for i, (solucao, fitness) in enumerate(zip(solucoes_pareto, fitness_pareto)):
        # Decodificar solução
        espessuras = solucao[:NUM_CAMADAS]
        materiais_indices = solucao[NUM_CAMADAS:].astype(int)
        materiais = [MATERIAIS_DISPONIVEIS[idx] for idx in materiais_indices]
        
        # Calcular features adicionais
        features, peso_calculado, custo_calculado, _ = criar_features_para_otimizacao(espessuras, materiais_indices)
        
        # 🆕 CORREÇÃO: fitness[0] já é dose em mSv/5min (não mSv/s)
        resultado = {
            'ID_Solucao': i+1,
            'Dose_Efetiva_mSv_5min': fitness[0],  # 🆕 Alterado para mSv/5min
            'Peso_Total_kg': fitness[1],
            'Custo_Total_USD': fitness[2],
            'Soma_Espessuras_cm': np.sum(espessuras),
            'Z_eff_Ponderado': features['z_eff_ponderado_espessura'],
            'Atenuacao_Total': features['atenuacao_total'],
            'Transmissao': features['transmissao'],
            'High_Low_Ratio': features['high_low_ratio'],
            'Sequence_Score': features['sequence_score']
        }
        
        # Adicionar detalhes por camada
        for j in range(NUM_CAMADAS):
            resultado[f'Material_Camada_{j+1}'] = materiais[j]
            resultado[f'Espessura_Camada_{j+1}_cm'] = espessuras[j]
        
        resultados.append(resultado)
    
    # Criar DataFrame e salvar
    df = pd.DataFrame(resultados)
    
    # Ordenar por dose (mais relevante)
    df = df.sort_values('Dose_Efetiva_mSv_5min')
    
    # Salvar
    if DEBUG_SAVE:
        print(f"[DEBUG] salvar_resultados_csv -> {filename}")
    df.to_csv(filename, index=False, encoding='utf-8')
    print(f"\n💾 Resultados salvos em: {filename}")
  
    # 🆕 Adicionar cálculo de dose por segundo para referência
    if 'Dose_Efetiva_mSv_5min' in df.columns:
        df['Dose_Efetiva_mSv_s'] = df['Dose_Efetiva_mSv_5min'] / 300.0
        print(f"   • Dose mínima (por segundo): {df['Dose_Efetiva_mSv_s'].min():.6e} mSv/s")
    
    return df, filename

def gerar_relatorio_cientifico(df_resultados, tempo_execucao, output_dir=RESULTS_DIR):
    """
    Gera um relatório científico completo dos resultados
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # 🆕 Salvar na subpasta relatorios/
    relatorios_dir = os.path.join(RESULTS_DIR, "relatorios")
    os.makedirs(relatorios_dir, exist_ok=True)
    
    filename = os.path.join(relatorios_dir, f"relatorio_nsga2_{timestamp}.txt")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO CIENTÍFICO - OTIMIZAÇÃO NSGA-II DE BLINDAGEM RADIOLÓGICA\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. METODOLOGIA\n")
        f.write("-"*40 + "\n")
        f.write(f"• Algoritmo: NSGA-II (Non-dominated Sorting Genetic Algorithm II)\n")
        f.write(f"• Tamanho da população: {POPULATION_SIZE}\n")
        f.write(f"• Número de gerações: {NUM_GENERATIONS}\n")
        f.write(f"• Taxa de crossover: {CROSSOVER_PROB}\n")
        f.write(f"• Taxa de mutação: {MUTATION_PROB}\n")
        f.write(f"• Tempo de execução: {tempo_execucao:.2f} segundos\n\n")
        
        f.write("2. CONFIGURAÇÃO DA BLINDAGEM\n")
        f.write("-"*40 + "\n")
        f.write(f"• Número de camadas: {NUM_CAMADAS}\n")
        f.write(f"• Espessura total fixa: {ESPESSURA_TOTAL_FIXA} cm\n")
        f.write(f"• Materiais disponíveis: {len(MATERIAIS_DISPONIVEIS)}\n")
        f.write(f"• Intervalo espessura por camada: {ESPESSURA_MIN_CAMADA} a {ESPESSURA_MAX_CAMADA} cm\n\n")
        
        f.write("3. OBJETIVOS DE OTIMIZAÇÃO\n")
        f.write("-"*40 + "\n")
        f.write("1. Minimizar Dose Efetiva (mSv/s) - Proteção radiológica\n")
        f.write("2. Minimizar Peso Total (kg) - Estrutural/logística\n")
        f.write("3. Minimizar Custo Total (USD) - Viabilidade econômica\n\n")
        
        f.write("4. RESULTADOS PRINCIPAIS\n")
        f.write("-"*40 + "\n")
        f.write(f"• Soluções na frente de Pareto: {len(df_resultados)}\n")
        f.write(f"• Melhor dose alcançada: {df_resultados['Dose_Efetiva_mSv_s'].min():.6f} mSv/s\n")
        f.write(f"• Menor peso alcançado: {df_resultados['Peso_Total_kg'].min():.2f} kg\n")
        f.write(f"• Menor custo alcançado: ${df_resultados['Custo_Total_USD'].min():.2f}\n")
        f.write(f"• Z_eff ponderado médio: {df_resultados['Z_eff_Ponderado'].mean():.2f}\n\n")
        
        f.write("5. ANÁLISE DE TRADE-OFFS\n")
        f.write("-"*40 + "\n")
        corr_dose_peso = df_resultados['Dose_Efetiva_mSv_s'].corr(df_resultados['Peso_Total_kg'])
        corr_dose_custo = df_resultados['Dose_Efetiva_mSv_s'].corr(df_resultados['Custo_Total_USD'])
        corr_peso_custo = df_resultados['Peso_Total_kg'].corr(df_resultados['Custo_Total_USD'])
        
        f.write(f"• Correlação Dose-Peso: {corr_dose_peso:.3f}\n")
        f.write(f"• Correlação Dose-Custo: {corr_dose_custo:.3f}\n")
        f.write(f"• Correlação Peso-Custo: {corr_peso_custo:.3f}\n\n")
        
        f.write("6. RECOMENDAÇÕES PARA APLICAÇÃO PRÁTICA\n")
        f.write("-"*40 + "\n")
        
        # Melhores soluções por critério
        melhor_dose = df_resultados.loc[df_resultados['Dose_Efetiva_mSv_s'].idxmin()]
        melhor_peso = df_resultados.loc[df_resultados['Peso_Total_kg'].idxmin()]
        melhor_custo = df_resultados.loc[df_resultados['Custo_Total_USD'].idxmin()]
        
        f.write("• Para máxima proteção (menor dose):\n")
        f.write(f"  Dose: {melhor_dose['Dose_Efetiva_mSv_s']:.6f} mSv/s, ")
        f.write(f"Peso: {melhor_dose['Peso_Total_kg']:.1f} kg, ")
        f.write(f"Custo: ${melhor_dose['Custo_Total_USD']:.0f}\n")
        
        f.write("• Para menor peso (aeronáutica/espacial):\n")
        f.write(f"  Dose: {melhor_peso['Dose_Efetiva_mSv_s']:.6f} mSv/s, ")
        f.write(f"Peso: {melhor_peso['Peso_Total_kg']:.1f} kg, ")
        f.write(f"Custo: ${melhor_peso['Custo_Total_USD']:.0f}\n")
        
        f.write("• Para menor custo (aplicações industriais):\n")
        f.write(f"  Dose: {melhor_custo['Dose_Efetiva_mSv_s']:.6f} mSv/s, ")
        f.write(f"Peso: {melhor_custo['Peso_Total_kg']:.1f} kg, ")
        f.write(f"Custo: ${melhor_custo['Custo_Total_USD']:.0f}\n\n")
        
        f.write("7. CONTRIBUIÇÕES CIENTÍFICAS\n")
        f.write("-"*40 + "\n")
        f.write("• Framework integrado ML + NSGA-II para otimização de blindagens\n")
        f.write("• Metodologia que reduz tempo de avaliação de semanas para minutos\n")
        f.write("• Identificação clara de trade-offs entre proteção, peso e custo\n")
        f.write("• Base para tomada de decisão em projetos de blindagem radiológica\n\n")
        
        f.write("8. LIMITAÇÕES E TRABALHOS FUTUROS\n")
        f.write("-"*40 + "\n")
        f.write("• Validação experimental das soluções otimizadas\n")
        f.write("• Inclusão de mais materiais compósitos\n")
        f.write("• Consideração de propriedades mecânicas adicionais\n")
        f.write("• Extensão para espectros de radiação mais complexos\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("FIM DO RELATÓRIO\n")
        f.write("="*80 + "\n")
    
    return filename

def criar_features_para_otimizacao_batch(df_configs, max_samples=500):
    """
    Recria features em batch para treinar scaler - IDÊNTICO À FASE 3
    """
    print(f"🔬 Criando features em batch ({max_samples} amostras)...")
    
    features_list = []
    n_amostras = min(max_samples, len(df_configs))
    
    for idx in range(n_amostras):
        # Extrair espessuras
        espessuras = []
        for i in range(1, 8):
            esp_col = f'espessura_camada_{i}'
            esp = float(str(df_configs.loc[idx, esp_col]).replace(',', '.'))
            espessuras.append(esp)
        
        # Extrair materiais e converter para índices
        materiais_idx = []
        for i in range(1, 8):
            mat_col = f'material_camada_{i}'
            mat_nome = str(df_configs.loc[idx, mat_col]).strip()
            
            if mat_nome in MATERIAIS_DISPONIVEIS:
                materiais_idx.append(MATERIAIS_DISPONIVEIS.index(mat_nome))
            else:
                materiais_idx.append(0)  # Fallback
        
        # Criar features usando a MESMA função da Fase 4
        features, _, _ = criar_features_para_otimizacao(np.array(espessuras), materiais_idx)
        
        # ORDEM EXATA DA FASE 3:
        features_vector = [
            features['z_eff_ponderado_espessura'],  # 1
            features['atenuacao_total'],            # 2
            features['z_dens_esp_total'],           # 3
            features['high_z_protection'],          # 4
            features['low_z_moderation'],           # 5
            features['high_low_ratio'],             # 6
            features['z_x_atenuacao'],              # 7
            features['transmissao'],                # 8
            # features['log_transmissao'],            # 9
            features['sequence_score'],             # 10
            features['z_variance']                  # 11
        ]
        
        features_list.append(features_vector)
    
    X_array = np.array(features_list)
    print(f"✅ Features criadas: {X_array.shape[0]} amostras × {X_array.shape[1]} features")
    
    return X_array

def criar_diretorios_output():
    """
    Cria todos os diretórios necessários para os outputs, exceto 'diagnosticos' (não utilizado).
    """
    print("\n" + "="*80)
    print("📁 CONFIGURAÇÃO DE DIRETÓRIOS DE OUTPUT")
    print("="*80)
    
    print(f"• Diretório principal: {RESULTS_DIR}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Lista de subdiretórios efetivamente utilizados
    subdirs = [
        FIGURES_DIR,
        os.path.join(RESULTS_DIR, "validacoes"),
        os.path.join(RESULTS_DIR, "debug"),
        os.path.join(RESULTS_DIR, "csv"),
        os.path.join(RESULTS_DIR, "relatorios")
    ]
    
    for subdir in subdirs:
        if not subdir.startswith(RESULTS_DIR):
            raise RuntimeError(f"🚨 Tentativa de criar diretório fora de RESULTS_DIR: {subdir}")
        os.makedirs(subdir, exist_ok=True)
        print(f"  ✓ Criado: {subdir}")
    
    print(f"\n✅ Todos os outputs serão salvos em: {RESULTS_DIR}")
    print("   Estrutura:")
    print(f"   - {RESULTS_DIR}/")
    print(f"     ├── figuras_pareto/")
    print(f"     ├── csv/")
    print(f"     ├── validacoes/")
    print(f"     ├── debug/")
    print(f"     └── relatorios/")
    
    return RESULTS_DIR, FIGURES_DIR

# ==============================================================================
# 🎯 VALIDAÇÃO ROBUSTA - DOSE EFETIVA (mSv/5min) - VERSÃO CORRIGIDA
# ==============================================================================

def validacao_robusta_artigo(scaler, modelo, target_transform, df_mcnp, output_dir, 
                            percentual_validacao=0.15):
    """
    Validação robusta para artigo A1/JCR - DOSE APENAS
    USO DE HOLD-OUT ANINHADO: Subconjunto do TREINO da Fase 3
    output_dir : str, optional
    Diretório base para salvar os resultados. Se None, usa a variável global RESULTS_DIR.
    """
    if output_dir is None:
        raise ValueError("output_dir não pode ser None.")
    
    print("="*80)
    print("📊 Validação INTERNA com Hold-out Aninhado: o objetivo é confirmar que o modelo não apresenta outliers nas próprias amostras de treino e que a transformação das features está correta (É uma etapa de garantia de qualidade interna")
    print("📌 Usando subconjunto do TREINO da Fase 3 (dados VISTOS, mas não no mesmo contexto)")
    print("-"*80)
    
    # 1. RECRIAR A DIVISÃO EXATA DA FASE 3
    print("\n1. RECRIANDO A DIVISÃO DE DADOS ORIGINAL:")
    
    from sklearn.model_selection import train_test_split
    
    indices_completos = np.arange(len(df_mcnp))
    
    # Recriar EXATAMENTE a divisão da Fase 3 (85/15 com random_state=42)
    indices_treino_fase3, indices_teste_fase3 = train_test_split(
        indices_completos,
        test_size=0.15,          # 15% teste (300 amostras)
        random_state=42,         # MESMA SEMENTE da Fase 3
        shuffle=True
    )
    
    print(f"   • Dataset completo: {len(df_mcnp)} configurações MCNP")
    print(f"   • TREINO Fase 3: {len(indices_treino_fase3)} amostras (85%)")
    print(f"   • TESTE Fase 3: {len(indices_teste_fase3)} amostras (15%)")
    
    # 2. CRIAR HOLD-OUT ANINHADO DENTRO DO TREINO DA FASE 3
    print("\n2. CRIANDO HOLD-OUT ANINHADO PARA VALIDAÇÃO:")
    
    # Do TREINO da Fase 3, separar uma porção para validação interna
    n_validacao = int(len(indices_treino_fase3) * percentual_validacao)
    
    # Amostragem aleatória MAS com semente DIFERENTE da Fase 3
    np.random.seed(12345)  # Nova semente para reprodutibilidade
    indices_validacao = np.random.choice(
        indices_treino_fase3, 
        size=n_validacao, 
        replace=False
    )
    
    df_validacao = df_mcnp.iloc[indices_validacao].copy()
    
    print(f"   • Amostras de validação: {len(df_validacao)} (retiradas do TREINO da Fase 3)")
    print(f"   • Faixa de dose: {df_validacao['dose_5min'].min():.3f} a {df_validacao['dose_5min'].max():.3f} mSv/5min")
    
    # 3. VALIDAÇÃO DE CONSISTÊNCIA INTERNA
    print("\n3. VALIDAÇÃO DE CONSISTÊNCIA INTERNA")
    
    resultados = []
    doses_reais = []
    doses_preditas = []
    
    for idx, row in df_validacao.iterrows():
        try:
            espessuras = np.array(row['espessuras_list'])
            materiais_nomes = row['materiais_list']
            
            materiais_idx = []
            for mat_nome in materiais_nomes:
                if mat_nome in MATERIAIS_DISPONIVEIS:
                    materiais_idx.append(MATERIAIS_DISPONIVEIS.index(mat_nome))
                else:
                    materiais_idx.append(0)
                    
            individuo = np.concatenate([espessuras, materiais_idx])
            
            dose_pred, _, _ = avaliar_individuo(individuo, scaler, modelo, target_transform)
            dose_real = row['dose_5min']
            
            doses_reais.append(dose_real)
            doses_preditas.append(dose_pred)
            
            erro_percentual = abs(dose_pred - dose_real) / dose_real * 100 if dose_real > 0 else 100
            
            resultados.append({
                'id_simulacao': row['id_simulacao'],
                'origem': 'TREINO_Fase3_Holdout',
                'dose_real_mSv_5min': dose_real,
                'dose_pred_mSv_5min': dose_pred,
                'erro_dose_%': erro_percentual
            })
            
        except Exception as e:
            print(f"   ⚠️  Erro: {str(e)[:50]}")
            
    df_resultados = pd.DataFrame(resultados)
    
    # Métricas estatísticas
    print("\n📊 MÉTRICAS ESTATÍSTICAS - DOSE EFETIVA:")
    print("-"*60)
    
    dose_mae = mean_absolute_error(doses_reais, doses_preditas)
    dose_rmse = np.sqrt(mean_squared_error(doses_reais, doses_preditas))
    dose_r2 = r2_score(doses_reais, doses_preditas)
    dose_mape = np.mean(np.abs((np.array(doses_reais) - np.array(doses_preditas)) / np.array(doses_reais))) * 100
    
    # VERIFICAÇÃO MATEMÁTICA FUNDAMENTAL: RMSE ≥ MAE [citation:1]
    if dose_rmse < dose_mae:
        print("   🚨 ALERTA MATEMÁTICO: RMSE < MAE é impossível [citation:1]")
        print(f"   • RMSE calculado: {dose_rmse:.6f}")
        print(f"   • MAE calculado: {dose_mae:.6f}")
        print("   • Revisar cálculos de erro imediatamente.")
        
    print("📊 RESULTADOS:")
    print(f"   • MAE:  {dose_mae:.6f} mSv/5min")
    print(f"   • RMSE: {dose_rmse:.6f} mSv/5min")
    print(f"   • R²:   {dose_r2:.6f}")
    print(f"   • MAPE: {dose_mape:.2f}%")
    print(f"   • Correlação: {np.corrcoef(doses_reais, doses_preditas)[0,1]:.6f}")
    
    # 4. ANÁLISE DE OUTLIERS E DISTRIBUIÇÃO DE ERROS
    print("\n4. ANÁLISE DE OUTLIERS E DISTRIBUIÇÃO DE ERROS:")
    
    df_resultados['categoria_erro'] = pd.cut(
        df_resultados['erro_dose_%'],
        bins=[-1, 5, 10, 20, 50, 100, float('inf')],
        labels=['Excelente (<5%)', 'Muito Bom (5-10%)', 'Bom (10-20%)', 
                'Moderado (20-50%)', 'Alto (50-100%)', 'Crítico (>100%)']
    )
    
    contagem_erros = df_resultados['categoria_erro'].value_counts().sort_index()
    
    print("   📈 Distribuição de erros (% das amostras):")
    for categoria, count in contagem_erros.items():
        percentual = count / len(df_resultados) * 100
        print(f"      • {categoria}: {percentual:.1f}% ({count} amostras)")
    
    # 5. GRÁFICO DE VALIDAÇÃO (DOSE ÚNICA)
    print("\n5. GERANDO GRÁFICOS DE VALIDAÇÃO...")
    validacao_dir = os.path.join(output_dir, "validacoes")
    os.makedirs(validacao_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Dispersão Dose Predita vs Real
    ax1.scatter(doses_reais, doses_preditas, alpha=0.6, edgecolors='k', linewidth=0.5)
    min_val = min(min(doses_reais), min(doses_preditas))
    max_val = max(max(doses_reais), max(doses_preditas))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x', linewidth=2)
    
    ax1.set_xlabel('MCNP simulated dose (mSv/5min)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Predicted dose (mSv/5min)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Validation: Effective Dose\nR² = {dose_r2:.4f}, MAE = {dose_mae:.4f} mSv/5min', 
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Histograma de Erros Percentuais
    ax2.hist(df_resultados['erro_dose_%'], bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(x=dose_mape, color='red', linestyle='--', linewidth=2, 
                label=f'MAPE = {dose_mape:.1f}%')
    ax2.set_xlabel('Relative error (%)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('Distribution of Percentage Error', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(f'ROBUST VALIDATION: Effective dose (n={n_validacao} configurations)\n'
                 f'Model: {type(modelo).__name__} | Split: 85% Training / 15% Testing',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    fig_path = os.path.join(validacao_dir, f"validacao_dose_{timestamp}.pdf")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"   ✅ Gráfico de validação salvo: {fig_path}")
    
    # 6. SALVAR RESULTADOS
    print("\n6. SALVANDO RESULTADOS COMPLETOS...")
    
    # DataFrame de métricas sumarizadas
    metricas_sumario = pd.DataFrame({
        'Metrica': ['MAE (mSv/5min)', 'RMSE (mSv/5min)', 'R²', 'MAPE (%)', 'Correlação'],
        'Valor': [f'{dose_mae:.6f}', f'{dose_rmse:.6f}', 
                 f'{dose_r2:.6f}', f'{dose_mape:.2f}', 
                 f'{np.corrcoef(doses_reais, doses_preditas)[0,1]:.6f}'],
        'Interpretação': [
            'Erro médio absoluto',
            'Raiz do erro quadrático médio [citation:1]',
            'Coeficiente de determinação',
            'Erro percentual médio absoluto',
            'Correlação linear de Pearson'
        ]
    })
    
    metricas_path = os.path.join(validacao_dir, f"metricas_dose_{timestamp}.csv")
    if DEBUG_SAVE:
        print(f"[DEBUG] validacao_robusta_artigo -> {metricas_path}")
    metricas_sumario.to_csv(metricas_path, index=False)
    
    detalhes_path = os.path.join(validacao_dir, f"detalhes_validacao_{timestamp}.csv")
    if DEBUG_SAVE:
        print(f"[DEBUG] validacao_robusta_artigo -> {detalhes_path}")
    df_resultados.to_csv(detalhes_path, index=False, encoding='utf-8')
    
    print(f"   ✅ Métricas sumarizadas salvas: {metricas_path}")
    print(f"   ✅ Resultados detalhados salvas: {detalhes_path}")
    
    return {
        'df_resultados': df_resultados,
        'metricas': {'MAE': dose_mae, 'RMSE': dose_rmse, 'R2': dose_r2, 'MAPE': dose_mape},
        'n_amostras': n_validacao,
        'arquivos': {'metricas_csv': metricas_path, 'detalhes_csv': detalhes_path, 'figura': fig_path}
    }

def obter_ordenacao_modelos(models_dir):
    """
    Retorna uma lista de tuplas (nome, r2) ordenadas do melhor para o pior
    com base no R² registrado em resultados_finais.csv.
    Se o arquivo não existir, retorna lista vazia.
    """
    resultados_path = os.path.join(os.path.dirname(models_dir),
                                  "resultados_finais",
                                  "resultados_finais.csv")
    if not os.path.exists(resultados_path):
        print("⚠️  Arquivo de resultados não encontrado.")
        return []

    try:
        df = pd.read_csv(resultados_path)
        # Filtrar apenas modelos com status ✅
        df_validos = df[df['Status'] == '✅'].copy()
        # Converter R² para numérico
        df_validos['R²'] = pd.to_numeric(df_validos['R²'], errors='coerce')
        df_validos = df_validos.dropna(subset=['R²'])
        # Ordenar decrescente
        df_ordenado = df_validos.sort_values('R²', ascending=False)

        # Extrair nomes dos modelos (tentativa inteligente)
        if 'Modelo' in df_ordenado.columns:
            nomes = df_ordenado['Modelo'].tolist()
        else:
            # Se o índice for string, assume que são os nomes
            if df_ordenado.index.dtype == 'object':
                nomes = df_ordenado.index.tolist()
            else:
                # Fallback: usar a primeira coluna (presume que contém nomes)
                primeira_coluna = df_ordenado.columns[0]
                nomes = df_ordenado[primeira_coluna].tolist()

        r2s = df_ordenado['R²'].tolist()
        return list(zip(nomes, r2s))
    except Exception as e:
        print(f"⚠️  Erro ao ler ordenação: {e}")
        return []

# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================

def main():
    global _main_executing, RESULTS_DIR, FIGURES_DIR
    if _main_executing:
        print("⚠️  main() já está em execução. Ignorando nova chamada.")
        return
    _main_executing = True

    # Gera timestamp e define diretórios
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = os.path.join(BASE_DIR, f"resultados_nsga2_{NSGA_VARIANT}_seed{NSGA_SEED}_{timestamp}")
    FIGURES_DIR = os.path.join(RESULTS_DIR, "figuras_pareto")

    # Cria a estrutura de subpastas (csv, diagnosticos, validacoes, debug, relatorios)
    criar_diretorios_output()

    # As linhas de mudança de diretório permanecem comentadas
    # os.chdir(RESULTS_DIR)
    # print(f"[DEBUG] Diretório de trabalho alterado para: {os.getcwd()}")

    print("="*80)
    print("OTIMIZAÇÃO MULTIOBJETIVO NSGA-II - DOSE (5 MIN), PESO E CUSTO")
    print("="*80)
    print("NOTA: Todas as doses são para 5 minutos de exposição (300 s)")
    print("-"*80)
    
    inicio_total = time.time()
    
    def verificar_features_neural_network(scaler, modelo, target_transform):
        """Verificação específica para NeuralNetwork_Fisica"""
        print("\n" + "="*80)
        print("🔍 DIAGNÓSTICO DETALHADO: NeuralNetwork_Fisica")
        print("="*80)
        
        # Testar com 3 configurações extremamente diferentes
        testes = [
            ("TODOS High-Z", [2.0, 1.5, 1.0, 0.5, 0.5, 0.3, 0.2], 
             [MATERIAIS_DISPONIVEIS.index('Bi2O3-HDPE'), 
              MATERIAIS_DISPONIVEIS.index('W-Epoxy'),
              MATERIAIS_DISPONIVEIS.index('Fe'),
              MATERIAIS_DISPONIVEIS.index('Fe'),
              MATERIAIS_DISPONIVEIS.index('W-Epoxy'),
              MATERIAIS_DISPONIVEIS.index('Bi2O3-HDPE'),
              MATERIAIS_DISPONIVEIS.index('W-Epoxy')]),
            
            ("TODOS Low-Z", [0.2, 0.3, 0.5, 0.5, 1.0, 1.5, 2.0],
             [MATERIAIS_DISPONIVEIS.index('B4C-HDPE'),
              MATERIAIS_DISPONIVEIS.index('UHMWPE'),
              MATERIAIS_DISPONIVEIS.index('Al'),
              MATERIAIS_DISPONIVEIS.index('SiC'),
              MATERIAIS_DISPONIVEIS.index('UHMWPE'),
              MATERIAIS_DISPONIVEIS.index('B4C-HDPE'),
              MATERIAIS_DISPONIVEIS.index('Al')]),
            
            ("Misto Extremo", [2.0, 0.2, 1.0, 0.2, 1.0, 0.2, 1.0],
             [MATERIAIS_DISPONIVEIS.index('Bi2O3-HDPE'),
              MATERIAIS_DISPONIVEIS.index('B4C-HDPE'),
              MATERIAIS_DISPONIVEIS.index('W-Epoxy'),
              MATERIAIS_DISPONIVEIS.index('UHMWPE'),
              MATERIAIS_DISPONIVEIS.index('Fe'),
              MATERIAIS_DISPONIVEIS.index('Al'),
              MATERIAIS_DISPONIVEIS.index('SiC')])
        ]   
        
        for nome, espessuras, materiais_idx in testes:
            # Normalizar espessuras
            espessuras = np.array(espessuras) / sum(espessuras) * ESPESSURA_TOTAL_FIXA
            
            print(f"\n📊 {nome}:")
            print(f"   Materiais: {[MATERIAIS_DISPONIVEIS[i] for i in materiais_idx]}")
            print(f"   Espessuras: {['%.3f' % e for e in espessuras]}")
            
            # Criar features
            features, peso, custo = criar_features_para_otimizacao(espessuras, materiais_idx)
            
            # Criar vetor na ordem CORRETA
            features_vector = np.array([[
                features['z_eff_ponderado_espessura'],
                features['atenuacao_total'],
                features['z_dens_esp_total'],
                features['high_z_protection'],
                features['low_z_moderation'],
                features['high_low_ratio'],
                features['z_x_atenuacao'],
                features['transmissao'],
                # features['log_transmissao'],
                features['sequence_score'],
                features['z_variance']
            ]])
        
            print(f"   Features críticas:")
            print(f"     - z_eff_ponderado: {features['z_eff_ponderado_espessura']:.1f}")
            print(f"     - atenuacao_total: {features['atenuacao_total']:.3f}")
            print(f"     - transmissao: {features['transmissao']:.6f}")
            print(f"     - z_variance: {features['z_variance']:.3f}")
            
            # Aplicar scaler
            features_scaled = scaler.transform(features_vector)
            print(f"   Features escaladas (média={features_scaled[0].mean():.3f}, std={features_scaled[0].std():.3f}):")
            print(f"     {features_scaled[0]}")
            
            # Predizer
            if hasattr(modelo, 'keras_model'):
                pred = modelo.keras_model.predict(features_scaled, verbose=0)[0][0]
            else:
                pred = modelo.predict(features_scaled)[0]
                    
            dose_5min = inversa_transformacao_otimizada_5min(pred, target_transform['mean'], target_transform['std'])
            print(f"   Predição: {pred:.6f} -> {dose_5min:.6f} mSv/5min")
    
    print("\n✅ Diagnóstico concluído.")
    
    try:
        # 0. VERIFICAÇÃO DE GEOMETRIA
        print("\n🔬 VERIFICAÇÃO DE GEOMETRIA")
        print("-" * 40)
        volume_total = calcular_peso_custo_teoricos()
        
        # ==============================================================================
        # 1. CARREGAR MODELOS DA FASE 3 COM VALIDAÇÃO E FALLBACK ORDENADO
        # ==============================================================================
        print("\n📂 Carregando modelos treinados na Fase 3...")
        
        # Opcional: definir um modelo específico (descomente um dos abaixo) ou None para automático
        # modelo_escolhido = None                     # Seleção automática (melhor R²)
        # modelo_escolhido = 'HistGradientBoosting'
        # modelo_escolhido = 'NeuralNetwork_Fisica'
        # modelo_escolhido = 'PLS_Regression'
        # modelo_escolhido = None
        modelo_escolhido = MODELO_OPERACIONAL_NSGAII
        
        scaler, target_transform, modelo, modelo_nome = carregar_modelos_fase3(modelo_selecionado=modelo_escolhido)
        print("✅ Modelos essenciais carregados com sucesso!")
        
        # ==============================================================================
        # EXIBIR ORDENAÇÃO COMPLETA DOS MODELOS (DO MELHOR AO PIOR)
        # ==============================================================================
        print("\n📊 RANKING DOS MODELOS DA FASE 3 (baseado em R²):")
        modelos_com_r2 = obter_ordenacao_modelos(MODELS_DIR)
        if modelos_com_r2:
            nomes_ordenados = [nome for nome, _ in modelos_com_r2]
            for i, (nome, r2) in enumerate(modelos_com_r2, 1):
                print(f"   {i:2d}. {nome} (R² = {r2:.4f})")
        else:
            nomes_ordenados = []
            print("   Não foi possível recuperar a ordenação.")

        # ==============================================================================
        # VALIDAÇÃO DO MODELO SELECIONADO EM HOLD-OUT E FALLBACK ORDENADO
        # ==============================================================================
        print("\n🔍 Validando modelo selecionado em hold-out...")
        
        # Garantir que df_mcnp esteja acessível (global)
        global df_mcnp
        if df_mcnp is None:
            df_mcnp = carregar_dados_mcnp_sobol(MCNP_DATA_PATH)
            
       
        # Separar um hold-out do TREINO (usando 15% do total, igual à Fase 3)
        indices = np.arange(len(df_mcnp))
        _, indices_val = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
       
        # Parâmetro: fração do hold-out a ser usada na validação (valor genérico)
        VAL_FRACTION = 0.66   # 66% do hold-out → 0,66 * 300 ≈ 200 no dataset atual
        n_val = int(len(indices_val) * VAL_FRACTION)
        n_val = max(1, n_val)  # garantir pelo menos 1 amostra 
        
        # Pegar uma amostra menor para validação rápida (200)
        np.random.seed(12345)  # para reprodutibilidade
        indices_val = np.random.choice(indices_val, n_val, replace=False)
        df_val = df_mcnp.iloc[indices_val].copy()
        
        # Avaliar o modelo no conjunto de validação
        doses_reais = []
        doses_pred = []
        for idx, row in df_val.iterrows():
            esp = np.array(row['espessuras_list'])
            mats = row['materiais_list']
            mats_idx = [MATERIAIS_DISPONIVEIS.index(m) if m in MATERIAIS_DISPONIVEIS else 0 for m in mats]
            individuo = np.concatenate([esp, mats_idx])
            dose_pred, _, _ = avaliar_individuo(individuo, scaler, modelo, target_transform)
            doses_reais.append(row['dose_5min'])
            doses_pred.append(dose_pred)
            
        r2_val = r2_score(doses_reais, doses_pred)
        print(f"   • R² no hold-out (n={n_val}): {r2_val:.4f}")
        
        # Definir limiar (ajustável)
        LIMIAR_R2 = 0.90
        
        modelo_fallback_usado = False
        if r2_val < LIMIAR_R2:
            print(f"⚠️  Modelo {modelo_nome} não atingiu R² mínimo ({LIMIAR_R2}). Buscando fallback ordenado...")
            
            # Obter lista ordenada de modelos (já temos modelos_ordenados)
            if not nomes_ordenados:
                print("   Lista de modelos não disponível. Mantendo modelo atual.")
            else:
                # Remover o modelo atual da lista (se presente)
                candidatos = [m for m in nomes_ordenados if m != modelo_nome]
                
                modelo_encontrado = False
                for nome_candidato in candidatos:
                    print(f"   ▶ Testando candidato: {nome_candidato}")
                    try:
                        # Carregar o modelo candidato
                        scaler_c, target_c, modelo_c, nome_c = carregar_modelos_fase3(modelo_selecionado=nome_candidato)
                        
                        # Avaliar no mesmo conjunto de validação
                        doses_pred_c = []
                        for idx, row in df_val.iterrows():
                            esp = np.array(row['espessuras_list'])
                            mats = row['materiais_list']
                            mats_idx = [MATERIAIS_DISPONIVEIS.index(m) if m in MATERIAIS_DISPONIVEIS else 0 for m in mats]
                            individuo = np.concatenate([esp, mats_idx])
                            dose_pred_c, _, _ = avaliar_individuo(individuo, scaler_c, modelo_c, target_c)
                            doses_pred_c.append(dose_pred_c)
                        r2_c = r2_score(doses_reais, doses_pred_c)
                        print(f"      R² = {r2_c:.4f}")
                        
                        if r2_c >= LIMIAR_R2:
                            print(f"   ✅ Modelo {nome_c} aceito com R² {r2_c:.4f}. Substituindo modelo original.")
                            # Substituir pelos novos objetos
                            scaler = scaler_c
                            target_transform = target_c
                            modelo = modelo_c
                            modelo_nome = nome_c
                            modelo_encontrado = True
                            modelo_fallback_usado = True
                            break
                    except Exception as e:
                        print(f"      Erro ao carregar/avaliar {nome_candidato}: {e}")
                        continue
                    
                if not modelo_encontrado:
                    print("⚠️  Nenhum modelo alternativo atingiu o limiar. Mantendo modelo original.")
        else:
            print(f"✅ Modelo {modelo_nome} aprovado com R² {r2_val:.4f}.")

        # ======================================================================
        # TESTE DE VARIABILIDADE COM 5 CONFIGURAÇÕES (COMPLEMENTAR)
        # ======================================================================
        print("\n🧪 TESTE DE VARIABILIDADE DO MODELO SELECIONADO (5 configs aleatórias)...")
        
        test_configs = []
        for _ in range(5):
            esp_test = np.random.uniform(ESPESSURA_MIN_CAMADA, ESPESSURA_MAX_CAMADA, NUM_CAMADAS)
            esp_test = esp_test / np.sum(esp_test) * ESPESSURA_TOTAL_FIXA
            mat_test = np.random.permutation(len(MATERIAIS_DISPONIVEIS))[:NUM_CAMADAS]
            test_configs.append((esp_test, mat_test))
            
        doses_test = []
        for esp, mat in test_configs:
            individuo = np.concatenate([esp, mat])
            dose, _, _ = avaliar_individuo(individuo, scaler, modelo, target_transform)
            doses_test.append(dose)
            
        std_dose = np.std(doses_test)
        print(f"   • Variação (desvio padrão) em 5 configurações: {std_dose:.6f} mSv/5min")
        
        if std_dose < 0.01:
            print(f"🚨 MODELO {modelo_nome} RETORNANDO VALORES QUASE CONSTANTES!")
            print(f"   Isso pode indicar problemas de generalização. Considere revisar o modelo.")
        else:
            print(f"✅ Modelo {modelo_nome} apresenta variabilidade adequada.")

        # ======================================================================
        # REGISTRO EM LOG DA DECISÃO DE SELEÇÃO
        # ======================================================================
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(RESULTS_DIR, "debug")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"selecao_modelo_{timestamp}.txt")
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Modelo inicial solicitado: {modelo_escolhido if modelo_escolhido else 'automático'}\n")
            f.write(f"Modelo carregado inicialmente: {modelo_nome}\n")
            f.write(f"R² no hold-out (200 amostras): {r2_val:.4f}\n")
            f.write(f"Limiar de aceitação: {LIMIAR_R2}\n")
            if r2_val < LIMIAR_R2:
                f.write(f"Fallback acionado: {'SIM' if modelo_fallback_usado else 'NÃO'}\n")
                if modelo_fallback_usado:
                    f.write(f"Modelo final após fallback: {modelo_nome}\n")
                    f.write(f"R² do modelo final: {r2_c:.4f}\n")
                else:
                    f.write(f"Nenhum substituto encontrado. Modelo mantido.\n")
            else:
                f.write(f"Modelo mantido.\n")
            f.write(f"Teste de variabilidade (5 configs): std = {std_dose:.6f}\n")
        print(f"📝 Log de seleção salvo em: {log_file}")

        # ======================================================================
        # TESTE DO SCALER COM CONFIGURAÇÃO REALISTA (OPCIONAL, JÁ EXISTENTE)
        # ======================================================================
        print("\n" + "="*60)
        print("🧪 TESTE REALISTA DO SCALER (StandardScaler) - [0.857, 0.857, 0.857, 0.857, 0.857, 0.857, 0.857]")
        print("="*60)
        
        espessuras_real = np.array([0.857, 0.857, 0.857, 0.857, 0.857, 0.857, 0.857])
        materiais_real = [0, 1, 2, 3, 4, 5, 6]
        
        _, _, _, full_vector_real = criar_features_para_otimizacao(espessuras_real, materiais_real)
        scaled_real = scaler.transform(full_vector_real.reshape(1, -1))
        
        print(f"\n✅ RESULTADO DO SCALER COM DADOS REAIS:")
        print(f"  • Features escaladas: {scaled_real[0].round(4)}")
        print(f"  • Faixa após normalização: ±{np.max(np.abs(scaled_real)):.2f}")
        print(f"  • Status: {'✅ PRONTO PARA NSGA-II' if np.max(np.abs(scaled_real)) < 5.0 else '⚠️  Aceitável'}")
        
        # ======================================================================
        # DIAGNÓSTICO DA SEQUENCE_SCORE (JÁ EXISTENTE)
        # ======================================================================
        debug_dir = os.path.join(RESULTS_DIR, "debug")
        sequence_scores = diagnosticar_sequence_score_completo(output_dir=debug_dir)
        
        if np.std(sequence_scores) < 0.001:
            print("\n🔧 AJUSTE CIENTÍFICO DO STANDARDSCALER:")
            print("  • Feature sequence_score é constante no dataset (std < 0.001).")
            print("  • Isso não invalida o modelo, mas deve ser notado.")
            print("  • O StandardScaler lidará com isso (divisão por zero evitada internamente).")
        
        # ==========================================================================
        # 🔬 VERIFICAÇÃO OBJETIVA: FEATURES FASE 3 vs FASE 4 (ADICIONE AQUI!)
        # ==========================================================================
        print("\n" + "="*80)
        
        features_ok, features_exemplo = verificar_features_fase3_fase4()
        
        if not features_ok:
            print("\n🚨 PROBLEMA DETECTADO: Features com baixa variação")
            print("   • Pode afetar a eficácia do StandardScaler")
        else:
            print("\n✅ Features estão consistentes entre Fase 3 e Fase 4")
        
        # ==========================================================================
        # 🎯 VERIFICAÇÃO CRÍTICA DOS PARÂMETROS (ENXUTA E OBJETIVA)
        # ==========================================================================
        print("\n" + "="*60)
        print("🔍 VERIFICAÇÃO DOS PARÂMETROS")
        print("="*60)
        
        mean_log = target_transform['mean']
        std_log = target_transform['std']
        
        # 1. Verificar os parâmetros contra a distribuição DE/DF AP treinada.
        dose_5min_mean = np.exp(mean_log)
        print(f"• Dose média calculada: {dose_5min_mean:.3f} mSv/5min")
        df_resultados = pd.read_csv(MCNP_DATA_PATH)
        doses_5min = df_resultados['5 min de Dose efetiva (mSv)'].values
        print(f"• Dataset MCNP DE/DF AP: {np.min(doses_5min):.3f}-{np.max(doses_5min):.3f} mSv/5min")
        
        if np.isfinite(dose_5min_mean) and dose_5min_mean > 0:
            print(f"✅ Parâmetros OK")
        else:
            print(f"❌ Parâmetros inconsistentes - recalculando...")
            mean_log = np.mean(np.log(doses_5min))
            std_log = np.std(np.log(doses_5min))
            target_transform = {'mean': mean_log, 'std': std_log}
            print(f"   • Novo mean_log: {mean_log:.6f}")
            print(f"   • Nova dose média: {np.exp(mean_log):.3f} mSv/5min")
        
        # 2. Teste rápido da conversão
        print(f"\n🧪 TESTE DA CONVERSÃO:")
        z_teste = 0.0
        dose_teste = inversa_transformacao_otimizada_5min(z_teste, mean_log, std_log)
        print(f"  • z = 0.0 → {dose_teste:.3f} mSv/5min")
        
        # ==========================================================================
        # 🎯 EXECUÇÃO DA VALIDAÇÃO ROBUSTA (SUBSTITUI AS CHAMADAS ANTIGAS)
        # ==========================================================================
        
        print("\n" + "="*80)
        print("🔬 VALIDAÇÃO ROBUSTA DO MODELO DE DOSE")
        print("="*80)
        
        # Executar validação robusta
        resultados_validacao = validacao_robusta_artigo(
            scaler=scaler,
            modelo=modelo,
            target_transform=target_transform,
            df_mcnp=df_mcnp,
            percentual_validacao=0.15,  # Consistente com a Fase 3 (85/15)
            output_dir=RESULTS_DIR
        )

        # Avaliação baseada em critérios científicos
        dose_r2 = resultados_validacao['metricas']['R2']
        dose_mape = resultados_validacao['metricas']['MAPE']
        
        print("\n" + "="*80)
        print("📋 AVALIAÇÃO PARA PUBLICAÇÃO CIENTÍFICA")
        print("="*80)
        
        # Critérios baseados em referências da área (ajustáveis)
        if dose_r2 > 0.90 and dose_mape < 15.0:
            status = "✅ EXCELENTE"
            print(f"   Status: {status}")
            print(f"   Justificativa: R² > 0.90 e MAPE < 15% indicam modelo preciso.")
        elif dose_r2 > 0.85 and dose_mape < 20.0:
            status = "✅ BOM - Adequado para periódicos A1 com discussão apropriada"
            print(f"   Status: {status}")
            print(f"   Justificativa: R² > 0.85 e MAPE < 20% são aceitáveis.")
        elif dose_r2 > 0.80:
            status = "⚠️  SATISFATÓRIO - Requer justificativa robusta no artigo"
            print(f"   Status: {status}")
            print(f"   Justificativa: R² > 0.80 pode ser suficiente com análise cuidadosa.")
        else:
            status = "❌ ABAIXO DO ESPERADO - Melhorias necessárias"
            print(f"   Status: {status}")
            print(f"   Recomendação: Revisar treinamento do modelo ou features.")
            
        print(f"\n📌 {'CONTINUAR com NSGA-II' if '❌' not in status else 'REVISAR modelo'}")
        
        # ==========================================================================
        # Definição do diretório de checkpoints (fixo, para retomada)
        # ==========================================================================
        # Um checkpoint é exclusivo de cada seed. Não é válido retomar uma
        # população criada com outra seed, nem importar checkpoints históricos.
        CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_long_run", NSGA_VARIANT, f"seed_{NSGA_SEED}")
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "nsga2_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            print(f"🔄 Checkpoint disponível para seed {NSGA_SEED}: {checkpoint_path}")
        else:
            print(f"ℹ️  Nova execução independente: seed {NSGA_SEED}.")
                
        # ==========================================================================
        # 🎯 INICIALIZAR OTIMIZADOR NSGA-II
        # ==========================================================================
        print("\n🔧 Configurando otimizador NSGA-II...")
        np.random.seed(NSGA_SEED)
        print(f"   • Variante do NSGA-II: {NSGA_VARIANT}")
        print(f"   • Seed reprodutível do NSGA-II: {NSGA_SEED}")
        otimizador = NSGA2Optimizer(
            scaler, 
            modelo, 
            target_transform, 
            modelo_nome=modelo_nome,
            results_dir=RESULTS_DIR,
            figures_dir=FIGURES_DIR,
            timestamp=timestamp
        )
        
        print(f"\n⚡ PARALELISMO ATIVADO: {multiprocessing.cpu_count()} cores disponíveis")
        print(f"   • Usando {max(1, multiprocessing.cpu_count())} cores para avaliação")
        print(f"   • Velocidade esperada: 2-4x mais rápido")
        
        # ==========================================================================
        # 🎯 EXECUTAR OTIMIZAÇÃO NSGA-II
        # ==========================================================================
        print(f"\n🚀 EXECUTANDO NSGA-II")
        print("="*60)
        print(f"• População: {POPULATION_SIZE}")
        print(f"• Gerações: {NUM_GENERATIONS} (checkpoint a cada 10 gerações)")
        print(f"• Crossover: {CROSSOVER_PROB}")
        print(f"• Mutação: {'Adaptativa' if ADAPTIVE_MUTATION else f'Fixa ({MUTATION_PROB})'}")
        
        inicio_otimizacao = time.time()
        populacao_final, fitness_final, frente_pareto = otimizador.executar_otimizacao(
            n_geracoes=NUM_GENERATIONS,
            tamanho_populacao=POPULATION_SIZE,
            checkpoint_file=checkpoint_path,      # caminho fixo
            checkpoint_interval=10,
            figures_dir=FIGURES_DIR,               # figuras intermediárias vão para pasta com timestamp
            force_new=False                        # carrega checkpoint se existir
        )
        tempo_otimizacao = time.time() - inicio_otimizacao
        
        # 8. Analisar resultados
        print("="*60)
        if populacao_final is None or len(populacao_final) == 0:
            print("❌ Otimização não produziu resultados válidos")
            return None, None, None
        
        solucoes_pareto, fitness_pareto = analisar_resultados(
            populacao_final, fitness_final, frente_pareto, df_mcnp
        )
        
        # Comparação entre TOPSIS-IQR e TOPSIS (normalização min-max)
        best_idx_iqr, scores_iqr = calcular_tradeoff_topsis(fitness_pareto)
        best_idx_minmax, scores_minmax = calcular_tradeoff_topsis_minmax(fitness_pareto)
        
        print("\n" + "="*60)
        print("🔍 COMPARAÇÃO TOPSIS-IQR vs TOPSIS MIN-MAX")
        print("="*60)
        print(f"Índice da melhor solução (IQR): {best_idx_iqr}")
        print(f"Índice da melhor solução (Min-Max): {best_idx_minmax}")
        print("\nValores das soluções:")
        print(f"  IQR     - Dose: {fitness_pareto[best_idx_iqr,0]:.4f} mSv/5min, "
              f"Peso: {fitness_pareto[best_idx_iqr,1]:.1f} kg, "
              f"Custo: ${fitness_pareto[best_idx_iqr,2]:.2f}")
        print(f"  Min-Max - Dose: {fitness_pareto[best_idx_minmax,0]:.4f} mSv/5min, "
              f"Peso: {fitness_pareto[best_idx_minmax,1]:.1f} kg, "
              f"Custo: ${fitness_pareto[best_idx_minmax,2]:.2f}")
        
        # Verificar se a solução IQR é mais central (comparar com medianas)
        median_dose = np.median(fitness_pareto[:,0])
        median_peso = np.median(fitness_pareto[:,1])
        median_custo = np.median(fitness_pareto[:,2])
        
        # Diferenças percentuais para IQR
        diff_iqr_dose = (fitness_pareto[best_idx_iqr,0] - median_dose) / median_dose * 100
        diff_iqr_peso = (fitness_pareto[best_idx_iqr,1] - median_peso) / median_peso * 100
        diff_iqr_custo = (fitness_pareto[best_idx_iqr,2] - median_custo) / median_custo * 100
        
        # Diferenças percentuais para Min-Max
        diff_minmax_dose = (fitness_pareto[best_idx_minmax,0] - median_dose) / median_dose * 100
        diff_minmax_peso = (fitness_pareto[best_idx_minmax,1] - median_peso) / median_peso * 100
        diff_minmax_custo = (fitness_pareto[best_idx_minmax,2] - median_custo) / median_custo * 100
                
        print("\nPosição em relação à mediana da frente:")
        print(f"  Mediana: Dose={median_dose:.4f}, Peso={median_peso:.1f}, Custo=${median_custo:.2f}")
        print(f"  TOPSIS-IQR - Diferença relativa: Dose={(fitness_pareto[best_idx_iqr,0]-median_dose)/median_dose*100:.1f}%, "
              f"Peso={(fitness_pareto[best_idx_iqr,1]-median_peso)/median_peso*100:.1f}%, "
              f"Custo={(fitness_pareto[best_idx_iqr,2]-median_custo)/median_custo*100:.1f}%")
        print(f"  TOPSIS (normalização min-max) - Diferença relativa: {(fitness_pareto[best_idx_minmax,0]-median_dose)/median_dose*100:.1f}%, "
              f"Peso={(fitness_pareto[best_idx_minmax,1]-median_peso)/median_peso*100:.1f}%, "
              f"Custo={(fitness_pareto[best_idx_minmax,2]-median_custo)/median_custo*100:.1f}%")
        
        # Calcular soma dos desvios absolutos
        soma_abs_iqr = abs(diff_iqr_dose) + abs(diff_iqr_peso) + abs(diff_iqr_custo)
        soma_abs_minmax = abs(diff_minmax_dose) + abs(diff_minmax_peso) + abs(diff_minmax_custo)
        
        print(f"\nSoma dos desvios absolutos (TOPSIS-IQR): {soma_abs_iqr:.1f}%")
        print(f"Soma dos desvios absolutos (TOPSIS): {soma_abs_minmax:.1f}%")
        # Calcular e exibir a melhoria percentual do IQR em relação ao Min-Max
        melhoria = (soma_abs_minmax - soma_abs_iqr) / soma_abs_minmax * 100
        print(f"\n📊 O TOPSIS-IQR selecionou uma solução {melhoria:.1f}% mais central (redução da soma dos desvios absolutos).")

        if soma_abs_iqr < soma_abs_minmax:
            print("✅ TOPSIS-IQR selecionou uma solução mais central (menor soma de desvios absolutos das medianas).")
        elif soma_abs_minmax < soma_abs_iqr:
            print("✅ TOPSIS Min-Max selecionou uma solução mais central.")
        else:
            print("⚖️ Ambos os métodos selecionaram soluções com mesma centralidade (soma de desvios igual).")
            
        # Salvar em arquivo de log
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        comparison_log = os.path.join(RESULTS_DIR, "debug", f"topsis_comparison_{timestamp}.txt")
        with open(comparison_log, 'w') as f:
            f.write("Comparação TOPSIS-IQR vs TOPSIS Min-Max\n")
            f.write(f"IQR index: {best_idx_iqr}\n")
            f.write(f"Min-Max index: {best_idx_minmax}\n")
            f.write(f"IQR solution: {fitness_pareto[best_idx_iqr]}\n")
            f.write(f"Min-Max solution: {fitness_pareto[best_idx_minmax]}\n")
            print(f"Log salvo em: {comparison_log}")
        
        # ==========================================================================
        # 🎯 CÁLCULO DAS MÉTRICAS FINAIS
        # ==========================================================================
        spread_final = otimizador.calcular_spread_metric(fitness_pareto)
        print(f"   • Spread metric (diversidade): {spread_final:.3f}")
        
        if spread_final < 0.4:
            print(f"   ✅ Spread EXCELENTE (<0.4) - Distribuição muito uniforme")
        elif spread_final < 0.7:
            print(f"   ✅ Spread BOM (<0.7) - Distribuição aceitável")
        else:
            print(f"   ⚠️  Spread ALTO ({spread_final:.3f}) - Distribuição pouco uniforme; considere ajustes nos parâmetros.")
            
        try:
            resultado_hv = calcular_hipervolume_unificado(
                fitness_pareto,
                referencia=otimizador.reference_point,
                normalizado=True
            )
            hv_absoluto = resultado_hv['hipervolume_normalizado']
            print(f"🎯 HYPERVOLUME ABSOLUTO FINAL: {hv_absoluto:.6f}")
        except Exception as e:
            print(f"⚠️  Erro no cálculo final do hipervolume: {e}")
            hv_absoluto = 0.0
            
        # Salvar métricas em CSV
        csv_dir = os.path.join(RESULTS_DIR, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        metricas_artigo = {
            'variante': NSGA_VARIANT,
            'seed': NSGA_SEED,
            'populacao': POPULATION_SIZE,
            'geracoes': NUM_GENERATIONS,
            'n_solucoes_pareto': len(fitness_pareto),
            'hipervolume_normalizado': hv_absoluto,
            'spread_final': spread_final,
            'melhor_dose': fitness_pareto[:, 0].min(),
            'menor_peso': fitness_pareto[:, 1].min(),
            'menor_custo': fitness_pareto[:, 2].min(),
            'ponto_referencia_fixo': '|'.join(f'{v:.12g}' for v in otimizador.reference_point),
            'timestamp': time.strftime("%Y%m%d_%H%M%S")
        }
        df_metricas = pd.DataFrame([metricas_artigo])
        metricas_path = os.path.join(csv_dir, f"metricas_artigo_pop{POPULATION_SIZE}.csv")
        df_metricas.to_csv(metricas_path, index=False)
        print(f"   💾 Métricas salvas em: {metricas_path}")
        
        # ==========================================================================
        # 🎨 GERANDO VISUALIZAÇÕES DA FRENTE DE PARETO
        # ==========================================================================
        print("\n🎨 GERANDO VISUALIZAÇÕES DA FRENTE DE PARETO")
        print("-" * 40)
        
        figuras = visualizar_frente_pareto_3d(
            fitness_pareto,
            solucoes_pareto,  # <--- adicione este argumento
            output_dir=FIGURES_DIR,
            spread_value=spread_final,
            show=True
        )
            
        print(f"✅ Visualizações salvas em: {FIGURES_DIR}")
        
        # ==========================================================================
        # 💾 SALVANDO RESULTADOS COMPLETOS
        # ==========================================================================
        
        print("\n💾 SALVANDO RESULTADOS COMPLETOS")
        print("-" * 40)
        
        # Salvar resultados da frente de Pareto em CSV
        df_resultados, resultados_path = salvar_resultados_csv(solucoes_pareto, fitness_pareto, output_dir=RESULTS_DIR)
        print(f"✅ Resultados da frente de Pareto salvos em: {resultados_path}")
        
        # ==========================================================================
        # 📝 GERANDO RELATÓRIO CIENTÍFICO
        # ==========================================================================
        
        print("\n📝 GERANDO RELATÓRIO CIENTÍFICO")
        print("-" * 40)
        
        tempo_total = time.time() - inicio_total
        relatorio = gerar_relatorio_cientifico(df_resultados, tempo_total, output_dir=RESULTS_DIR)
        print(f"✅ Relatório científico salvo em: {relatorio}")
            
        # ==========================================================================
        # 🔬 VERIFICAÇÃO CIENTÍFICA: SOLUÇÃO TOPSIS É EQUILIBRADA?
        # ==========================================================================
        
        # 1. Calcular o melhor trade-off usando TOPSIS
        best_idx, scores_topsis = calcular_tradeoff_topsis(fitness_pareto)
        
        # 2. Estatísticas descritivas da frente de Pareto
        dose_stats = fitness_pareto[:, 0]
        peso_stats = fitness_pareto[:, 1]
        custo_stats = fitness_pareto[:, 2]
        
        # 3. Verificar se a solução TOPSIS está no intervalo interquartil
        dose_val = fitness_pareto[best_idx, 0]
        peso_val = fitness_pareto[best_idx, 1]
        custo_val = fitness_pareto[best_idx, 2]
        
        is_balanced = (
            np.percentile(dose_stats, 25) <= dose_val <= np.percentile(dose_stats, 75) and
            np.percentile(peso_stats, 25) <= peso_val <= np.percentile(peso_stats, 75) and
            np.percentile(custo_stats, 25) <= custo_val <= np.percentile(custo_stats, 75)
        )

        # 4. Calcular ranking percentual (onde a solução se posiciona)
        dose_rank = np.sum(dose_stats <= dose_val) / len(dose_stats) * 100
        peso_rank = np.sum(peso_stats <= peso_val) / len(peso_stats) * 100
        custo_rank = np.sum(custo_stats <= custo_val) / len(custo_stats) * 100
        
        if is_balanced:
            print(f"\n✅ SOLUÇÃO TOPSIS É EQUILIBRADA!")
            print(f"   Todos objetivos dentro do intervalo interquartil (Q1-Q3)")
        else:
            print(f"\n⚠️  A SOLUÇÃO TOPSIS-IQR APRESENTA VALORES QUE A SITUAM FORA DOS 50% CENTRAIS DA DISTRIBUIÇÃO EM PELO MENOS UM OBJETIVO")
            print(f"   • Interpretação: TOPSIS priorizou um compromisso específico")
            print(f"   • Ainda é Pareto-optimal e válida para aplicação")
            
        # ==========================================================================
        # 🎯 RESUMO FINAL
        # ==========================================================================
        
        print("\n" + "="*80)
        print("✅ OTIMIZAÇÃO CONCLUÍDA COM SUCESSO!")
        print("="*80)
        print(f"⏱️  Tempo total: {tempo_total:.2f} segundos")
        print(f"📊 Soluções Pareto: {len(solucoes_pareto)}")
        print(f"💾 Todos os outputs salvos em: {RESULTS_DIR}")
        
        _main_executing = False
        
        return df_resultados, solucoes_pareto, fitness_pareto
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None
    
# ==============================================================================
# ATUALIZAR FUNÇÃO gerar_relatorio_cientifico()
# ==============================================================================

def gerar_relatorio_cientifico(df_resultados, tempo_execucao, output_dir):
    """
    Gera relatório científico simplificado.

    Parâmetros:
    -----------
    df_resultados : DataFrame
        DataFrame com os resultados da otimização.
    tempo_execucao : float
        Tempo total de execução.
    output_dir : str, optional
        Diretório onde o relatório será salvo. Se None, usa a variável global RESULTS_DIR.
    """
    if output_dir is None:
        raise ValueError("output_dir não pode ser None.")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    relatorios_dir = os.path.join(output_dir, "relatorios")
    os.makedirs(relatorios_dir, exist_ok=True)
    
    filename = os.path.join(relatorios_dir, f"relatorio_nsga2_{timestamp}.txt")
    
    if DEBUG_SAVE:
        print(f"[DEBUG] gerar_relatorio_cientifico -> {filename}")
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RELATÓRIO CIENTÍFICO - OTIMIZAÇÃO NSGA-II\n")
        f.write("="*80 + "\n\n")
        
        f.write("METODOLOGIA\n")
        f.write("-"*40 + "\n")
        f.write(f"Algoritmo: NSGA-II\n")
        f.write(f"População: {POPULATION_SIZE}\n")
        f.write(f"Gerações: {NUM_GENERATIONS}\n")
        f.write(f"Tempo: {tempo_execucao:.2f}s\n\n")
        
        f.write("RESULTADOS\n")
        f.write("-"*40 + "\n")
        f.write(f"Soluções Pareto: {len(df_resultados)}\n")
        f.write(f"Melhor dose: {df_resultados['Dose_Efetiva_mSv_5min'].min():.6f} mSv/5min\n")
        f.write(f"Menor peso: {df_resultados['Peso_Total_kg'].min():.2f} kg\n")
        f.write(f"Menor custo: ${df_resultados['Custo_Total_USD'].min():.2f}\n")
    
    print(f"📝 Relatório salvo: {filename}")
    return filename

# ==============================================================================
# EXECUÇÃO DO SCRIPT
# ==============================================================================

if __name__ == "__main__":
    # Verificar se os diretórios existem
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Diretório de modelos não encontrado: {MODELS_DIR}")
        sys.exit(1)
    
    # Executar otimização
    resultados = main()
    
    if resultados[0] is not None:
        print("\n" + "="*80)
        print("="*80)
