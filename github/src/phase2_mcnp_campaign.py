# -*- coding: utf-8 -*-
"""
==========================================================================
Fase 2: AUTOMAÇÃO DAS SIMULAÇÕES MCNP E EXTRAÇÃO DE RESULTADOS (COM PARALELISMO)
========================================================================== 
NOTA DE CONTROLE (campanha final DE/DF AP): a especificação executável é
MCNP_TEMPLATE, definida abaixo do bloco de configurações. Ela usa fonte em
x=121 cm, W-epóxi com densidade 4,643 g/cm³, DE4/DF4 da ICRP 116 em AP,
FM4=7400 e NPS=10^8. Os exemplos históricos mantidos nesta introdução não
são executados e não devem ser usados como especificação metodológica.
==========================================================================
Objetivo: Automatizar a geração de 2000 arquivos inputs, executar as simulações
e extrair resultados de dose efetiva para cada configuração de blindagem gerada
via Sequências de Sobol (Fase 1 do Artigo).
==========================================================================
Como são 2000 configurações de blindagens, serão 2000 arquivos de input MCNP (.i)
==========================================================================
Atividade: Ler as 2000 configurações gravadas em 'configs_simulacao_reais_sobol.csv',
criar um input para cada linha, executar o MCNP e extrair os resultados.
==========================================================================
==========================================================================
Estrutura: Configurações do usuário (1); Definição dos materiais (2);
Template do input do MCNP (3); Funções auxiliares (4); Loop de geração (5);
==========================================================================
Loop de geração (5):

Cria os diretórios necessários; Carrega o 'configs_simulacao_reais_sobol.csv'; Verifica os materiais;
Inicia a geração dos input (Extração de materiais e espessuras; Validação da geometria;
Cálculo dos raios externos das camadas; Geração dos materiais; Preparação dos dados para
o template; Preenchimento das camadas; Geração do conteúdo do input MCNP; Criação do
diretório para a simulação; Salvamento do arquivo de input; Submeter simulações para
execução paralela; Coleta e processa os resultados das simulações; Salvamento dos resultados consolidados.
==========================================================================
Observação 1: Existem 2 materiais apenas que não estavam normalizados e foram normalizados, automaticamente,
pelo MCNP. São eles o M6 (Casco da granada), que faltou 0.000100 e o M8 (Ar), que faltou 0.000360.
Observação 2: RCC é Right Circular Cylinder (Cilindro circular reto: altura e diâmetro da base são independentes);
RCC X Y Z Hx Hy Hz R: X, Y, Z são coordenadas do ponto inicial (base do cilindro);
Hx, Hy, Hz compõem o vetor altura (direção e comprimento do cilindro) e R é o raio do cilindro.
==========================================================================
==========================================================================
Inputs:
- 'sobol_configs_artigo/configs_simulacao_reais_sobol.csv': Configurações de blindagem geradas via Sobol.
==========================================================================
==========================================================================
Outputs:
- 'mcnp_simulations_sobol/sim_XXXXX/input.i': Arquivos de input MCNP gerados.
- 'mcnp_simulations_sobol/sim_XXXXX/output.o': Arquivos de output MCNP gerados.
- 'mcnp_results_sobol/resultados_mcnp_sobol.csv': Resultados de dose efetiva (mSv).
==========================================================================
==========================================================================
MCNP_TEMPLATE = c (Células)
1 {mat_id_1} -{dens_1:.4f} 1 -2 4 -10 11 imp:p=1 $ 1a camada: {mat_name_1}
2 {mat_id_2} -{dens_2:.4f} 2 -3 4 -10 11 imp:p=1 $ 2a camada: {mat_name_2}
3 {mat_id_3} -{dens_3:.4f} 3 -5 4 -10 11 imp:p=1 $ 3a camada: {mat_name_3}
4 {mat_id_4} -{dens_4:.4f} 5 -6 4 -10 11 imp:p=1 $ 4a camada: {mat_name_4}
5 {mat_id_5} -{dens_5:.4f} 6 -7 4 -10 11 imp:p=1 $ 5a camada: {mat_name_5}
6 {mat_id_6} -{dens_6:.4f} 7 -8 4 -10 11 imp:p=1 $ 6a camada: {mat_name_6}
7 {mat_id_7} -{dens_7:.4f} 8 -9 4 -10 11 imp:p=1 $ 7a camada: {mat_name_7}
8 {mat_id_Co60} -{dens_Co60:.4f} -25 imp:p=1 $ Fonte volumétrica
20 {mat_id_Air} -{dens_Air:.4f} -21 imp:p=1 $ Detector
11 0 -30 #1 #2 #3 #4 #5 #6 #7 #8 #20 imp:p=1 $ Interior
30 0 30 imp:p=0 $ Graveyard

c (Superfícies)
1 CZ {BLINDAGEM_RAIO_INTERNO_CM:.1f}      $ Cilindro raio interno
2 CZ {r_ext_1:.3f}    $ Cilindro raio 1a camada
3 CZ {r_ext_2:.3f}    $ Cilindro raio 2a camada
4 PZ {BLINDAGEM_Z_BASE_CM:.1f}      $ Plano inferior (z base)
5 CZ {r_ext_3:.3f}    $ Cilindro raio 3a camada
6 CZ {r_ext_4:.3f}    $ Cilindro raio 4a camada
7 CZ {r_ext_5:.3f}    $ Cilindro raio 5a camada
8 CZ {r_ext_6:.3f}    $ Cilindro raio 6a camada
9 CZ {r_ext_7:.3f}    $ Cilindro raio 7a camada
10 PZ {BLINDAGEM_Z_TOPO_CM:.1f}    $ Plano superior (z topo)
11 PX 0        $ Plano cortante (x = 0)
21 RCC 0 0 120.0 0 0 10.0 5.0 $ Detector
25 RCC 121.0 0.0 124.0 0 0 2 1.0 $ Fonte
30 S 125 0 125 150 $ VOLUME DE CONTROLE

c (Materiais)
MODE P
{material_definitions}c (Tally e Fonte)
F4:P 20
FM4 {FATOR_CONVERSAO_FLUENCIA_DOSE} {FONTE_CO60_ATIVIDADE_BQ}
NPS 100000000 $ 10^8 Número de partículas
SDEF POS=121 0 125 AXS=0 0 1 EXT=D1 RAD=D2 ERG=D3 PAR=2
SI1 -1 1
SP1 0 1
SI2 0 1
SP2 -21 1
SI3 L 1.1732 1.3325
SP3 1 1

==========================================================================
==========================================================================
Equação da taxa de dose simplificada: 

Ď = ϕ × k × A; Taxa de dose efetiva = F4:P × k × A
Onde:
- Ď = Taxa de Dose efetiva [mSv/s]
- ϕ = Fluência [fótons/cm²/decaimento] (F4:P)
- k = Coeficiente de conversão fluência-dose efetiva [(mSv·cm²)/fóton] 
- A = Atividade [Bq]

# F4 = Tally de fluência médio de fótons na célula do detector (célula 20)
# FM4: Fator Multiplicador (para o Tally F4); Multiplica a fluência
[(fótons/cm²)/decaimento] pelo coeficiente de conversão de fluência
para dose efetiva [(mSv.cm^2)/fóton] e pela atividade da fonte (Bq = decaimento por segundo).
# Fonte Co-60: 2 partículas por história
# Com PAR=2 no SDEF, cada história representa um decaimento completo (2 fótons).
# O resultado no output do MCNP será a TAXA DE DOSE EFETIVA em mSv/segundo.
# SP1 1.0 1.0 + PAR=2 = MCNP gera 2 partículas por história com Probabilidade de
# 100% para cada energia
# Ď = ϕ × (∑ de i a n (ki × wi)) × A; onde:
# Ď = Taxa de dose (mSv/s)
# ϕ: fluência de fótons [fótons/(cm².história)]; fi grande é fluência;
# ki = coeficiente de conversão para a energia i [(mSv.cm²)/fóton]; FLUENCE_TO_EFFECTIVE_DOSE_COEFFICIENT_MSV_CM2
# wi = nº de fótons emitidos per decaimento na energia i
# A: Atividade da fonte em Bq (decaimentos/segundo).
==========================================================================
==========================================================================
Explicação matemática do uso do Tally F4:
 
Fluência média de fotons: [(fótons/cm^2)/decaimento]

Coeficiente de conversão: [(mSv.cm^2)/fóton]

Atividade: 3700 Gbq (#/segundo)


Resposta:


Taxa de dose efetiva = F4:P × k × A

Taxa de dose efetiva (mSv/s) = {[(fótons/cm^2)/decaimento] . [(mSv.cm^2)/fóton] . (decaimentos/s)} = mSv/s   --> dividindo por fótons no numerador e denominador

Taxa de dose efetiva (mSv/s) = {[(1/cm^2)/decaimento da fonte] . [(mSv.cm^2)/1] . (decaimentos/s)} = mSv/s  --> dividindo por decaimento da fonte no numerador e no denominador

Taxa de dose efetiva (mSv/s) = {(1/cm^2)/1] . [(mSv.cm^2)/1] . (1/s)} = mSv/s  --> dividindo por cm^2 no numerador e no denominador

Taxa de dose efetiva (mSv/s) = {(1) . (mSv) . (1/s)} = mSv/s --> Output do MCNP (Ok !)



Etapa entermediária entre a fase 2 e 3: multiplicação do output do MCNP pelo tempo (em segundos) de exposição. 
==========================================================================
Input para cálculo de D0 (Dose efetiva sem blindagem):

c (Células)
1 0 1 -2 4 -10 11 imp:p=1 $ 1a camada: Vácuo
2 0 2 -3 4 -10 11 imp:p=1 $ 2a camada: Vácuo
3 0 3 -5 4 -10 11 imp:p=1 $ 3a camada: Vácuo
4 0 5 -6 4 -10 11 imp:p=1 $ 4a camada: Vácuo
5 0 6 -7 4 -10 11 imp:p=1 $ 5a camada: Vácuo
6 0 7 -8 4 -10 11 imp:p=1 $ 6a camada: Vácuo
7 0 8 -9 4 -10 11 imp:p=1 $ 7a camada: Vácuo
8 1 -8.9000 -25 imp:p=1 $ Fonte pontual
20 2 -0.0012 -21 imp:p=1 $ Detector
11 0 -30 #1 #2 #3 #4 #5 #6 #7 #8 #20 imp:p=1 $ Interior
30 0 30 imp:p=0 $ Graveyard

c (Superfícies)
1 CZ 35.0      $ Cilindro raio interno
2 CZ 35.274    $ Cilindro raio 1a camada
3 CZ 36.085    $ Cilindro raio 2a camada
4 PZ 90.0      $ Plano inferior (z base)
5 CZ 37.898    $ Cilindro raio 3a camada
6 CZ 38.341    $ Cilindro raio 4a camada
7 CZ 38.773    $ Cilindro raio 5a camada
8 CZ 39.886    $ Cilindro raio 6a camada
9 CZ 41.000    $ Cilindro raio 7a camada
10 PZ 160.0    $ Plano superior (z topo)
11 PX 0         $ Plano cortante (x = 0)
15 S 241.0 10.0 125.0 4.5 $ Esfera externa
16 S 241.0 10.0 125.0 3.5 $ Esfera interna
21 RCC 0 0 120.0 0 0 10.0 5.0 $ Detector
25 RCC 241.0 0.0 124.0 0 0 2 1.0 $ Fonte
30 S 125 0 125 150 $ VOLUME DE CONTROLE

c (Materiais)
MODE P
M1 27060 1.000000 $ Co60_Source
M2 7000 0.780840 $ Air
      8000 0.209460
      18000 0.009340
c (Tally e Fonte)
F4:P 20
FM4 39442 $ 5.33e-09 . 3.7e12 . 2
NPS 100000000 $ 10^8 Número de partículas
SDEF POS=241 0 125 AXS=0 0 1 EXT=D1 RAD=D2 ERG=D3 PAR=2
SI1 -1 1
SP1 0 1
SI2 0 1
SP2 -21 1
SI3 L 1.1732 1.3325
SP3 1 1
==========================================================================
Cálculo da Taxa de Dose Inicial (D0)

O D0 em mSv/h, para uma fonte pontual, é dada por: D0 = (S⋅Γ)/(r^2)​

Onde:
D0 = Taxa de Dose Inicial (mSv/h)
S = Atividade da Fonte (GBq)
Γ = Constante Gama de Taxa de Dose do radionuclídeo (mSv⋅m^2)/(GBq⋅h)
r = Distância da fonte ao ponto de interesse (m)

Dados de projeto: 
    
Atividade da fonte: 3700 GBq
Distância: 2,41m
Constante Gama (Co-60): 0.351 (mSv⋅m^2)/(GBq⋅h); Fonte: "https://en.wikipedia.org/wiki/Cobalt-60";

D0 = (3700.0,351)/(2,41)^2 = 223.60mSv/h = 3,7267 mSv/minuto = 0,0621 mSv/s
D0 = (3700.0,3125)/(2,41)^2 = 199,07mSv/h = 3,318 mSv/minuto = 0,0553 mSv/s
D0 (MCNP) = 191,88mSv/h = 3,198 mSv/minuto = 0,0533 mSv/s

Observação: Historicamente, essa a constante Gama é derivada de valores como a Constante de Taxa de Exposição (Gamão x), que é geralmente citada em (R.m^2)/(Ci.h).
A conversão mais comum é: Gamão (Co-60) = 1,32 R.m^2/(Ci.h), cuja fonte é "https://www.nrc.gov/docs/ML1122/ML11229A688.pdf";
Cálculo de conversão de 1,32 R.m^2/(Ci.h) em 0.351 (mSv⋅m^2)/(GBq⋅h):

Fatores: 1 R ≈ 10 mSv (Dose Efetiva) | 1 Ci = 37 GBq (Atividade)

Gamma_SI = 1.32 * (10 mSv / 1 R) * (1 Ci / 37 GBq) = 1.32 * 10 / 37 = 0.35675... (Arredondado para 0.351 no consenso regulatório, um fator rápido e conservador)
Gamma_SI = 1.32 * (8.76 mSv / 1 R) * (1 Ci / 37 GBq) = 1.32 * 8.76 / 37 = 0.3125... (Arredondado para 0.3125)    
Usando o gamão (Attix, 1986) de 12,97 (R.cm2)/mCi.h, temos: 0.3075 (mSv.m2)/(GBq.h)
==========================================================================
Sintaxe da Fonte pontual:

c SDEF POS=241.0 0.0 125.0 ERG=D1 PAR=2
c SI1 L 1.1732 1.3325
c SP1 1.0 1.0

Explicação detalhada da sintaxe da Fonte Volumétrica:

POS=241.0 0.0 125.0: ponto central do cilindro
AXS=0 0 1: Vetor direção do eixo do cilindro (direção Z)
EXT=D1: Distribuição da EXTensão (altura) ao longo do eixo, ele controla como as partículas são distribuídas na altura; D1: Refere-se às definições SI1/SP1
RAD=D2: Distribuição do RAio (distância do eixo), ele controla como as partículas são distribuídas radialmente; D2: Refere-se às definições SI2/SP2.
ERG=D3: Distribuição da Energia, ele representa o espectro energético do Co-60;
D3: Refere-se às definições SI3/SP3.
PAR=2: Refere-se ao Tipo de PARtícula; 2 = Fótons.
SI1 -1 1: Source Information para distribuição 1 (EXT), refere-se a
a altura que é amostrada uniformemente de -1 a +1 cm do centro; -1 a 1 pois o centro está Z=125.0
SP1 0 1: Source Probability para distribuição 1, refere-se a distribuição uniforme
ao longo da altura; 1 significa a probabilidade relativa no ponto inicial (-1 cm) e
1 se refere a probabilidade relativa no ponto final (+1 cm), lembrando que a
distribuição deve ser uniforme entre eles; MCNP não aceitou 1 no lugar do 0.
SI2 0 1: Source Information para distribuição 2 (RAD), raio varia de 0 (centro) a 1.0 cm (borda).
SP2 -21 1: Source Probability para distribuição 2, refere-se a distribuição radial uniforme no disco.
Use a distribuição correta para cilindros, onde a probabilidade é proporcional ao raio, garantindo distribuição uniforme em TODO O VOLUME; -2: Distribuição proporcional a r¹ (linear) e 1: Expoente = 1 (r¹):

Código	Significado	        Uso
-21	Distribuição (probabilidade) ∝ r	Cilindros
0	Distribuição uniforme	Linhas, eixos
-22	Distribuição ∝ r²	Esferas

SI3 L 1.1732 1.3325: Source Information para distribuição 3 (ERG), refere-se a lista das energias do Co-60.
SP3 1 1: Source Probability para distribuição 3, referem-se as probabilidades iguais para ambas
energias do Co-60 (50% cada).
==========================================================================
Sobre o FM4 (página 3-94 do volume 2 do manual MCNP): "If neither a
material nor any reactions are given, the tally is multiplied by the constant C."
Testar ainda...
==========================================================================
Validação do FM4:

Análise da Constante Gama (Γ) e Coerência

1 R≈10 mSv (Fator conservador)	        0.3567 (Arredondado para 0.351)	 0.0621
1 R≈8.76 mSv (Fator Kerma/Exposição)	0.3125	                         0.0553

MCNP: 0.0533 mSv/s
Analítico 1 (Γ=0.3125): 0.0553 mSv/s (Diferença de ≈3.7%)
Analítico 2 (Γ=0.351): 0.0621 mSv/s (Diferença de ≈16.5%)


# DADOS GERAIS DO PROJETO
# ----------------------
ATIVIDADE_A_GBQ = 3700.0  # Atividade da Fonte (GBq)
ATIVIDADE_A_BQ = 3.7e12   # Atividade da Fonte (Bq = particulas/s)
DISTANCIA_R_M = 2.41      # Distancia Fonte-Detector (m)

# FATOR DE CONVERSAO FLUENCIA-DOSE (ICRP 116, AP, Co-60 Medio)
# -----------------------------------------------------------
# DC_medio = Coeficiente de Dose Efetiva Medio (1.1732 MeV e 1.3325 MeV)
# Valor rigorosamente interpolado e simplificado.
DC_MEDIO_SV_CM2_FOTON = 5.33e-12  # (Sv * cm^2) / foton

# FATOR DE EMISSAO DO Co-60
# ------------------------
# O Co-60 emite ~2 fotons gama per decaimento (particula de fonte).
# Como o SDEF simula 1 foton por decaimento, usamos FATOR_EMISSAO = 2 no FM4.
FATOR_EMISSAO = 2.0  # fotons / decaimento simulado
FATOR_MSV_SV = 1000.0 # mSv / Sv

# ==============================================================================
# PARTE 1: CALCULO ANALITICO DA DOSE INICIAL (D0)
# ==============================================================================

# Objetivo: Encontrar o D0 analitico mais coerente com a metodologia ICRP (MCNP).
# A constante Gama (Gamma) e sensível a conversão de R (Exposição) para mSv (Dose Efetiva).

# 1.1. CALCULO CONSERVADOR (Gamma = 0.351)
# ----------------------------------------
# Conversão: 1 R ~ 10 mSv (Fator conservador de Proteção Radiológica)
GAMMA_CONSERVADOR = 0.351 # (mSv * m^2) / (GBq * h)
D0_H_CONSERVADOR = (ATIVIDADE_A_GBQ * GAMMA_CONSERVADOR) / (DISTANCIA_R_M**2)
D0_S_CONSERVADOR = D0_H_CONSERVADOR / 3600.0

# D0 Analitico (Conservador):
# D0 = (3700 * 0.351) / (2.41)^2 = 223.60 mSv/h
# D0_S = 223.60 / 3600 = 0.06211 mSv/s
# ------------------------------------------------------------------------------

# 1.2. CALCULO MAIS RIGOROSO/COERENTE (Gamma = 0.3125)
# ----------------------------------------------------
# Conversão: 1 R ~ 8.76 mSv (Fator mais próximo de Kerma/Exposição)
# Este Gamma (0.3125) e o mais coerente com a metodologia de Dose Efetiva ICRP.
GAMMA_RIGOROSO = 0.3125 # (mSv * m^2) / (GBq * h)
D0_H_RIGOROSO = (ATIVIDADE_A_GBQ * GAMMA_RIGOROSO) / (DISTANCIA_R_M**2)
D0_S_RIGOROSO = D0_H_RIGOROSO / 3600.0

# D0 Analitico (Mais Coerente):
# D0 = (3700 * 0.3125) / (2.41)^2 = 199.07 mSv/h
# D0_S = 199.07 / 3600 = 0.05530 mSv/s
# ------------------------------------------------------------------------------
# CONCLUSÃO: O D0 analitico de 0.0553 mSv/s e o valor de referência a ser buscado,
# pois e mais coerente com a metodologia ICRP detalhada.
# ==============================================================================


# ==============================================================================
# PARTE 2: AJUSTE DO FATOR DE MULTIPLICACAO (FM4) PARA MCNP
# ==============================================================================

# O FM4 deve converter o resultado do F4 [fotons/cm^2 / decaimento_simulado]
# para Taxa de Dose Efetiva [mSv/s].

# FM4 = DC_medio * Fator_Emissao * Atividade_Bq * Fator_mSv/Sv

# Calculo detalhado do FM4:
FM4_FACTOR = DC_MEDIO_SV_CM2_FOTON * FATOR_EMISSAO * ATIVIDADE_A_BQ * FATOR_MSV_SV
# FM4 = (5.33e-12) * 2 * (3.7e12) * 1000
# FM4 = 39.442 * 1000
# FM4 = 39442.0

FM4_RIGOROSO = 39442.0

# Unidade do FM4: (mSv * cm^2 * decaimento_simulado) / (s * foton)
# ------------------------------------------------------------------------------

# 2.1. PREVISÃO DO RESULTADO MCNP
# -------------------------------
# Usando o F4 de um run anterior (1.3509e-06) e o FM4 rigoroso:
F4_RESULTADO_MCNP = 1.3509e-06 # Fluência / Decaimento Simulado
DOSE_MCNP_S = F4_RESULTADO_MCNP * FM4_RIGOROSO

# DOSE_MCNP = 1.37158e-06 * 39442.0 = 0.0540977 mSv/s
# ------------------------------------------------------------------------------
# CONCLUSÃO FINAL:
# A Taxa de Dose MCNP (0.0540977 mSv/s) tem uma pequena e aceitável diferença (~2.22%)
# em relação ao D0 analitico mais coerente (0.0553 mSv/s), confirmando a coerência
# da metodologia Monte Carlo com a ICRP para a geometria especifica.

# ==============================================================================
# INPUT FINAL PARA O MCNP
# ==============================================================================
# O MCNP deverá ter o seguinte cartão:
# F4:P 20
# FM4 39442.0 $ Fator de conversão para Dose Efetiva [mSv/s] (ICRP 116 DC * Atividade * Fator 2)
# ==============================================================================

Trecho do input do FM4 deve ser:
   
F4:P 20
FM4 39442
==========================================================================
Coeficiente de conversão fluência-dose efetiva: Na pagina 126 da ICRP 116 (2010), a tabela A.1 apresenta os valores
para posição ap (antero-posterior) de 4,9 e 5,59 pSv.cm^2, respectivamente, para 1,117 Mev e para 1,33 mev. Por interpolação,
encontrei 5,082 pSv.cm^2 para a energia 1,1732 mev; portanto, para 1,25 MeV é 5,33 (pSv.cm^2)/partícula = 5.33e-09 (mSv.cm^2)/partícula;
==========================================================================
📊 O QUE MUDA:
✅ Paralelismo Python: 4 processos simultâneos

✅ Paralelismo MCNP: Cada processo usa tasks 6

✅ Performance individual: Cada input ~1-2 minutos (75% mais rápido)

✅ Performance total: 4 inputs a cada ~2 minutos

⚡ NOVA ESTIMATIVA:
1 input: ~2 minutos (com tasks 6)

4 inputs simultâneos: ~2 minutos

2000 inputs: 2000/4 = 500 lotes × 2min = 1000 minutos = ~17 horas

⚠️ ALERTA DE SOBRECARGA:
4 processos × 6 tasks = 24 "threads" concorrendo

PC tem: 2 núcleos físicos + 4 lógicos

Pode: Travar o sistema ou ficar mais lento por contenção
==========================================================================
==========================================================================
"""

# Importação de bibliotecas necessárias
import hashlib
import pandas as pd  # Para manipulação de dados tabulares (CSV)
import numpy as np   # Para cálculos numéricos
import os            # Para operações com arquivos e diretórios
print(f"Diretório atual: {os.getcwd()}")
import subprocess    # Para executar comandos externos (MCNP)
import re            # Para expressões regulares (extração de dados do output)
import sys           # Para sair do script em caso de erro
import time          # Para medir o tempo de execução
from concurrent.futures import ThreadPoolExecutor, as_completed  # Para paralelismo
import json          # Para registro de logs em formato JSON
import logging       # Para registro de atividades
from datetime import datetime  # Para marcar horários
from pathlib import Path

# ==============================================================================
# --- CONFIGURAÇÕES ESPECÍFICAS PARA EXECUTÁVEL PYINSTALLER ---
# ==============================================================================

def get_base_path():
    """Determina o diretório base corretamente para PyInstaller"""
    if getattr(sys, 'frozen', False):
        # Executando como .exe - usar diretório do executável
        return Path(sys.executable).parent
    else:
        # Executando como script - usar diretório atual
        return Path(__file__).parent


# Configurar diretório base
BASE_DIR = get_base_path()

# ==============================================================================
# --- (1) CONFIGURAÇÕES DO USUÁRIO (AJUSTES AQUI!) ---
# ==============================================================================

# Sequência Sobol original (Fase 1): nunca regenerar esta tabela durante a
# campanha MCNP, pois ela define os mesmos 2.000 casos do estudo.
INPUT_CONFIGS_CSV = (
    BASE_DIR.parent / 'Fase 1' / 'simulacao_configs (Fase 1) - Sobol'
    / 'configs_simulacao_reais_sobol.csv'
)

# Campanha final DE/DF AP: manter isolada dos resultados escalares históricos.
MCNP_SIM_DIR = BASE_DIR / 'mcnp_simulations_sobol_dedf_ap'
MCNP_RESULTS_DIR = BASE_DIR / 'mcnp_results_sobol_dedf_ap'
MCNP_RESULTS_FILE = MCNP_RESULTS_DIR / 'resultados_mcnp_sobol_dedf_ap.csv'

# Executável verificado no PC institucional (MCNP 6.3.0).
MCNP_EXECUTABLE = Path(r"C:\mcnp633\mcnp-6.3.0-Windows\bin\mcnp6.exe")

# Configurações geométricas fixas
BLINDAGEM_RAIO_INTERNO_CM = 35.0         # Distância do operador à blindagem
BLINDAGEM_ALTURA_CM = 70.0               # Altura total da blindagem
BLINDAGEM_Z_BASE_CM = 90.0               # Posição Z da base da blindagem
BLINDAGEM_Z_TOPO_CM = BLINDAGEM_Z_BASE_CM + BLINDAGEM_ALTURA_CM  # Adicione esta linha

# Configurações do detector do operador
DETECTOR_CELULA_ID = 20                  # ID da célula do detector no MCNP
DETECTOR_RAIO_CM = 5.0                   # Raio do cilindro do detector
DETECTOR_ALTURA_CM = 10.0                # Altura do cilindro do detector
DETECTOR_Z_BASE_CM = BLINDAGEM_Z_BASE_CM + (BLINDAGEM_ALTURA_CM / 2.0) - (DETECTOR_ALTURA_CM / 2.0)

# Parâmetros da fonte e resposta fluência-dose efetiva.
FONTE_CO60_ATIVIDADE_BQ = 3.7e12
FOTONS_POR_DECAIMENTO_CO60 = 2.0
NPS_POR_SIMULACAO = 100_000_000
# DF4 está em pSv cm²; FM4 converte a resposta por fóton para mSv/s.
FATOR_FONTE_PARA_MSV_S = FONTE_CO60_ATIVIDADE_BQ * FOTONS_POR_DECAIMENTO_CO60 * 1e-9
TEMPO_EXPOSICAO_PARA_ANALISE_SEGUNDOS = 300  # 5 minutos para análise
# Sementes ímpares, distintas e registradas: reproduzem cada caso e permitem
# reexecuções estatisticamente independentes quando necessário.
RAND_SEED_BASE = 19073486328125
RAND_SEED_STEP = 2

# PC institucional: 24 núcleos físicos. Três processos MCNP com oito tasks
# cada ocupam 24 núcleos sem ultrapassar a capacidade medida da máquina.
MAX_PARALLEL_SIMULATIONS = int(os.environ.get('SPHIN_WORKERS', '3'))
TASKS_POR_SIMULACAO = int(os.environ.get('SPHIN_TASKS', '8'))
if MAX_PARALLEL_SIMULATIONS < 1 or TASKS_POR_SIMULACAO < 1:
    raise ValueError('SPHIN_WORKERS e SPHIN_TASKS devem ser inteiros positivos.')

# ==============================================================================
# 4. CONFIGURAÇÃO DO LOGGING (APÓS DEFINIR MCNP_SIM_DIR)
# ==============================================================================
# Criar diretório antes do logging
os.makedirs(MCNP_SIM_DIR, exist_ok=True)

# Configurar logging
LOG_FILE = MCNP_SIM_DIR / f'simulation_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format='%(message)s',
    encoding='utf-8'
)

# ==============================================================================
# --- (2) DEFINIÇÃO DOS MATERIAIS PARA MCNP ---
# ==============================================================================

# Dicionário completo de materiais com composição atômica e densidades
MCNP_MATERIALS_DATA = {
    'SiC': {'densidade': 3.2100, 'composicao': [(6000, 0.300200), (14000, 0.699800)]},
    'Al': {'densidade': 2.7000, 'composicao': [(13000, 1.000000)]},
    'Fe': {'densidade': 7.8700, 'composicao': [(26000, 1.000000)]},
    'Bi2O3-HDPE': {'densidade': 2.5570, 'composicao': [(83000, 0.627900), (8000, 0.072100), (6000, 0.256900), (1001, 0.043100)]},
    'Co60_Source': {'densidade': 8.9000, 'composicao': [(27060, 1.000000)]}, # Cobalto-60 puro
    'Air': {'densidade': 0.0012, 'composicao': [(7000, 0.780840), (8000, 0.209460), (18000, 0.009340)]}, # Ar
    'W-Epoxy': {
        'densidade': 4.643,
        'composicao': [(74000, 0.800000), (8000, 0.040000), (6000, 0.140000), (1001, 0.020000)]
    },
    'B4C-HDPE': {
        'densidade': 1.428,
        'composicao': [(5000, 0.414500), (6000, 0.517900), (1001, 0.067600)]
    },
    'UHMWPE': {
        'densidade': 0.94,
        'composicao': [(6000, 0.856280), (1001, 0.143720)]
    }
}

# Mapeamento de IDs de materiais para o MCNP (Renumerados sequencialmente)
MCNP_MATERIAL_IDS = {
    # Materiais das camadas (1-7)
    'SiC': 1,
    'Al': 2,
    'Fe': 3,
    'Bi2O3-HDPE': 4,
    'Co60_Source': 5, # Co60_Source agora é material 5
    'Air': 8, # Air agora é material 8
    'W-Epoxy': 9,
    'B4C-HDPE': 10,
    'UHMWPE': 11,
}

# ICRP Publication 116, Table A.1, corrigenda 2015: photons, AP, pSv cm².
ICRP116_AP_ENERGIES_MEV = [
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.15,
    0.2, 0.3, 0.4, 0.5, 0.511, 0.6, 0.662, 0.8, 1.0, 1.117, 1.33, 1.5,
    2.0, 3.0, 4.0, 5.0, 6.0, 6.129, 8.0, 10.0, 15.0, 20.0, 30.0, 40.0,
    50.0, 60.0, 80.0, 100.0, 150.0, 200.0, 300.0, 400.0, 500.0, 600.0,
    800.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0,
    8000.0, 10000.0,
]
ICRP116_AP_DF_PSV_CM2 = [
    0.0685, 0.156, 0.225, 0.312, 0.350, 0.369, 0.389, 0.411, 0.443, 0.518,
    0.747, 1.00, 1.51, 2.00, 2.47, 2.52, 2.91, 3.17, 3.73, 4.49, 4.90,
    5.60, 6.12, 7.48, 9.75, 11.7, 13.4, 15.0, 15.1, 17.8, 20.5, 26.1,
    30.8, 37.9, 43.2, 47.1, 50.1, 54.5, 57.8, 63.2, 67.2, 72.3, 75.4,
    77.4, 78.7, 80.4, 81.6, 83.7, 85.0, 86.6, 87.8, 88.6, 89.1, 89.9,
    90.4,
]

def format_response_card(card_name, values, values_per_line=7):
    """Formata DE/DF com continuações MCNP válidas e linhas curtas."""
    tokens = [f"{value:.6g}" for value in values]
    lines = []
    for start in range(0, len(tokens), values_per_line):
        prefix = card_name if start == 0 else "     "
        lines.append(prefix + " " + " ".join(tokens[start:start + values_per_line]))
    return "\n".join(lines)

DE4_ICRP116_AP = format_response_card("DE4", ICRP116_AP_ENERGIES_MEV)
DF4_ICRP116_AP = format_response_card("DF4", ICRP116_AP_DF_PSV_CM2)


# ==============================================================================
# --- (3) TEMPLATE DO ARQUIVO DE INPUT MCNP ---
# ==============================================================================

MCNP_TEMPLATE = """SPHIN DE/DF AP - simulacao Sobol {sim_id:04d}
c (Células)
1 {mat_id_1} -{dens_1:.4f} 1 -2 4 -10 11 imp:p=1 $ 1a camada: {mat_name_1}
2 {mat_id_2} -{dens_2:.4f} 2 -3 4 -10 11 imp:p=1 $ 2a camada: {mat_name_2}
3 {mat_id_3} -{dens_3:.4f} 3 -5 4 -10 11 imp:p=1 $ 3a camada: {mat_name_3}
4 {mat_id_4} -{dens_4:.4f} 5 -6 4 -10 11 imp:p=1 $ 4a camada: {mat_name_4}
5 {mat_id_5} -{dens_5:.4f} 6 -7 4 -10 11 imp:p=1 $ 5a camada: {mat_name_5}
6 {mat_id_6} -{dens_6:.4f} 7 -8 4 -10 11 imp:p=1 $ 6a camada: {mat_name_6}
7 {mat_id_7} -{dens_7:.4f} 8 -9 4 -10 11 imp:p=1 $ 7a camada: {mat_name_7}
8 {mat_id_Co60} -{dens_Co60:.4f} -25 imp:p=1 $ Fonte volumétrica
20 {mat_id_Air} -{dens_Air:.4f} -21 imp:p=1
11 0 -30 #1 #2 #3 #4 #5 #6 #7 #8 #20 imp:p=1 $ Interior
30 0 30 imp:p=0 $ Graveyard

c (Superfícies)
1  CZ {BLINDAGEM_RAIO_INTERNO_CM:.1f}      $ Cilindro raio interno
2  CZ {r_ext_1:.3f}    $ Cilindro raio 1a camada
3  CZ {r_ext_2:.3f}    $ Cilindro raio 2a camada
4  PZ {BLINDAGEM_Z_BASE_CM:.1f}      $ Plano inferior (z base)
5  CZ {r_ext_3:.3f}    $ Cilindro raio 3a camada
6  CZ {r_ext_4:.3f}    $ Cilindro raio 4a camada
7  CZ {r_ext_5:.3f}    $ Cilindro raio 5a camada
8  CZ {r_ext_6:.3f}    $ Cilindro raio 6a camada
9  CZ {r_ext_7:.3f}    $ Cilindro raio 7a camada
10 PZ {BLINDAGEM_Z_TOPO_CM:.1f}     $ Plano superior (z topo)
11 PX 0         $ Plano cortante (x = 0)
21 RCC 0 0 120.0 0 0 10.0 5.0 $ Detector
25 RCC 121.0 0.0 124.0 0 0 2 1.0 $ Fonte
30 S 125 0 125 150 $ VOLUME DE CONTROLE

c (Materiais)
MODE P
{material_definitions}c (Tally e Fonte)
F4:P 20
{DE4_CARD}
{DF4_CARD}
c DF4 em pSv cm2; FM4 = atividade * 2 fotons/decaimento * 1e-9 mSv/pSv
FM4 {FATOR_FONTE_PARA_MSV_S:.0f}
NPS {NPS_POR_SIMULACAO} $ 10^8 historias
RAND GEN=1 SEED={RAND_SEED}
SDEF POS=121 0 125 AXS=0 0 1 EXT=D1 RAD=D2 ERG=D3 PAR=2
SI1 -1 1
SP1 0 1
SI2 0 1
SP2 -21 1
SI3 L 1.1732 1.3325
SP3 1 1"""

# ==============================================================================
# --- (4) FUNÇÕES AUXILIARES ---
# ==============================================================================

# --- 1. Definição da função format_material_mcnp ---
def format_material_mcnp(mat_id, densidade, composicao, material_name=None):
    """
    Formata uma definição de material para MCNP no formato 'M# ZAI frac'.
    A densidade NÃO é incluída neste card, é definida na célula.
    Cada par ZAI/fração mássica é colocado em uma nova linha, com 6 espaços de indentação.
    :param mat_id: ID do material (int)
    :param densidade: Densidade do material (float) - Não usada diretamente no retorno para o M# card
    :param composicao: Lista de tuplas (ZAI, fraction)
    :param material_name: Nome do material para comentário (str, opcional)
    :return: String formatada para o MCNP
    """
    material_lines = []

    # Primeira linha do material: M# e o primeiro par ZAI frac (se houver)
    if composicao:
        first_zai, first_fraction = composicao[0]
        first_line_parts = [f"M{mat_id}", f"{first_zai} {first_fraction:.6f}"]
        comment = f" $ {material_name}" if material_name else ""
        material_lines.append(" ".join(first_line_parts) + comment)
        
        # Adiciona os pares ZAI frac restantes, cada um em sua própria linha, com 5 espaços de indentação
        for zai, fraction in composicao[1:]: # Começa do segundo elemento
            material_lines.append(f"      {zai} {fraction:.6f}") # 5 espaços de indentação
    else:
        # Se não houver composição, apenas M# e um comentário (raro, mas possível)
        comment = f" $ {material_name}" if material_name else ""
        material_lines.append(f"M{mat_id}" + comment)

    return "\n".join(material_lines)

# --- 2. Definição da função generate_all_materials ---
def generate_all_materials(materials_used):
    """
    Gera todos os materiais únicos necessários para a simulação.
    Args:
        materials_used (list): Lista de nomes de materiais usados nas camadas da blindagem.
    Returns:
        str: String com todas as definições de materiais formatadas para o MCNP.
    """
    material_definitions_list = []
    
    # 1. Processa os materiais das camadas (removendo duplicatas)
    # Garante a ordem dos materiais pelo ID para consistência
    all_unique_materials = sorted(list(set(materials_used) | set(['Co60_Source', 'Air'])), 
                                  key=lambda x: MCNP_MATERIAL_IDS[x])
    
    for mat_name in all_unique_materials:
        mat_id = MCNP_MATERIAL_IDS[mat_name]
        mat_data = MCNP_MATERIALS_DATA[mat_name]
        
        densidade = mat_data['densidade']
        composicao = mat_data['composicao']
        
        material_definitions_list.append(
            format_material_mcnp(mat_id, densidade, composicao, mat_name)
        )
            
    return "\n".join(material_definitions_list) + "\n" # Garante uma quebra de linha final


# --- 3. Definição da função clean_mcnp_line ---
def clean_mcnp_line(line):
    """
    Limpa e normaliza uma linha do input MCNP, preservando indentação para dados.
    """
    line = line.replace('\t', ' ')
    
    if line.strip().startswith('$'):
        return line.strip() + '\n'
        
    if line.strip().startswith('c'):
        return ' '.join(line.strip().split()) + '\n'
    
    parts = line.split('$')
    data_part = parts[0]
    comment_part = '$' + parts[1] if len(parts) > 1 else ''

    leading_spaces = len(data_part) - len(data_part.lstrip(' '))
    
    cleaned_data = ' '.join(data_part.strip().split())

    if cleaned_data:
        result_line = ' ' * leading_spaces + cleaned_data
        if comment_part:
            result_line += ' ' + comment_part.strip()
        return result_line.strip() + '\n'
    elif comment_part:
        return comment_part.strip() + '\n'
    else:
        return '\n'

def validate_geometry(thicknesses):
    """Verifica se as espessuras das camadas são válidas"""
    total = sum(thicknesses)
    if not (5.99 <= total <= 6.01):  # Margem pequena para erros de ponto flutuante
        raise ValueError(f"A espessura total deve ser 6cm (±0.01cm) (atual: {total:.2f}cm)")
    if any(t <= 0 for t in thicknesses):
        raise ValueError("Todas as espessuras devem ser maiores que zero")

def format_mcnp_input_for_vised(content):
    """
    Preserva cartões MCNP válidos para execução.

    Os cartões DE/DF já são gerados em linhas curtas. Não usar "&" para
    quebrar linhas, pois ele não é continuação válida de cartão MCNP.
    Args:
        content (str): Conteúdo completo do input MCNP
    Returns:
        str: Conteúdo formatado com quebras de linha conforme necessário
    """
    return content

def calculate_optimal_tasks():
    """
    Calcula número ótimo de tasks MCNP considerando overhead do sistema
    - CORREÇÃO: Limitar tasks para evitar 100% CPU e resultados inválidos
    """
    cpu_count = os.cpu_count()
    print(f"🔍 CPU detectada: {cpu_count} núcleos físicos")
    
    optimal_tasks = TASKS_POR_SIMULACAO
    
    print(f"🔧 Configuração: {MAX_PARALLEL_SIMULATIONS} processos × {optimal_tasks} tasks")
    print(f"🔧 Alocação: {optimal_tasks * MAX_PARALLEL_SIMULATIONS} threads MCNP")
    if optimal_tasks * MAX_PARALLEL_SIMULATIONS > cpu_count:
        print("⚠️  AVISO: a configuração excede os núcleos detectados.")
    
    return optimal_tasks

def run_mcnp_simulation_sync(input_file_path, output_dir, sim_id, mcnp_exec=MCNP_EXECUTABLE):
    """
    Executa uma simulação MCNP de forma SÍNCRONA (corrigido para paralelismo)
    Args:
        input_file_path (str): Caminho completo do arquivo de input
        output_dir (str): Diretório para salvar os outputs
        sim_id (int): ID da simulação para nomear arquivos
        mcnp_exec (str): Caminho para o executável do MCNP (com valor padrão)
    Returns:
        dict: Resultado da execução com status e caminhos
    """
    
    output_file_name = f"output_{sim_id}.o"
    runtpe_file_name = f"runtpe_{sim_id}.r"
    output_file_path = os.path.join(output_dir, output_file_name)
    runtpe_file_path = os.path.join(output_dir, runtpe_file_name)

    # ✅ CALCULAR TASKS DINAMICAMENTE
    optimal_tasks = calculate_optimal_tasks()
    print(f"🔧 Sim {sim_id}: Usando {optimal_tasks} tasks MCNP (CPU: {os.cpu_count()}, Paralelismo: {MAX_PARALLEL_SIMULATIONS})")
    
    # O MCNP recebe nomes locais porque cwd já é a pasta da simulação. Isso
    # evita que espaços em "Área de Trabalho" sejam reinterpretados pelo MCNP.
    command = [str(mcnp_exec), "i=input.i", f"o={output_file_name}",
               f"r={runtpe_file_name}", "tasks", str(optimal_tasks)]
    
    try:
        execution_start = time.perf_counter()
        # ✅ MUDANÇA CRÍTICA: subprocess.run() em vez de Popen()
        # Isso faz a execução ser SÍNCRONA e controlada
        result = subprocess.run(
            command, 
            cwd=output_dir, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            timeout=7200  # Timeout de 2 horas por segurança
        )
        
        execution_seconds = time.perf_counter() - execution_start
        output_created = os.path.exists(output_file_path)
        return {
            'success': result.returncode == 0 and output_created,
            'process': None,  # Não retornamos mais o processo vivo
            'output_file': output_file_path,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
            'tempo_parede_segundos': execution_seconds,
            'tasks_mcnp': optimal_tasks,
            'error': None if result.returncode == 0 and output_created else (
                f"MCNP returncode={result.returncode}; OUTP criado={output_created}; "
                f"stdout={result.stdout[-1000:]}; stderr={result.stderr[-1000:]}"
            )
        }
        
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'process': None,
            'output_file': output_file_path,
            'error': 'Timeout (2 horas) excedido'
        }
    except Exception as e:
        return {
            'success': False,
            'process': None, 
            'output_file': output_file_path,
            'error': str(e)
        }

def extract_mcnp_tally_result(output_file_path, tally_id):
    """
    Extrai resultados de tally do output do MCNP - CORRIGIDA
    Args:
        output_file_path (str): Caminho do arquivo de output
        tally_id (int): Número do tally a ser extraído (ex: 4)
    Returns:
        tuple: (valores, erros, células) ou (None, None, None) em caso de erro
    """
    if not output_file_path or not os.path.exists(output_file_path):
        print(f"Arquivo não encontrado: {output_file_path}")
        return None, None, None
    
    try:
        with open(output_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Formato do OUTP do MCNP: "cell  20" em uma linha e valor/erro na seguinte.
        # A ancoragem evita confundir este resultado com números de outra seção.
        number = r'[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?'
        pattern = re.compile(
            rf'^\s*cell\s+{tally_id}\s*\r?\n\s*({number})\s+({number})\s*$',
            re.IGNORECASE | re.MULTILINE
        )
        match = pattern.search(content)
        if match:
            value = float(match.group(1))
            error = float(match.group(2))
            print(f"✅ Tally {tally_id} extraído: {value:.2e} ± {error:.2e}")
            return [value], [error], [tally_id]
        
        # ❌ Se nenhum padrão funcionou
        print(f"❌ Tally {tally_id} não encontrado com os padrões atuais")
        
        # DEBUG: Encontrar a seção do tally manualmente
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if f'tally {tally_id}' in line.lower() and 'cell' in line.lower():
                print(f"🔍 Linha do tally encontrada: {line}")
                # Verificar próximas linhas
                for j in range(i+1, min(i+5, len(lines))):
                    print(f"🔍 Próxima linha {j}: {lines[j]}")
        
        return None, None, None
            
    except Exception as e:
        print(f"❌ Erro durante extração do tally {tally_id}: {str(e)}")
        return None, None, None

def extract_tally_fallback(output_file_path, tally_id):
    """
    Método de fallback para extrair tally - busca por padrões mais amplos
    - CORREÇÃO: Adicionado tratamento de erro mais robusto
    """
    try:
        with open(output_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Não há fallback numérico genérico: ele poderia registrar como dose um
        # número de uma seção sem relação com o tally 4. Uma extração ambígua é
        # reportada como erro para inspeção, nunca aceita silenciosamente.
        print(f"❌ Não foi possível confirmar o tally {tally_id} no formato do OUTP")
        return None, None, None
        
    except Exception as e:
        print(f"❌ Erro no fallback: {e}")
        return None, None, None

def extract_mcnp_quality(output_file_path):
    """Extrai evidências mínimas de qualidade diretamente do OUTP do MCNP."""
    quality = {
        'nps_tally4': None,
        'terminou_nps_alvo': False,
        'tfc_linha': None,
        'tfc_todos_aprovados': False,
    }
    try:
        with open(output_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        nps_match = re.search(r'1tally\s+4\s+nps\s*=\s*(\d+)', content, re.IGNORECASE)
        if nps_match:
            quality['nps_tally4'] = int(nps_match.group(1))

        target_match = re.search(
            rf'run terminated when\s+{NPS_POR_SIMULACAO}\s+particle histories were done',
            content, re.IGNORECASE
        )
        quality['terminou_nps_alvo'] = target_match is not None

        tfc_match = re.search(r'passed\?\s+((?:yes|no)(?:\s+(?:yes|no)){9})', content, re.IGNORECASE)
        if tfc_match:
            tfc_values = tfc_match.group(1).lower().split()
            quality['tfc_linha'] = ' '.join(tfc_values)
            quality['tfc_todos_aprovados'] = len(tfc_values) == 10 and all(v == 'yes' for v in tfc_values)
    except Exception as e:
        print(f"⚠️ Não foi possível extrair NPS/TFC: {e}")
    return quality

# ==============================================================================
# --- (4.5) SISTEMA DE MONITORAMENTO EM TEMPO REAL ---
# ==============================================================================

def monitorar_progresso(results_dir, total_simulacoes, intervalo_minutos=5):
    """Monitora o progresso das simulações automaticamente"""
    import time
    from threading import Thread
    
    def monitor():
        while True:
            try:
                # Verificar se arquivo de resultados existe
                resultados_file = MCNP_RESULTS_FILE
                if resultados_file.exists():
                    df = pd.read_csv(resultados_file)
                    completadas = len(df[df['status_execucao_mcnp'] == 'sucesso'])
                    em_andamento = len(df[df['status_execucao_mcnp'] == 'pendente'])
                    com_erro = len(df[df['status_execucao_mcnp'] == 'erro'])
                    
                    print(f"\n📊 [MONITOR] {completadas}/{total_simulacoes} completadas | "
                          f"Em andamento: {em_andamento} | Erros: {com_erro}")
                    
                    # Estimar tempo restante (base: 2 minutos por simulação)
                    if completadas > 0:
                        tempo_medio = 2  # minutos
                        restantes = total_simulacoes - completadas
                        tempo_restante = (restantes * tempo_medio) / MAX_PARALLEL_SIMULATIONS
                        horas = int(tempo_restante / 60)
                        minutos = int(tempo_restante % 60)
                        print(f"⏱️  [ESTIMATIVA] Tempo restante: ~{horas}h{minutos:02d}m")
                    
                    # Se todas completadas, parar monitoramento
                    if completadas + com_erro >= total_simulacoes:
                        print("🎉 [MONITOR] Todas as simulações concluídas!")
                        break
                
                time.sleep(intervalo_minutos * 60)  # Esperar X minutos
                
            except Exception as e:
                print(f"⚠️  [MONITOR] Erro: {e}")
                time.sleep(60)  # Esperar 1 minuto em caso de erro
    
    # Iniciar monitoramento em thread separada
    monitor_thread = Thread(target=monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread

# ==============================================================================
# --- FUNÇÃO GLOBAL PARA PARALELISMO ---
# ==============================================================================

def executar_simulacao_wrapper(sim_info):
    """
    Função GLOBAL para executar simulação MCNP (deve estar fora de main() para pickle)
    Args:
        sim_info: Dicionário com informações da simulação
    Returns:
        dict: Resultado da execução
    """
    return run_mcnp_simulation_sync(
        sim_info['input_file'], 
        sim_info['sim_dir'], 
        sim_info['sim_id'],
        MCNP_EXECUTABLE  # ✅ Passar como argumento
    )

def salvar_resultados_parciais(results, results_file):
    """
    Salva resultados parcialmente para recuperação em caso de interrupção
    """
    try:
        df_temp = pd.DataFrame(results)
        # Garante que o diretório existe
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        df_temp.to_csv(results_file, index=False)
        print(f"💾 Resultados parciais salvos: {len(results)} simulações")
    except Exception as e:
        print(f"⚠️ Erro ao salvar resultados parciais: {e}")

# ==============================================================================
# --- (5) LÓGICA PRINCIPAL / LOOP DE GERAÇÃO E EXECUÇÃO ---
# ==============================================================================

def main():
    """
    Função principal para orquestrar a geração de inputs, execução do MCNP
    e extração de resultados.
    """
    
    # Verificar se está no PC correto
    print(f"Diretório base: {BASE_DIR}")
    print(f"Executável MCNP: {MCNP_EXECUTABLE}")
    
    if not MCNP_EXECUTABLE.exists():
        print(f"❌ ERRO: MCNP não encontrado em {MCNP_EXECUTABLE}")
        print("Verifique a instalação do MCNP no PC destino")
        input("Pressione Enter para sair...")
        return
    
    # Verificar arquivo de configurações (AJUSTADO PARA SOBOL)
    if not INPUT_CONFIGS_CSV.exists():
        print(f"❌ ERRO: CSV não encontrado em {INPUT_CONFIGS_CSV}")
        print("Copie o arquivo 'configs_simulacao_reais_sobol.csv' para a pasta correta")
        input("Pressione Enter para sair...")
        return
    
    # Criação dos diretórios necessários
    try:
        os.makedirs(MCNP_RESULTS_DIR, exist_ok=True)
        os.makedirs(MCNP_SIM_DIR, exist_ok=True)
    except Exception as e:
        print(f"Falha crítica ao criar diretórios: {str(e)}")
        sys.exit(1)

    # Carregamento das configurações (CONFIGURAÇÕES SOBOL)
    print(f"\nCarregando configurações de: {INPUT_CONFIGS_CSV}")
    print("Nota: Configurações geradas via Sequências de Sobol (Fase 1 do Artigo)")
    try:
        df_configs = pd.read_csv(INPUT_CONFIGS_CSV)
        total_simulacoes = len(df_configs)
        print(f"Total de {total_simulacoes} configurações carregadas.")

        # Verificação de materiais
        materiais_utilizados = set()
        for i in range(1, 8):
            materiais_utilizados.update(df_configs[f'material_camada_{i}'].unique())

        materiais_nao_definidos = materiais_utilizados - set(MCNP_MATERIALS_DATA.keys())
        if materiais_nao_definidos:
            raise ValueError(f"ERRO: Materiais não definidos: {materiais_nao_definidos}")

        # Teste operacional opcional, sem modificar a lista Sobol nem o código.
        # Exemplo PowerShell: $env:SPHIN_TEST_IDS='1,1218'
        test_ids_raw = os.environ.get('SPHIN_TEST_IDS', '').strip()
        if test_ids_raw:
            test_ids = {int(value.strip()) for value in test_ids_raw.split(',') if value.strip()}
            ids_disponiveis = set(df_configs['id_simulacao'].astype(int))
            ids_ausentes = test_ids - ids_disponiveis
            if ids_ausentes:
                raise ValueError(f"IDs de teste inexistentes: {sorted(ids_ausentes)}")
            df_configs = df_configs[df_configs['id_simulacao'].astype(int).isin(test_ids)].copy()
            total_simulacoes = len(df_configs)
            print(f"🧪 MODO TESTE: executando somente os IDs {sorted(test_ids)}.")

    except Exception as e:
        print(f"ERRO ao carregar configurações: {str(e)}")
        sys.exit(1)

    # ✅ NOVO: FASE 1 - Gerar todos os inputs primeiro
    # print(f"\n📦 FASE 1: Gerando {total_simulacoes} inputs...")
    # start_time = time.time()
    # inputs_para_executar = []
    # all_mcnp_results = []

    # for index, row in df_configs.iterrows():
        
    # ✅ CORREÇÃO: VERIFICAR RESULTADOS EXISTENTES primeiro
    print(f"\n🔍 Verificando simulações existentes...")
    start_time = time.time()
    
    # Carregar resultados existentes ou criar novo
    results_file = MCNP_RESULTS_FILE
    if os.path.exists(results_file):
        try:
            df_existing = pd.read_csv(results_file)
            completas_existente = len(df_existing[df_existing['status_execucao_mcnp'] == 'sucesso'])
            pendentes_existente = len(df_existing[df_existing['status_execucao_mcnp'] == 'pendente'])
            print(f"📊 Simulações existentes: {completas_existente} completas, {pendentes_existente} pendentes")
            
            # Usar resultados existentes
            all_mcnp_results = df_existing.to_dict('records')
            print(f"🎯 Continuando de onde parou...")
        except Exception as e:
            print(f"⚠️  Erro ao carregar resultados existentes: {e}")
            all_mcnp_results = []
            print(f"🆕 Iniciando do zero...")
    else:
        all_mcnp_results = []
        print(f"🆕 Arquivo de resultados não existe, iniciando do zero")

    # ✅ CORREÇÃO: FASE 1 - Preparar APENAS simulações PENDENTES
    print(f"\n📦 FASE 1: Preparando inputs para simulações pendentes...")
    inputs_para_executar = []
    inputs_gerados = 0

    # Criar dicionário rápido para busca
    resultados_por_id = {r['id_simulacao']: r for r in all_mcnp_results}

    for index, row in df_configs.iterrows():
        sim_id = row['id_simulacao']
        
        # ✅ CORREÇÃO CRÍTICA: Se simulação já está COMPLETA, PULAR completamente
        if sim_id in resultados_por_id and resultados_por_id[sim_id]['status_execucao_mcnp'] == 'sucesso':
            # Verificar se o arquivo de input existe para possível reexecução
            result_data = resultados_por_id[sim_id]
            input_file = result_data.get('arquivo_input')
            if input_file and os.path.exists(input_file):
                inputs_para_executar.append({
                    'sim_id': sim_id,
                    'input_file': input_file,
                    'sim_dir': os.path.dirname(input_file),
                    'result_index': all_mcnp_results.index(result_data),
                    'materials': result_data['materiais'].split(', ') if result_data.get('materiais') else [],
                    'thicknesses': [float(t) for t in result_data['espessuras'].split(', ')] if result_data.get('espessuras') else []
                })
            continue  # ✅ PULAR simulação já completa
        
        # ✅ Se simulação NÃO existe ou está PENDENTE, processar
        try:
            # Extração de materiais e espessuras
            materials = [row[f'material_camada_{i+1}'] for i in range(7)]
            thicknesses = [row[f'espessura_camada_{i+1}'] for i in range(7)]

            # Validação da geometria
            total_espessura = sum(thicknesses)
            if not (5.99 <= total_espessura <= 6.01):
                raise ValueError(f"Espessura total deve ser 6cm (atual: {total_espessura:.2f}cm)")
            if any(t <= 0 for t in thicknesses):
                raise ValueError("Todas as espessuras devem ser positivas")

            # Cálculo dos raios externos das camadas
            current_radius = BLINDAGEM_RAIO_INTERNO_CM
            r_ext_values = []
            for thickness in thicknesses:
                current_radius += thickness
                r_ext_values.append(current_radius)

            # Geração dos materiais
            materials_used_in_config = list(set(materials))
            material_definitions = generate_all_materials(materials_used_in_config)

            # Preparação dos dados para o template
            template_data = {
                'sim_id': sim_id,
                'BLINDAGEM_RAIO_INTERNO_CM': BLINDAGEM_RAIO_INTERNO_CM,
                'BLINDAGEM_ALTURA_CM': BLINDAGEM_ALTURA_CM,
                'BLINDAGEM_Z_BASE_CM': BLINDAGEM_Z_BASE_CM,
                'BLINDAGEM_Z_TOPO_CM': BLINDAGEM_Z_BASE_CM + BLINDAGEM_ALTURA_CM,
                'DETECTOR_CELULA_ID': DETECTOR_CELULA_ID,
                'DETECTOR_RAIO_CM': DETECTOR_RAIO_CM,
                'DETECTOR_ALTURA_CM': DETECTOR_ALTURA_CM,
                'DETECTOR_Z_BASE_CM': DETECTOR_Z_BASE_CM,
                'DE4_CARD': DE4_ICRP116_AP,
                'DF4_CARD': DF4_ICRP116_AP,
                'FATOR_FONTE_PARA_MSV_S': FATOR_FONTE_PARA_MSV_S,
                'NPS_POR_SIMULACAO': NPS_POR_SIMULACAO,
                'RAND_SEED': RAND_SEED_BASE + RAND_SEED_STEP * (sim_id - 1),
                'material_definitions': material_definitions,
                'mat_id_Co60': MCNP_MATERIAL_IDS['Co60_Source'],
                'dens_Co60': MCNP_MATERIALS_DATA['Co60_Source']['densidade'],
                'mat_id_Air': MCNP_MATERIAL_IDS['Air'],
                'dens_Air': MCNP_MATERIALS_DATA['Air']['densidade'],
            }

            # Preenchimento das camadas
            for i in range(7):
                template_data[f'mat_id_{i+1}'] = MCNP_MATERIAL_IDS[materials[i]]
                template_data[f'dens_{i+1}'] = MCNP_MATERIALS_DATA[materials[i]]['densidade']
                template_data[f'mat_name_{i+1}'] = materials[i]
                template_data[f'r_ext_{i+1}'] = r_ext_values[i]

            # Geração do conteúdo do input MCNP
            mcnp_input_content = MCNP_TEMPLATE.format(**template_data)
            formatted_content = format_mcnp_input_for_vised(mcnp_input_content)
            input_sha256 = hashlib.sha256(formatted_content.encode('utf-8')).hexdigest()

            # Criação do diretório para a simulação
            current_sim_dir = os.path.join(MCNP_SIM_DIR, f"sim_{sim_id:05d}")
            os.makedirs(current_sim_dir, exist_ok=True)

            # ✅ CORREÇÃO: Verificar se input já existe antes de gerar
            input_file_path = os.path.join(current_sim_dir, 'input.i')
            if not os.path.exists(input_file_path):
                with open(input_file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                    inputs_gerados += 1

            # ✅ CORREÇÃO: Buscar ou criar registro de resultado
            if sim_id in resultados_por_id:
                result_data = resultados_por_id[sim_id]
                result_index = all_mcnp_results.index(result_data)
                result_data['input_sha256'] = input_sha256
                result_data['rand_seed'] = template_data['RAND_SEED']
            else:
                result_data = {
                    'id_simulacao': sim_id,
                    'status_geracao_input': 'sucesso',
                    'status_execucao_mcnp': 'pendente',
                    'status_extracao': 'pendente',
                    'erro_geracao': None,
                    'erro_execucao': None,
                    'erro_extracao': None,
                    'dose_efetiva_mSv_seg': np.nan,
                    'erro_relativo': np.nan,
                    'arquivo_input': input_file_path,
                    'arquivo_output': None,
                    'input_sha256': input_sha256,
                    'rand_seed': template_data['RAND_SEED'],
                    'materiais': ', '.join(materials),
                    'espessuras': ', '.join(f"{t:.2f}" for t in thicknesses),
                    'nps_tally4': None,
                    'terminou_nps_alvo': False,
                    'tfc_linha': None,
                    'tfc_todos_aprovados': False,
                    'tempo_parede_segundos': np.nan,
                    'tasks_mcnp': None,
                }
                all_mcnp_results.append(result_data)
                result_index = len(all_mcnp_results) - 1
                resultados_por_id[sim_id] = result_data

            # ✅ MUDANÇA: Armazenar info para execução
            inputs_para_executar.append({
                'sim_id': sim_id,
                'input_file': input_file_path,
                'sim_dir': current_sim_dir,
                'result_index': result_index,
                'materials': materials,
                'thicknesses': thicknesses
            })

        except Exception as e:
            print(f"❌ Erro na GERAÇÃO do input para simulação {sim_id}: {str(e)}")
            # Criar registro de erro se não existe
            if sim_id not in resultados_por_id:
                result_data = {
                    'id_simulacao': sim_id,
                    'status_geracao_input': 'erro',
                    'status_execucao_mcnp': 'pendente',
                    'status_extracao': 'pendente',
                    'erro_geracao': str(e),
                    'erro_execucao': None,
                    'erro_extracao': None,
                    'dose_efetiva_mSv_seg': np.nan,
                    'erro_relativo': np.nan,
                    'arquivo_input': None,
                    'arquivo_output': None,
                    'input_sha256': None,
                    'rand_seed': None,
                    'materiais': None,
                    'espessuras': None,
                    'nps_tally4': None,
                    'terminou_nps_alvo': False,
                    'tfc_linha': None,
                    'tfc_todos_aprovados': False,
                    'tempo_parede_segundos': np.nan,
                    'tasks_mcnp': None,
                }
                all_mcnp_results.append(result_data)
                resultados_por_id[sim_id] = result_data
        
        # Progresso da FASE 1
        if (index + 1) % 10 == 0:
            print(f"📦 Progresso FASE 1: {index + 1}/{total_simulacoes} verificados")

    print(f"✅ FASE 1 CONCLUÍDA: {inputs_gerados} novos inputs gerados, {len(inputs_para_executar)} totais para execução")
    
    # ✅ NOVO: FASE 2 - Executar simulações com controle rigoroso
    print(f"\n⚡ FASE 2: Executando {len(inputs_para_executar)} simulações (máximo {MAX_PARALLEL_SIMULATIONS} em paralelo)...")
    
    # Iniciar sistema de monitoramento
    print(f"🔍 Iniciando monitoramento automático...")
    monitor_thread = monitorar_progresso(MCNP_RESULTS_DIR, total_simulacoes, intervalo_minutos=5)

    simulacoes_com_erro_execucao = 0
    simulacoes_com_erro_extracao = 0

    # ✅ CORREÇÃO: Usar ThreadPoolExecutor em vez de ProcessPoolExecutor para evitar problemas de pickle
    # Com MAX_PARALLEL_SIMULATIONS = 2, o desempenho será satisfatório
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print(f"🔧 Configuração de paralelismo: {MAX_PARALLEL_SIMULATIONS} threads (evitando problemas de serialização)")

    # ✅ CORREÇÃO: Filtrar APENAS simulações PENDENTES para execução
    simulacoes_pendentes = []
    for sim_info in inputs_para_executar:
        result_index = sim_info['result_index']
        if (all_mcnp_results[result_index]['status_execucao_mcnp'] != 'sucesso'
                or all_mcnp_results[result_index].get('status_extracao') != 'sucesso'):
            simulacoes_pendentes.append(sim_info)

    print(f"🎯 Executando {len(simulacoes_pendentes)} simulações PENDENTES (pulando {len(inputs_para_executar) - len(simulacoes_pendentes)} completas)")

    # Executar APENAS simulações pendentes
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SIMULATIONS) as executor:
        # Submeter apenas simulações PENDENTES
        future_to_sim = {
            executor.submit(executar_simulacao_wrapper, sim_info): sim_info 
            for sim_info in simulacoes_pendentes
        }
        
        print(f"🎯 {len(future_to_sim)} simulações submetidas para execução...")
        
        # Processar resultados conforme forem completando
        for future in as_completed(future_to_sim):
            sim_info = future_to_sim[future]
            result_index = sim_info['result_index']
            
            try:
                resultado = future.result()
                all_mcnp_results[result_index]['tempo_parede_segundos'] = resultado.get('tempo_parede_segundos', np.nan)
                all_mcnp_results[result_index]['tasks_mcnp'] = resultado.get('tasks_mcnp')
                
                if resultado['success']:
                    all_mcnp_results[result_index]['status_execucao_mcnp'] = 'sucesso'
                    all_mcnp_results[result_index]['erro_execucao'] = None
                    all_mcnp_results[result_index]['arquivo_output'] = resultado['output_file']
                    
                    # ✅ CORREÇÃO: Extração ROBUSTA do tally
                    values, errors, cells = extract_mcnp_tally_result(resultado['output_file'], DETECTOR_CELULA_ID)
                    quality = extract_mcnp_quality(resultado['output_file'])
                    all_mcnp_results[result_index].update(quality)
                    
                    # Fallback se não encontrou
                    if not values:
                        print(f"🔄 Tentando fallback para sim {sim_info['sim_id']}...")
                        values, errors, cells = extract_tally_fallback(resultado['output_file'], DETECTOR_CELULA_ID)
                        
                    if values and cells:
                        # Usa o primeiro valor (assume que é para a célula do detector)
                        flux_value = values[0]
                        flux_error = errors[0] if errors is not None and len(errors) > 0 else 0.0
                                              
                        dose_rate_mSv_sec = flux_value
                        relative_error = flux_error
                        
                        all_mcnp_results[result_index]['dose_efetiva_mSv_seg'] = dose_rate_mSv_sec
                        all_mcnp_results[result_index]['erro_relativo'] = relative_error
                        all_mcnp_results[result_index]['status_extracao'] = 'sucesso'
                        all_mcnp_results[result_index]['erro_extracao'] = None
                        
                        print(f"✅ Sim {sim_info['sim_id']}: Dose = {dose_rate_mSv_sec:.2e} mSv/s")
                        if not quality['terminou_nps_alvo'] or quality['nps_tally4'] != NPS_POR_SIMULACAO:
                            print(f"⚠️ Sim {sim_info['sim_id']}: verificar NPS no OUTP "
                                  f"(tally4={quality['nps_tally4']}, alvo={NPS_POR_SIMULACAO})")
                        if not quality['tfc_todos_aprovados']:
                            print(f"⚠️ Sim {sim_info['sim_id']}: TFC incompleto; programar reexecução independente")
                        
                    else:
                        all_mcnp_results[result_index]['status_extracao'] = 'erro'
                        all_mcnp_results[result_index]['erro_extracao'] = "Tally do detector não encontrado"
                        simulacoes_com_erro_extracao += 1
                        print(f"⚠️ Sim {sim_info['sim_id']}: Tally não encontrado")
                        
                else:
                    all_mcnp_results[result_index]['status_execucao_mcnp'] = 'erro'
                    erro_msg = resultado.get('error', resultado.get('stderr', 'Erro desconhecido'))
                    all_mcnp_results[result_index]['erro_execucao'] = f"MCNP falhou: {erro_msg}"
                    simulacoes_com_erro_execucao += 1
                    print(f"❌ Sim {sim_info['sim_id']}: {erro_msg}")

            except Exception as e:
                print(f"❌ Erro inesperado na simulação {sim_info['sim_id']}: {str(e)}")
                all_mcnp_results[result_index]['status_execucao_mcnp'] = 'erro'
                all_mcnp_results[result_index]['erro_execucao'] = f"Erro inesperado: {str(e)}"
                simulacoes_com_erro_execucao += 1
            
            # ✅ CORREÇÃO: Salvamento INCREMENTAL após CADA simulação
            logging.info(json.dumps(all_mcnp_results[result_index]))
            salvar_resultados_parciais(all_mcnp_results, MCNP_RESULTS_FILE)

            # Progresso detalhado
            completadas = len([r for r in all_mcnp_results if r['status_execucao_mcnp'] in ['sucesso', 'erro']])
            sucessos = len([r for r in all_mcnp_results if r['status_execucao_mcnp'] == 'sucesso'])
            erros = len([r for r in all_mcnp_results if r['status_execucao_mcnp'] == 'erro'])
            
            if completadas % 5 == 0:  # A cada 5 simulações
                print(f"📊 Progresso: {completadas}/{total_simulacoes} | ✅ {sucessos} | ❌ {erros}")

    # ✅ FASE 3: Finalização e estatísticas
    end_time = time.time()
    total_time = end_time - start_time

    # Estatísticas finais
    sucesso_total = len([r for r in all_mcnp_results if r['status_execucao_mcnp'] == 'sucesso'])
    erro_exec_total = len([r for r in all_mcnp_results if r['status_execucao_mcnp'] == 'erro'])
    erro_geracao_total = len([r for r in all_mcnp_results if r['status_geracao_input'] == 'erro'])
    erro_extracao_total = len([r for r in all_mcnp_results if r.get('status_extracao') == 'erro'])
    
    # Salvamento final
    df_results = pd.DataFrame(all_mcnp_results)
    df_results.to_csv(MCNP_RESULTS_FILE, index=False)
    
    print(f"\n🎉 SIMULAÇÕES CONCLUÍDAS!")
    print(f"⏱️  Tempo total: {total_time/3600:.2f} horas")
    
    print(f"\n📊 RESUMO FINAL:")
    print(f"   ✅ Simulações bem-sucedidas: {sucesso_total}/{total_simulacoes} ({sucesso_total/total_simulacoes*100:.1f}%)")
    print(f"   ❌ Erros de geração de input: {erro_geracao_total}")
    print(f"   ❌ Erros de execução MCNP: {erro_exec_total}")
    print(f"   ❌ Erros de extração de dados: {erro_extracao_total}")
    
    if sucesso_total > 0:
        doses = [r['dose_efetiva_mSv_seg'] for r in all_mcnp_results if not np.isnan(r['dose_efetiva_mSv_seg'])]
        if doses:
            print(f"   📈 Dose mínima: {min(doses):.2e} mSv/s")
            print(f"   📈 Dose máxima: {max(doses):.2e} mSv/s")
            print(f"   📈 Dose média: {np.mean(doses):.2e} mSv/s")

    print(f"\n💾 Resultados salvos em: {MCNP_RESULTS_FILE}")
    print("✓ Processo concluído!")

if __name__ == "__main__":
    main()
