import pandas as pd
import os

def transform_csv(input_path, output_path):
    """
    Lê o CSV original e transforma no formato desejado.
    """
    # Leitura do CSV original
    df = pd.read_csv(input_path, encoding='utf-8')
    
    # Verifica se as colunas esperadas existem
    required_cols = ['id_simulacao', 'dose_efetiva_mSv_seg', 'erro_relativo', 
                     'materiais', 'espessuras']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("O arquivo de entrada não contém todas as colunas necessárias.")
    
    # Seleciona as colunas de interesse e renomeia conforme o formato de saída
    df_out = df[required_cols].copy()
    
    # Converte a coluna dose para numérico (caso venha como string)
    df_out['dose_efetiva_mSv_seg'] = pd.to_numeric(df_out['dose_efetiva_mSv_seg'], errors='coerce')
    
    # Calcula a dose para 5 minutos
    df_out['5 min de Dose efetiva (mSv)'] = (df_out['dose_efetiva_mSv_seg'] * 300.0).round(4)
    
    # Ajusta a ordem das colunas conforme solicitado
    colunas_final = ['id_simulacao', 'dose_efetiva_mSv_seg', 'erro_relativo', 
                     'materiais', 'espessuras', '5 min de Dose efetiva (mSv)']
    df_out = df_out[colunas_final]
    
    # Salva o resultado
    df_out.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Arquivo transformado salvo em: {output_path}")

if __name__ == "__main__":
    # Define os caminhos (ajuste conforme necessário)
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    pasta_python = os.path.join(desktop, "Python")
    
    input_file = os.path.join(pasta_python, "resultados_mcnp_sobol_raw.csv")
    output_file = os.path.join(pasta_python, "resultados_mcnp_sobol.csv")
    
    if not os.path.exists(input_file):
        print(f"Erro: arquivo de entrada não encontrado em {input_file}")
    else:
        transform_csv(input_file, output_file)