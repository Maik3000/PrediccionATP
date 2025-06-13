import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

def process_data(atp_tennis: pd.DataFrame) -> pd.DataFrame:

    df = atp_tennis.copy()
    #eliminacion de variables insignificante
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    limpieza=['Court', 'Series']
    df.drop(columns=limpieza, inplace=True)
    df = df.dropna()

    df_actual = df[df["Date"].dt.year > 2003]
    
    #------- Mejora futura año tras año: Para filtrar ranking actual cambiar 2025----------

    # Filtrar solo los partidos de 2025
    df_2025 = df_actual[df_actual["Date"].dt.year == 2025]

    # Juntar jugadores con sus rankings en 2025
    players_1 = df_2025[["Player_1", "Rank_1"]].rename(columns={"Player_1": "Player", "Rank_1": "Rank"})
    players_2 = df_2025[["Player_2", "Rank_2"]].rename(columns={"Player_2": "Player", "Rank_2": "Rank"})
    ranking_2025 = pd.concat([players_1, players_2])

    # Mantener solo el mejor ranking de cada jugador
    ranking_2025 = ranking_2025.groupby("Player")["Rank"].min().reset_index()

    # Filtrar top 200
    top100_players = ranking_2025[ranking_2025["Rank"] <= 100]["Player"].unique()
    top100_players = set(top100_players)

    df_top100 = df_actual[
    (df_actual["Player_1"].isin(top100_players)) |
    (df_actual["Player_2"].isin(top100_players))]

    return df_top100


def feature_engineering(df_top100: pd.DataFrame) -> pd.DataFrame:
    
    df_top100['rank_diff'] = df_top100['Rank_1'] - df_top100['Rank_2']
    df_top100['pts_diff'] = df_top100['Pts_1'] - df_top100['Pts_2']
    df_top100['odds_diff'] = df_top100['Odd_1'] - df_top100['Odd_2']
    df_top100['Ranking_Ganador'] = df_top100.apply(lambda row: row['Rank_1'] if row['Winner'] == row['Player_1'] else row['Rank_2'], axis=1)

    df_top100['Gano_el_que_tenia_mas_puntos'] = df_top100.apply(
        lambda row: (row['Pts_1'] > row['Pts_2'] and row['Winner'] == row['Player_1']) or 
                    (row['Pts_2'] > row['Pts_1'] and row['Winner'] == row['Player_2']), 
        axis=1
    )

    df_top100['Gano_el_que_tenia_menor_odd'] = df_top100.apply(
        lambda row: (row['Odd_1'] < row['Odd_2'] and row['Winner'] == row['Player_1']) or 
                    (row['Odd_2'] < row['Odd_1'] and row['Winner'] == row['Player_2']), 
        axis=1
    )

    return df_top100


def encoder(df_top100_features: pd.DataFrame) -> pd.DataFrame:
    
    df_top100=df_top100_features.copy()
    # Inicializar OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')  # Deja solo una columna (3 → 0, 5 → 1)

    # Aplicar One-Hot Encoding
    encoded_array = encoder.fit_transform(df_top100[['Best of']])

    # Agregar la columna transformada al DataFrame
    df_top100['Best of'] = encoded_array.astype(int)  # Convertir los valores flotantes a enteros

    df_top100['Gano_el_que_tenia_mas_puntos'] = df_top100['Gano_el_que_tenia_mas_puntos'].astype(int)
    df_top100['Gano_el_que_tenia_menor_odd'] = df_top100['Gano_el_que_tenia_menor_odd'].astype(int)

    surface_mapping = {'Hard': 2, 'Clay': 1, 'Grass': 0}

    # Aplicar el mapeo a la columna Surface
    df_top100['Surface'] = df_top100['Surface'].map(surface_mapping)

    # Crear una nueva variable de resultado binaria
    df_top100['Winner_binary'] = (df_top100['Winner'] == df_top100['Player_1']).astype(int)

    # Verificar la nueva variable
    print(df_top100[['Player_1', 'Player_2', 'Winner', 'Winner_binary']].head())

    # Eliminar la columna original Winner (opcional)
    df_top100.drop(columns=['Winner'], inplace=True)

    return df_top100


def scaler(df_top100_encoder: pd.DataFrame) -> pd.DataFrame:

    df_top100=df_top100_encoder.copy()
    
    numeric_columns = ['Surface', 'Best of', 'Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2', 'rank_diff', 'pts_diff', 'odds_diff', 'Ranking_Ganador', 'Gano_el_que_tenia_mas_puntos', 'Gano_el_que_tenia_menor_odd']
    # Crear el escalador
    scaler = StandardScaler()

    # Aplicar la estandarización a las variables numéricas
    df_top100[numeric_columns] = scaler.fit_transform(df_top100[numeric_columns])

    return df_top100


