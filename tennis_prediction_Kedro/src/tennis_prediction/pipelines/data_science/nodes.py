import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from xgboost import XGBClassifier

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    X = data[['Rank_1', 'Rank_2', 'Pts_1', 'Pts_2', 'Odd_1', 'Odd_2',
              'Ranking_Ganador', 'Gano_el_que_tenia_mas_puntos',
              'Gano_el_que_tenia_menor_odd']]
    y = data['Winner_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters.get("test_size", 0.2), random_state=42
    )
    return X_train, X_test, y_train.to_frame(), y_test.to_frame()


def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> tuple:
    
    #---XGBoost---
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train.values.ravel())
    y_pred = xgb.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=["prediction"])

    return xgb, y_pred


def evaluate_model(model: XGBClassifier, y_test: pd.Series, y_pred: pd.Series) -> dict:
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Matriz de Confusión XGBoost:")
    print(cm)
    print("\nReporte de Clasificación XGBoost:")
    print(classification_report(y_test, y_pred))
    
    return {
        "accuracy": accuracy,
        "precision": report['1']['precision'],
        "recall": report['1']['recall'],
        "f1-score": report['1']['f1-score'],
    }