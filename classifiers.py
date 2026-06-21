import pandas as pd
import warnings
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import NuSVC

warnings.filterwarnings("ignore", category=FutureWarning)


def build_pipeline(model):
    """
    cria um pipeline que encadeia duas etapas automaticamente para cada modelo
    faz o StandardScaler normalizando os dados
    e depois passa para o classificador porque o SVM e k-NN sao sensiveis a escala
    """
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def get_models():
    """
    Essa funcao retorna uma lista de 20 modelos de classificação incluindo os pedidos
    pelo professor (k-NN, Árvore de Decisão, SVM, MLP, Random Forest).
    Posteriormente, o desempenho de todos esses sera comparado na etapa de avaliação
    durante o VotingClassifier, onde o ensemble de todos os modelos sera criado.
    O desempenho individual de cada modelo tambem sera avaliado e salvo em um arquivo CSV
    para cada dataset.

     Inclui:
        - Modelos lineares (Logistic Regression, Ridge)
        - Baseados em distância (k-NN, Nearest Centroid)
        - Árvores e ensembles (Random Forest, Extra Trees, Gradient Boosting)
        - SVMs com diferentes kernels
        - Redes neurais (MLP)
        - Modelos probabilísticos (Naive Bayes)
        - XGBoost
    """
    return [
        # 1. k-NN (4 variações)
        ("knn3", KNeighborsClassifier(3)),
        ("knn5", KNeighborsClassifier(5)),
        ("knn7", KNeighborsClassifier(7)),
        ("knn11", KNeighborsClassifier(11)),
        # 2. SVM (4 variações)
        ("svm_rbf", SVC(kernel="rbf", probability=True)),
        ("svm_linear", SVC(kernel="linear", probability=True)),
        ("svm_poly", SVC(kernel="poly", probability=True)),
        ("svm_sigmoid", SVC(kernel="sigmoid", probability=True)),
        # 3. Random Forest (4 variações)
        ("rf50", RandomForestClassifier(50)),
        ("rf100", RandomForestClassifier(100)),
        ("rf200", RandomForestClassifier(200)),
        ("rf300", RandomForestClassifier(300)),
        # 4. MLP (4 variações)
        ("mlp1", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500)),
        ("mlp2", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)),
        ("mlp3", MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=500)),
        ("mlp4", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)),
        # 5. Árvores de Decisão (4 variações)
        ("dt_inf", DecisionTreeClassifier(max_depth=None)),
        ("dt_d3", DecisionTreeClassifier(max_depth=3)),
        ("dt_d5", DecisionTreeClassifier(max_depth=5)),
        ("dt_d10", DecisionTreeClassifier(max_depth=10)),
    ]


def save_confusion_matrix(cm_percent, name):
    """
    Gera e salva a matriz de confusão em formato percentual.

    - cm_percent: matriz normalizada por classe real
    - name: nome do dataset (ResNet50 ou ViT)

    A matriz mostra:
    - Linhas = classe real
    - Colunas = classe predita
    """
    filename = f"confusion_{name}.png"
    print(f"-> [PLOT] Gerando gráfico da matriz de confusão para {name}...", flush=True)
    plt.figure(figsize=(8, 6))
    # matriz já normalizada em percentual
    plt.imshow(cm_percent)
    plt.title(f"{name} - Confusion Matrix (%)")
    plt.colorbar()

    n = cm_percent.shape[0]
    plt.xticks(np.arange(n))
    plt.yticks(np.arange(n))

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{cm_percent[i, j]:.1f}%", ha="center", va="center")

    plt.xlabel("Predito")
    plt.ylabel("Real")

    plt.tight_layout()
    plt.savefig(f"confusion_{name}.png", dpi=300)
    plt.show()
    plt.close()
    print(f"-> [PLOT] Matriz salva com sucesso em: {filename}", flush=True)


all_results = []


def evaluate(X, y, name):
    """
    Executa a avaliação completa de um dataset:

    1. Codifica labels
    2. Aplica validação cruzada (10-fold)
    3. Avalia os 28 modelos individuais
    4. Avalia ensemble (VotingClassifier)
    5. Gera matriz de confusão do ensemble
    6. Salva resultados em CSV
    """
    print(f"\n================ {name} ================", flush=True)

    le = LabelEncoder()
    y = le.fit_transform(y)
    print("Classes:", le.classes_, flush=True)

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    models = get_models()

    print(
        f"-> Iniciando validação cruzada para os 20 modelos individuais...", flush=True
    )
    for model_name, model in tqdm(models, desc=f"Models ({name})"):
        print(f"\n[MODELO] {model_name}", flush=True)

        pipeline = build_pipeline(model)
        preds = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)

        acc = accuracy_score(y, preds) * 100
        f1 = f1_score(y, preds, average="weighted") * 100

        all_results.append([name, model_name, acc, f1])
        print(f"{model_name} | ACC={acc:.2f}% | F1={f1:.2f}%", flush=True)

    print("\n[ENSEMBLE] Iniciando VotingClassifier...", flush=True)
    # n_jobs=1 aqui evita o conflito de concorrência com o cross_val_predict de baixo
    ensemble = VotingClassifier(
        estimators=[(n, build_pipeline(m)) for n, m in models], voting="soft", n_jobs=1
    )

    print("-> [ENSEMBLE] Executando predições com validação cruzada...", flush=True)
    preds = cross_val_predict(ensemble, X, y, cv=cv, n_jobs=-1)

    acc = accuracy_score(y, preds) * 100
    f1 = f1_score(y, preds, average="weighted") * 100

    print(f"\nENSEMBLE ACC: {acc:.2f}%", flush=True)
    print(f"ENSEMBLE F1: {f1:.2f}%", flush=True)

    cm = confusion_matrix(y, preds)
    cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    save_confusion_matrix(cm_percent, name)

    csv_name = f"ensemble_{name}.csv"
    pd.DataFrame(
        [[name, "ensemble", acc, f1]], columns=["dataset", "model", "acc", "f1"]
    ).to_csv(csv_name, index=False)
    print(f"-> Arquivo {csv_name} salvo com sucesso.", flush=True)


def merge_results():
    """
    Junta resultados dos dois backbones (ResNet e ViT)
    e gera análise global.
    """
    ensemble_resnet = pd.read_csv("ensemble_ResNet50.csv")
    ensemble_vit = pd.read_csv("ensemble_ViT.csv")
    individuals = pd.read_csv("results_all.csv")  # já tem os dois backbones

    ensemble_resnet["backbone"] = "ResNet50"
    ensemble_vit["backbone"] = "ViT"

    individuals["backbone"] = individuals["dataset"].map(
        {"ResNet50": "ResNet50", "ViT": "ViT"}
    )

    df = pd.concat([individuals, ensemble_resnet, ensemble_vit])
    df.to_csv("results_ALL.csv", index=False)

    print("\n===== TOP 10 MODELS OVERALL =====")
    print(df.sort_values("acc", ascending=False).head(10))

    print("\n===== MEAN ACC BY BACKBONE =====")
    print(df.groupby("backbone")["acc"].mean())


if __name__ == "__main__":
    print("Loading datasets...")

    df_resnet = pd.read_csv("result_final_resnet50.csv")
    df_vit = pd.read_csv("result_final_vit_large.csv")

    X_resnet = df_resnet.drop(columns=["image_path", "label"]).values
    y_resnet = df_resnet["label"].values

    X_vit = df_vit.drop(columns=["image_path", "label"]).values
    y_vit = df_vit["label"].values

    print("ResNet shape:", X_resnet.shape)
    print("ViT shape:", X_vit.shape)

    evaluate(X_resnet, y_resnet, "ResNet50")
    evaluate(X_vit, y_vit, "ViT")
    df_all = pd.DataFrame(all_results, columns=["dataset", "model", "acc", "f1"])
    df_all.to_csv("results_all.csv", index=False)
    # analise global dos resultados individuais e do ensemble
    merge_results()
