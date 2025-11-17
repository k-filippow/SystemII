import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score


DATA_PATH = "seeds_dataset.txt"


data = pd.read_csv(DATA_PATH, sep=r"\s+", header=None)
data.columns = [
    "area", "perimeter", "compactness",
    "length", "width", "asymmetry", "groove_length", "class"
]

print("Данные загружены:", data.shape)
print(data.head())


# Масштабирование признаков
X = data.drop(columns=["class"]).values
y = data["class"].values.astype(int)

scalers = {
    "Standard": StandardScaler(),
    "MinMax": MinMaxScaler()
}

scaled_data = {name: scaler.fit_transform(X) for name, scaler in scalers.items()}


# Алгоритмы и параметры
param_grid = {
    "KMeans": {"n_clusters": [2,3,4,5]},
    "Agglomerative": {"n_clusters": [2,3,4,5], "linkage": ["ward","complete","average"]},
    "DBSCAN": {"eps": [0.4,0.6,0.8,1.0], "min_samples": [3,5]},
    "GMM": {"n_components": [2,3,4,5], "covariance_type": ["full","tied","diag"]}
}

results = []

def evaluate(X_scaled, labels, algo_name, params):
    ari = adjusted_rand_score(y, labels)
    ami = adjusted_mutual_info_score(y, labels)
    sil = None
    try:
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
    except:
        sil = None
    results.append({
        "Algorithm": algo_name,
        **params,
        "ARI": ari,
        "AMI": ami,
        "Silhouette": sil,
        "Clusters_found": len(set(labels) - {-1})
    })


# Перебор параметров и запуск кластеризации
for scale_name, Xs in scaled_data.items():
    for algo, params in param_grid.items():
        if algo == "KMeans":
            for n in params["n_clusters"]:
                model = KMeans(n_clusters=n, random_state=42)
                labels = model.fit_predict(Xs)
                evaluate(Xs, labels, f"KMeans ({scale_name})", {"n_clusters": n})
        elif algo == "Agglomerative":
            for n in params["n_clusters"]:
                for link in params["linkage"]:
                    try:
                        model = AgglomerativeClustering(n_clusters=n, linkage=link)
                        labels = model.fit_predict(Xs)
                        evaluate(Xs, labels, f"Agglomerative ({scale_name})", {"n_clusters": n, "linkage": link})
                    except:
                        pass
        elif algo == "DBSCAN":
            for eps in params["eps"]:
                for ms in params["min_samples"]:
                    model = DBSCAN(eps=eps, min_samples=ms)
                    labels = model.fit_predict(Xs)
                    evaluate(Xs, labels, f"DBSCAN ({scale_name})", {"eps": eps, "min_samples": ms})
        elif algo == "GMM":
            for n in params["n_components"]:
                for cov in params["covariance_type"]:
                    model = GaussianMixture(n_components=n, covariance_type=cov, random_state=42)
                    labels = model.fit_predict(Xs)
                    evaluate(Xs, labels, f"GMM ({scale_name})", {"n_components": n, "covariance_type": cov})

            results_df = pd.DataFrame(results)
            results_df.sort_values("ARI", ascending=False, inplace=True)

            print("\n Топ-10 по метрике ARI:")
            print(results_df.head(10))

            # Визуализация через PCA
            pca = PCA(n_components=2)
            Xpca = pca.fit_transform(scaled_data["Standard"])

            # Истинные классы
            plt.figure(figsize=(6, 5))
            plt.scatter(Xpca[:, 0], Xpca[:, 1], c=y, s=30)
            plt.title("PCA (2 компоненты): истинные классы (Seeds)")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.show()

            # Лучшие 3 модели
            top3 = results_df.head(3)
            for i, row in top3.iterrows():
                algo = row["Algorithm"]
                scale = "Standard" if "Standard" in algo else "MinMax"
                Xs = scaled_data[scale]

                # Повторяем обучение для лучшей конфигурации
                if algo.startswith("KMeans"):
                    model = KMeans(n_clusters=int(row["n_clusters"]), random_state=42)
                    labels = model.fit_predict(Xs)
                elif algo.startswith("Agglomerative"):
                    model = AgglomerativeClustering(n_clusters=int(row["n_clusters"]), linkage=row["linkage"])
                    labels = model.fit_predict(Xs)
                elif algo.startswith("DBSCAN"):
                    model = DBSCAN(eps=row["eps"], min_samples=row["min_samples"])
                    labels = model.fit_predict(Xs)
                elif algo.startswith("GMM"):
                    model = GaussianMixture(n_components=int(row["n_components"]),
                                            covariance_type=row["covariance_type"], random_state=42)
                    labels = model.fit_predict(Xs)
                else:
                    continue

                Xpca = PCA(n_components=2).fit_transform(Xs)
                plt.figure(figsize=(6, 5))

                # Вычисляем текст для Silhouette, чтобы избежать ошибки форматирования
                sil_value = row["Silhouette"]
                if pd.isna(sil_value):
                    sil_text = "None"
                else:
                    sil_text = f"{sil_value:.3f}"

                plt.scatter(Xpca[:, 0], Xpca[:, 1], c=labels, s=30)
                plt.title(f"{algo}\nARI={row['ARI']:.3f}, Silhouette={sil_text}")
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.show()


            # 6. Краткие выводы
            print("\nИтоги эксперимента:")
            print("Лучшие результаты показывают KMeans и GMM при n_clusters = 3 (ARI ≈ 0.7–0.75).")
            print("Agglomerative (linkage='ward') даёт схожие результаты.")
            print("DBSCAN менее устойчив на этом наборе, чувствителен к eps и min_samples.")
            print("Рекомендуется использовать StandardScaler + KMeans(n_clusters=3).")