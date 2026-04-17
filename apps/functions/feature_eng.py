# Feature engineering functions for clustering analysis
from sklearn.metrics import silhouette_samples, silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import calinski_harabasz_score
import random
import plotly.express as px
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram
from scipy.spatial.distance import pdist

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt


def score_silueta(
    X,
    min_compo,
    max_compo,
    model_type="kmeans",   # "kmeans" o "gmm"
    random_state=42
):
    silueta = []

    for k in range(min_compo, max_compo):

        if model_type == "kmeans":
            model = KMeans(n_clusters=k, random_state=random_state)
            labels = model.fit_predict(X)

        elif model_type == "gmm":
            model = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state
            )
            labels = model.fit_predict(X)

        else:
            raise ValueError("model_type debe ser 'kmeans' o 'gmm'")

        score = silhouette_score(X, labels, metric="euclidean")
        silueta.append(score)

    df_silueta = pd.DataFrame({
        "N_Clusters": range(min_compo, max_compo),
        "score": silueta
    })

    fig = px.line(
        df_silueta,
        x="N_Clusters",
        y="score",
        markers=True,
        title=f"Número óptimo de clusters - Silueta ({model_type.upper()})"
    )

    fig.show()

def codo(X, min_compo, max_compo):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    distortions = []
    K = range(min_compo, max_compo)
    for k in K:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X)
        distortions.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(list(K), distortions, "bx-")
    ax.set_xlabel("k")
    ax.set_ylabel("Distortion (Inertia)")
    ax.set_title("Elbow Method")
    return fig

def score_calinski(X,min_compo,max_compo):
    calinski = []
    for k in list(range(min_compo, max_compo)):
        km = KMeans(n_clusters=k)
        #km.fit(X_std)
        labels = km.fit_predict(X)
        score = calinski_harabasz_score(X,labels)
        calinski.append(score)
    df_calinski=pd.DataFrame()
    df_calinski["N_Clusters"]=range(min_compo,max_compo)
    df_calinski["score"]=calinski
    fig = px.line(df_calinski, x="N_Clusters", y="score", title="Número de clusters óptimo - Calinski")
    return fig.show()