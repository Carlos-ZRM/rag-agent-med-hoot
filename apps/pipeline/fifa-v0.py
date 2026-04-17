"""Example of a pipeline to demonstrate a simple real world data science workflow."""

from ast import arguments
import html
from io import BytesIO
import os
import sys
import pandas as pd

import kfp.compiler
from kfp import dsl
from kfp_helper import execute_pipeline_run

base_image = os.getenv("BASE_IMAGE", "image-registry.openshift-image-registry.svc:5000/openshift/python:latest")
kubeflow_endpoint = os.environ["KUBEFLOW_ENDPOINT"]
bearer_token = os.environ["BEARER_TOKEN"]

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "requests"],
)
def read_csv_from_url(
    url: str,
    summary: dsl.Output[dsl.Artifact],
    dataset: dsl.Output[dsl.Dataset],
) -> int:
    """Downloads a CSV from a URL and persists it as a KFP Dataset artifact."""
    import pandas as pd
    import json
 
    df = pd.read_csv(url)
 


    dataset.metadata["source_url"] = url
    dataset.metadata["num_rows"] = df.shape[0]
    dataset.metadata["num_cols"] = df.shape[1]
 
    # Write a summary artifact
    summary_data = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "describe": df.describe(include="all").to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
    }
    with open(summary.path, "w") as f:
        json.dump(summary_data, f, indent=2, default=str)
    summary.metadata["source_url"] = url
 
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\n{df.head()}")
    print(f"\nNull counts:\n{df.isnull().sum()}")
 
    df.columns = df.columns.str.replace(' ', '_')
    df.columns= df.columns.str.lower()
    df.head()

    # Persist the dataframe as CSV artifact
    #df.to_csv(dataset.path, index=False)
    df.to_pickle(dataset.path)
    return df.shape[0]

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn"],
)
def create_eda_report(
    dataset: dsl.Input[dsl.Dataset],
    eda_report: dsl.Output[dsl.Artifact],
    eda_dataset_output: dsl.Output[dsl.Dataset],
):
    import pandas as pd
    import json

    df = pd.read_pickle(dataset.path)

    # Doing a simple EDA by checking for duplicates and null values, and then saving a report as a JSON artifact. In a real world scenario, this could be a more comprehensive EDA with visualizations and insights.
    eda_dataset = df.drop_duplicates(subset=["name","nationality","age"])

    num_duplicados = df.shape[0] - eda_dataset.shape[0]
    eda_dataset = eda_dataset.drop(
        ['photo','jersey_number','flag',
        'joined','real_face','wage','value','wage','release_clause',
        'body_type','club_logo','loaned_from','contract_valid_until'],axis=1)
    
    completitud=(eda_dataset.isnull().sum()/eda_dataset.shape[0]).reset_index()
    completitud.columns=["columna","completitud"]
    completitud.sort_values(by=["completitud"],ascending=False)


    eda_dataset.to_pickle(eda_dataset_output.path)

    eda_data = {
        "shape": list(eda_dataset.shape),
        "completitud": completitud.to_dict(orient="records"),
        #"columns": eda_dataset.columns.tolist(),
        #"dtypes": {col: str(dtype) for col, dtype in eda_dataset.dtypes.items()},
        #"describe": eda_dataset.describe(include="all").to_dict(),
        "null_counts": eda_dataset.isnull().sum().to_dict(),
        "num_duplicados": num_duplicados
    }
    with open(eda_report.path, "w") as f:
        json.dump(eda_data, f, indent=2, default=str)
    
@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "requests", "matplotlib", "seaborn"],
)
def clean_data(
    eda_dataset: dsl.Input[dsl.Dataset],
    plot_distribution: dsl.Output[dsl.HTML],
    plot_tad_correlation: dsl.Output[dsl.HTML],
    tad_dataset: dsl.Output[dsl.Dataset],

):
    import pandas as pd
    import requests
    import sys
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import seaborn as sns



    # Import function from EDA component. In a real world scenario, this could be a separate Python module with various data cleaning functions that we import and use here.
    eda_url = "https://raw.githubusercontent.com/Carlos-ZRM/rag-agent-med-hoot/main/apps/functions/eda.py"
    r = requests.get(eda_url)

    with open("/tmp/eda.py", "w") as f:
        f.write(r.text)

    sys.path.insert(0, "/tmp")
    from eda import completitud, desc_table, detectar_outliers, rename_column

    datos = pd.read_pickle(eda_dataset.path)
    #separa posiciones en columnas individuales
    separa=[ 'ls', 'st', 'rs', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam', 'ram', 'lm', 'lcm', 'cm','rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm', 'rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb']
    
    datos[separa] = datos[separa].fillna(0).astype(int)

    # Reemplazamos caracteres no numéricos y convertimos a float
    datos['height']=datos['height'].replace("'",'.',regex=True).astype(float)
    datos['weight']=datos['weight'].replace("lbs",'',regex=True).astype(float)


    datos["work_rate"] = (
        datos["work_rate"]
        .replace({"Low": "1", "Medium": "2", "High": "3", "/ ": "."}, regex=True)
        .fillna(0)
        .astype(float)
    )

    # Eliminamos columnas que no aportan valor al análisis y que además tienen alta proporción de nulos
    datos.drop(['international_reputation','preferred_foot','id'],axis=1,inplace=True)

    # Identificamos variables numéricas para el análisis de outliers y correlación
    numericas=desc_table(datos,datos.columns,'num')
    numericas=numericas['index'].values.tolist()

    datos_out, resumen, scores = detectar_outliers(
        datos,
        variables=numericas,
        medida_iqr=1.5,
        pct_extremos=(0.005, 0.995),
        usar_z_robusto=True,
        thr_z_robusto=3.5,
        usar_isoforest=True,
        contamination="auto"
    )

    # Filtramos los outliers detectados por Isolation Forest para el análisis de distribución y correlación
    datos=datos_out[datos_out["out_iforest"]==0]
    datos=datos.drop(columns=["out_pct_extremo",
                    "out_iqr",
                    "out_zrob",
                    "out_uni","out_iforest","out_total"])
    
    fig = datos[numericas].hist(
        figsize=(30, 20), bins=20, align="left",
        color="cyan", edgecolor="black",
    )
    plt_fig = fig.flatten()[0].get_figure()

    # Convertir a base64 para embeber en HTML
    buf = BytesIO()
    plt_fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(plt_fig)


    html = f'<html><body><img src="data:image/png;base64,{img_b64}"/></body></html>'
    with open(plot_distribution.path, "w") as f:
        f.write(html)
    
    # Renombramos variables con formato de columna
    for i in numericas:
        if i=='work_rate':
            rename_column(datos,i,'categorica')
        else:
            rename_column(datos,i,'numerica')
    # Analizamos correlación entre variables numéricas
    vars=[x for x in datos.columns if x not in ['nationality','name','club','position']]
    tad=datos[vars]
    
    tad = tad.reset_index(drop=True)
    
    corr_df = tad.corr()
    
    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        round(corr_df, 2),
        annot=True, cmap="YlGnBu", linewidths=3, fmt=".1g", ax=ax,
    )
    ax.set_title("Correlation Matrix")

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    html = f'<html><body><img src="data:image/png;base64,{img_b64}"/></body></html>'
    with open(plot_tad_correlation.path, "w") as f:
        f.write(html)

    tad.to_pickle(tad_dataset.path)

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "plotly", "matplotlib", "requests", "yellowbrick"],
)
def feature_engineering(
    tad_dataset: dsl.Input[dsl.Dataset],
    pca_dataset: dsl.Output[dsl.Dataset],
    plot_pca: dsl.Output[dsl.HTML],
    plot_elbow: dsl.Output[dsl.HTML],
):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px
    
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import pandas as pd
    import requests
    import sys
    

    feature_eng_url = "https://raw.githubusercontent.com/Carlos-ZRM/rag-agent-med-hoot/refs/heads/main/apps/functions/feature_eng.py"

    r = requests.get(feature_eng_url)

    with open("/tmp/feature_eng.py", "w") as f:
            f.write(r.text)
    sys.path.insert(0, "/tmp")


    from feature_eng import codo

    tad = pd.read_pickle(tad_dataset.path)

    # standardize the data before applying PCA
    SS = StandardScaler()
    df_scaled = pd.DataFrame(SS.fit_transform(tad),columns=tad.columns)
    df_scaled.head()

    # make pca and calculate cumulative explained variance for different number of components to determine how many components to keep for dimensionality reduction. In a real world scenario, this could be part of a more comprehensive feature engineering step where we create new features, select important features, and apply dimensionality reduction techniques like PCA.
   
    pca_comp=[]
    for n_compo in range(1,25):
        pca = PCA(n_components=n_compo)
        pca.fit(df_scaled)
        resultado=sum(pca.explained_variance_ratio_)
        pca_comp.append(resultado)

    df_pca=pd.DataFrame()
    df_pca["componente"]=range(1,25)
    df_pca["varianza"] = pd.Series(pca_comp).round(3)

    fig = px.histogram(df_pca, x="componente", y="varianza", nbins=25, text_auto=True, facet_row_spacing=0.2)
    
    # Convertir a base64 para embeber en HTML
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    with open(plot_pca.path, "w") as f:
        f.write(html)
    
    pca = PCA(n_components=16)
    
    pca_df = pca.fit_transform(df_scaled)
    df_pca =pd.DataFrame(pca_df)
    df_pca.columns=['comp1','comp2','comp3','comp4','comp5','comp6','comp7','comp8', 'comp9','comp10','comp11','comp12','comp13','comp14','comp15','comp16']


    fig_elbow = codo(pca_df, 2, 20)

    buf = BytesIO()
    fig_elbow.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig_elbow)

    html = f'<html><body><img src="data:image/png;base64,{img_b64}"/></body></html>'
    with open(plot_elbow.path, "w") as f:
        f.write(html)

    df_pca.to_pickle(pca_dataset.path)

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "plotly", "matplotlib", "requests", "yellowbrick"],
)
def score_best_num_class_kmeans(
    pca_dataset: dsl.Input[dsl.Dataset],
    plot_kmeans: dsl.Output[dsl.HTML],
):
    import requests
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd
    

    feature_eng_url = "https://raw.githubusercontent.com/Carlos-ZRM/rag-agent-med-hoot/refs/heads/main/apps/functions/feature_eng.py"
    r = requests.get(feature_eng_url)
    with open("/tmp/feature_eng.py", "w") as f:
            f.write(r.text)
    sys.path.insert(0, "/tmp")
    from feature_eng import score_silueta

    df_pca = pd.read_pickle(pca_dataset.path)
    fig_kmeans = score_silueta(df_pca, 2, 20, model_type="kmeans")


    html = fig_kmeans.to_html(include_plotlyjs="cdn", full_html=True)
    with open(plot_kmeans.path, "w") as f:
        f.write(html)

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "plotly", "matplotlib", "requests", "yellowbrick"],
)
def score_best_num_class_gmm(
    pca_dataset: dsl.Input[dsl.Dataset],
    plot_gmm: dsl.Output[dsl.HTML],
):
    import requests
    import sys
    import matplotlib.pyplot as plt
    import pandas as pd


    feature_eng_url = "https://raw.githubusercontent.com/Carlos-ZRM/rag-agent-med-hoot/refs/heads/main/apps/functions/feature_eng.py"
    r = requests.get(feature_eng_url)
    with open("/tmp/feature_eng.py", "w") as f:
            f.write(r.text)
    sys.path.insert(0, "/tmp")
    from feature_eng import score_silueta

    df_pca = pd.read_pickle(pca_dataset.path)

    fig_gmm = score_silueta(df_pca, 2, 20, model_type="gmm")

    html = fig_gmm.to_html(include_plotlyjs="cdn", full_html=True)
    with open(plot_gmm.path, "w") as f:
        f.write(html)

@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "plotly", "matplotlib", "requests", "yellowbrick","jinja2"],
)
def clustering_knn(
    pca_dataset: dsl.Input[dsl.Dataset],
    num_class: int,
    plot_kmeans_table: dsl.Output[dsl.HTML],
    plot_kmeans_3d: dsl.Output[dsl.HTML],
):

    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px



    df_pca = pd.read_pickle(pca_dataset.path)
    from sklearn.cluster import KMeans
    
    KM_n = KMeans(n_clusters=num_class, n_init=100, init='k-means++', random_state=0)
    KM_n.fit(df_pca)

    df_kmn = df_pca.copy()
    df_kmn["cluster"] = KM_n.labels_

        
    styled_html = (
        df_kmn.style
        .background_gradient(cmap="Greens", axis=1)
        .to_html()
    )

    html = f"<html><body>{styled_html}</body></html>"
    with open(plot_kmeans_table.path, "w") as f:
        f.write(html)

    cluster_labels =  KM_n.labels_

    fig_3d = px.scatter_3d(
        df_kmn,
        x=df_pca['comp1'],
        y=df_pca['comp2'],
        z=df_pca['comp3'],
        color=cluster_labels,
        color_continuous_scale="Viridis",
        title=f"Clusters 3D - KMeans {num_class} Clusters",
        opacity=0.7,
        width=900,
        height=700,
    )

    fig_3d.update_traces(marker=dict(size=12,
                         line=dict(width=5,
                         color='Black')),
              selector=dict(mode='markers'))
    
    
    html = fig_3d.to_html(include_plotlyjs="cdn", full_html=True)
    with open(plot_kmeans_3d.path, "w") as f:
        f.write(html)


@dsl.component(
    base_image=base_image,
    packages_to_install=["pandas", "scikit-learn", "plotly", "matplotlib", "requests", "yellowbrick","jinja2"],
)
def clustering_gmm(
    pca_dataset: dsl.Input[dsl.Dataset],
    num_class: int,
    plot_gmm_table: dsl.Output[dsl.HTML],
    plot_gmm_3d: dsl.Output[dsl.HTML],
):

    import matplotlib.pyplot as plt
    import pandas as pd
    import plotly.express as px

    df_pca = pd.read_pickle(pca_dataset.path)
    from sklearn.mixture import GaussianMixture
    
    GMM_n = GaussianMixture(n_components=num_class, n_init=100, random_state=0)
    GMM_n.fit(df_pca)

    df_gmm = df_pca.copy()
    df_gmm["cluster"] = GMM_n.predict(df_pca)

        
    styled_html = (
        df_gmm.style
        .background_gradient(cmap="Blues", axis=1)
        .to_html()
    )

    html = f"<html><body>{styled_html}</body></html>"
    with open(plot_gmm_table.path, "w") as f:
        f.write(html)

    cluster_labels =  GMM_n.predict(df_pca)

    fig_3d = px.scatter_3d(
        df_gmm,
        x=df_pca['comp1'],
        y=df_pca['comp2'],
        z=df_pca['comp3'],
        color=cluster_labels,
        color_continuous_scale="Viridis",
        title="Clusters 3D - GMM",
        opacity=0.7,
        width=900,
        height=700,
    )

    fig_3d.update_traces(marker=dict(size=12,
                         line=dict(width=5,
                         color='Black')),
              selector=dict(mode='markers'))
    
    
    html = fig_3d.to_html(include_plotlyjs="cdn", full_html=True)
    with open(plot_gmm_3d.path, "w") as f:
        f.write(html)
    

@kfp.dsl.pipeline(name="Fifa dataset pipeline")
def fifa_pipeline(
        url: str = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        num_class: int = 5,
    ):

    data_prep_task = read_csv_from_url(
        url=url
    )
    eda_report_task = create_eda_report(
        dataset=data_prep_task.outputs["dataset"],
    )
    clean_data_task = clean_data(
        eda_dataset=eda_report_task.outputs["eda_dataset_output"],
    )
    feature_engineering_task = feature_engineering(
        tad_dataset=clean_data_task.outputs["tad_dataset"],
    )
    score_kmeans_task = score_best_num_class_kmeans(
        pca_dataset=feature_engineering_task.outputs["pca_dataset"],
    )
    score_gmm_task = score_best_num_class_gmm(
        pca_dataset=feature_engineering_task.outputs["pca_dataset"],
    )
    clustering_knn_task = clustering_knn(
        pca_dataset=feature_engineering_task.outputs["pca_dataset"],
        num_class=num_class,
    ).after(score_kmeans_task, score_gmm_task)

    clustering_gmm_task = clustering_gmm(
        pca_dataset=feature_engineering_task.outputs["pca_dataset"],
        num_class=num_class,
    ).after(score_kmeans_task, score_gmm_task)



if __name__ == "__main__":
    client = kfp.Client(
        host=kubeflow_endpoint,
        existing_token=bearer_token,
    )
    arguments = {
        "url": "https://s3-pipes-data.s3.us-east-2.amazonaws.com/data/data_fifa.csv",
        "num_class": 5,
    }
    client.create_run_from_pipeline_func(fifa_pipeline, arguments=arguments, experiment_name="fifa-example")
