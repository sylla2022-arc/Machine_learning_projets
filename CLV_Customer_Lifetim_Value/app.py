import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
#from tqdm import tqdm
from stqdm import stqdm
import time
import lifetimes as lt
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import ParetoNBDFitter, BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_probability_alive_matrix, plot_history_alive, plot_frequency_recency_matrix
from lifetimes.plotting import plot_incremental_transactions
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from yellowbrick.cluster import InterclusterDistance
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# Chargement des données
df = pd.read_excel('SuperMarket_Transaction_Data.xlsx')
#Partie segmention
X = pd.read_csv('data_segmentation.csv')
preprocessor = ColumnTransformer( [('scaler', StandardScaler(), 
                                     X.columns.tolist())
                                  ])

Model_KM = Pipeline([ ('preprocessor', preprocessor),
                     ('kmeans', KMeans(n_clusters = 3, random_state = 0))
                    ])
Model_KM.fit(X)
label_km = Model_KM.named_steps['kmeans'].labels_

centroids = Model_KM.named_steps['kmeans'].cluster_centers_
centroid_df = pd.DataFrame(centroids, columns= X.columns)
centroid_df['Cluster'] = ['Centroid 0', 'Centroid 1', 'Centroid 2'] 


X['Cluster'] = label_km
Data_clustering = X.groupby('Cluster').mean().reset_index()
Y = Data_clustering.copy()
clus = Data_clustering.pop("Cluster")
preprocessor_min = ColumnTransformer( [('scaler', MinMaxScaler(),  Data_clustering.columns.tolist())
                                  ]
                                    )
X_scaled = preprocessor_min.fit_transform(Data_clustering)
X_scaled = pd.DataFrame(X_scaled, index=Data_clustering.index, columns=Data_clustering.columns)
X_scaled["Cluster"] = clus
X_scaled_clusters = X_scaled.groupby("Cluster").mean().reset_index()

def plot_radars(data, group):

# Initialisation de classe go
    fig = go.Figure()

    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group]==k].iloc[:,1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster '+str(k)
        ))

    fig.update_layout(
        polar=dict(
        radialaxis=dict(
          visible=True,
          range=[0, 1]
        )),
        showlegend=True,
        title={
            'text': "Visualisation multidimensionelle des clusters",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=18)

    st.plotly_chart(fig)
    
def plot_dispersion_diagram(X):
    df_var = ['frequency', 'recency', 'T', 'monetary_value', 'proba_alive', 'Pred_future_trans']
    
    # Initialisation des colonnes
    col1, col2 = st.columns(2)

 
    for i, col in enumerate(stqdm(df_var, st_container=st.sidebar)):
        # Création du graphique avec Plotly Express
        plotly_fig = px.scatter(
            X,
            x=col,
            y='CLV_predict',
            title=f'Diagr. en dispersion : {col}',
            labels={col: col, 'CLV_predict': 'CLV_predict'},
            color  = X['Cluster'].map({0:'Cluster 0', 1:'Cluster 1', 2:' Cluster 2'}),
            height=450,
            width=400)
  
        plotly_fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.9,
        xanchor="right",
        x= 0.9
    ))
        plotly_fig.update_layout(title_x=0.2)
        plotly_fig.update_layout(showlegend=True)
        plotly_fig.update_traces(
                marker_coloraxis=None
            )
        # Affichage du graphique dans la colonne appropriée
        if i % 2 == 0:
            col1.plotly_chart(plotly_fig)
        else:
            col2.plotly_chart(plotly_fig)

def plot_violin_diagram(X):
    df_var = ['frequency', 'recency', 'T', 'monetary_value', "proba_alive",
              'CLV_predict', 'Val_moy_attendu', "Pred_future_trans",
                ]
    
    # Initialisation des colonnes
    col1, col2 = st.columns(2)
  
    
    for i, col in enumerate(stqdm(df_var, st_container=st.sidebar)):
        # Création du graphique avec Plotly Express
        plotly_fig = px.violin(
            X,
            y =col,
            x ='Cluster',
            title=f'Diagr. en violon : {col}',
            labels={col: col, 'CLV_predict': 'CLV_predict'},
            #color  = 'Cluster',
            
            height=400,
            width=350)
        
        plotly_fig.update_layout(title_x=0.2)
        # Affichage du graphique dans la colonne appropriée
        if i % 2 == 0:
            col1.plotly_chart(plotly_fig)
        else:
            col2.plotly_chart(plotly_fig)

def plot_dispersion_diagram_3d(X):
    df_var = ['frequency', 'recency', 'T', 'monetary_value', 'proba_alive', 'Pred_future_trans']
    
    col1, col2 = st.columns(2)

    # Sélection de quelques combinaisons importantes de variables
    combinaisons = [
        ('frequency', 'recency', 'CLV_predict'),
        ('monetary_value', 'T', 'CLV_predict'),
        ('proba_alive', 'Pred_future_trans', 'CLV_predict'),
        ('monetary_value', 'frequency', 'CLV_predict')
    ]

    for idx, (x_var, y_var, z_var) in enumerate(combinaisons):
        # Création du graphique en 3D avec Plotly Express
        plotly_fig = px.scatter_3d(
            X,
            x=x_var,
            y=y_var,
            z=z_var,
            title=f'Diagr. 3D : {x_var} vs {y_var} vs {z_var}',
            labels={x_var: x_var, y_var: y_var, z_var: z_var},
            color=X['Cluster'].map({0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}),
            height=450,
            width=400
        )

        plotly_fig.update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.9,
            xanchor="right",
            x=0.9
        ))
        plotly_fig.update_layout(title_x=0.0)
        plotly_fig.update_layout(showlegend=True)

        # Affichage du graphique dans la colonne appropriée
        if idx % 2 == 0:
            col1.plotly_chart(plotly_fig)
        else:
            col2.plotly_chart(plotly_fig)

def plot_density_diagram(X) :
    df_var = ['frequency', 'recency', 'T', 'monetary_value', 
              'proba_alive', 'Val_moy_attendu', "Pred_future_trans",
                'CLV_predict']
    
    # Initialisation des colonnes
    col1, col2 = st.columns(2)
    
    for i, col in enumerate(stqdm(df_var, st_container=st.sidebar)):
        # Création du graphique avec Plotly Express
        density_fig = px.histogram(
            X,
            x=col,
            color='Cluster',
            #marginal='rug',
            title=f'Diagramme de densité pour {col}',
            labels={col: col},
            histnorm='probability density',
            height=400,
            width = 408
            )
        
        density_fig.update_layout(title_x=0.2)
        # Affichage du graphique dans la colonne appropriée
        if i % 2 == 0:
            col1.plotly_chart(density_fig)
        else:
            col2.plotly_chart(density_fig)

def boxplot_diagram(X):
    df_var = ['frequency', 'recency', 'T', 'monetary_value',
              "CLV_predict" , 'Val_moy_attendu']
    
    # Initialisation des colonnes
    col1, col2 = st.columns(2)
  
    
    for i, col in enumerate(stqdm(df_var, st_container=st.sidebar)):
        # Création du graphique avec Plotly Express
        plotly_fig = px.box(
            X,
            y =col,
            x ='Cluster',
            title=f'Diagr. en boîte : {col}',
            labels={col: col, 'CLV_predict': 'CLV_predict'},
            #color  = 'Cluster',
            
            height=400,
            width=350)
        
        plotly_fig.update_layout(title_x=0.2)
        # Affichage du graphique dans la colonne appropriée
        if i % 2 == 0:
            col1.plotly_chart(plotly_fig)
        else:
            col2.plotly_chart(plotly_fig)

def distorsion_plot():
    distorsion = []
    for i in stqdm(range(1,8), st_container=st.sidebar) :
        km = KMeans(n_clusters = i, random_state = 0).fit(X)
        distorsion.append(km.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(1,8)),
                            y = distorsion
                        )
                )
    fig.update_layout(title_text = 'Methode de Coude', autosize = True,
                    xaxis_title = 'Nombre de cluster',
                    yaxis_title = 'Distorsion Score' )

    fig.update_layout(height = 400,
                    width = 670)
    st.plotly_chart(fig)
    #print(distorsion)

def plot_silhouettes_calinski():
    # Elbow method avec  differents metrics
    metrics = ["silhouette", "calinski_harabasz"]
    i = 0

    fig, axes = plt.subplots(nrows = 1, ncols = 2, sharex = True, figsize=(20,8))
    for m in metrics:
        kmeans_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("kelbowvisualizer", KElbowVisualizer(KMeans(init='k-means++',random_state=42), K=(2,7),
                                                metric = m,
                                                ax=axes[i]))
            ])
        kmeans_visualizer.fit(X)
        kmeans_visualizer.named_steps['kelbowvisualizer'].finalize()
        
        i+=1

    st.pyplot(fig)

def plot_qualite_cluster():
    # Créer la figure et les sous-graphiques
    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(14, 8))

    # Itérer sur les sous-graphiques et les nombres de clusters
    for ax, k in zip(axes.flatten(), range(2, 6)):
        # Silhouette Visualizer
        silhouette_visualizer = Pipeline([
            ("preprocessor", preprocessor),
            ("silhouettevisualizer", SilhouetteVisualizer(KMeans(n_clusters=k, random_state=42), 
                                                          ax=ax, colors='yellowbrick'))
        ])
        silhouette_visualizer.fit(X)
        silhouette_visualizer.named_steps['silhouettevisualizer'].finalize()

    plt.tight_layout()
    st.pyplot(fig)

list_numeric =  ['Qty', 'Rate', 'Tax','Total_Amt']

# Filtrage : retenir que les transactions non annulées
df_abr = df.loc[df["Qty"] < 0, :].sort_values(by ='Qty', ascending = True)
df.drop(index = df.loc[df["Qty"] < 0, :].index.to_list(), inplace = True)

# Calcul de la frq. rec et monetary_value
summar_data = summary_data_from_transaction_data(df, 'Cust_id', 'Txn_date', 'Total_Amt',
                                                  observation_period_end = '2018-12-31 00:00:00').reset_index()
summar_data_pos = summar_data[summar_data.frequency > 0]

def choice_model(key_name =  None, key_coef = None):
    penalizer_coef = [0.0, 0.01, 0.001, 0.0001]
    coef = st.select_slider('Choisissez la valeur de penalizer_coef', options = penalizer_coef, 
                            value = 0.001, key = key_coef)
    st.write(f'Valeur sélectionnée pour penalizer_coef  : {coef}')
    
    model_type = st.selectbox('Choisissez le modèle', ['ParetoNBDFitter', 'BetaGeoFitter'], key = key_name)
    if model_type == 'ParetoNBDFitter':
        model = ParetoNBDFitter(coef)
    else:
        model = BetaGeoFitter(coef)

    model.fit(summar_data['frequency'], summar_data['recency'], summar_data['T'])
    return model

def val_monetaire_trans():
    gmf = GammaGammaFitter(penalizer_coef=0.1)
    gmf.fit(summar_data_pos["frequency"], summar_data_pos["monetary_value"])
    summar_data_pos["Val_mon_attendu"] = np.round(gmf.conditional_expected_average_profit(summar_data_pos["frequency"], summar_data_pos["monetary_value"]), 2)
    return summar_data_pos

def predic_clv():
    model = choice_model(key_name =  'ket_uni_pena', key_coef = 'coef_pena')
    gmf = GammaGammaFitter(penalizer_coef=0.01)
    gmf.fit(summar_data_pos["frequency"], summar_data_pos["monetary_value"])
    summar_data_pos["CLV_predict_Brut"] = np.round(gmf.customer_lifetime_value(
        model,
        frequency=summar_data_pos['frequency'],
        recency=summar_data_pos['recency'],
        T=summar_data_pos['T'],
        monetary_value=summar_data_pos["monetary_value"],
        time=12,  # projection sur 12 mois
        freq="D",
        discount_rate=0.01
    ), 2)
    return summar_data_pos

def predic_fut_trans(model=None):
    t_value = [15, 30 , 45, 90]
    value_time = st.select_slider('Choisissez la valeur du temps t', options =
                                   t_value, 
                                   value = 15)
    summar_data['Pred_future_trans'] = np.round(model.conditional_expected_number_of_purchases_up_to_time(t=value_time, 
                                                    frequency=summar_data['frequency'], 
                                                    recency=summar_data['recency'],
                                                      T=summar_data['T']), 2)
    summar_freq_pos_trans_fut = summar_data[summar_data.frequency > 0]
    summar_freq_pos_trans_fut.sort_values(by = 'Pred_future_trans', ascending = True)
    return summar_freq_pos_trans_fut

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.sidebar.title('Customer lifetime Value')
    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #0074D9;
        }
    </style>
    """, unsafe_allow_html=True)
    pages = ['Conexte', 'Dataset', 'Statistique et Visualisations','Data préparation',
              'Predictions', 'CVL Brute Prediction' ,'Calcul CVL Net', "Segmentation", 
              'ClusterVisualisation', 'Analyse des clusters', 'Conclusion']
    st.write(f":blue[Auteur: M. SYLLA], [Linkedin](https://www.linkedin.com/in/mahamadou-sylla/)")
    pages = st.sidebar.radio('Choississez une option:', pages)

    if pages == 'Conexte':
        st.markdown("""#### Problème : Quelle est la valeur de vie de vos clients?

Lorsqu'on gére une entreprise, suivre et comprendre chaque client individuellement devient pénible et coûteux. Ici, nous allons voir une stratégie pour optimiser votre portefeuille de clients et mettre en place une bonne stratégie de marketing pour répérer les types de clients afin de les fidéliser davantage et maximiser votre rentabilité.

L'objectif : Prédire la valeur à vie du client en suivant les étapes suivantes:
- Développer des modèles de prédiction avancés pour estimer la valeur à vie des clients
- Identifier les segments de clients à haute valeur
- Et élaborer des stratégies de rétention et de fidélisation ciblées
        """)

    elif pages == 'Dataset':
        st.write('Chargement et lecture du fichier en cours...')
        st.dataframe(df.head())
        if st.checkbox("Dimension du dataframe"):
            st.write(f"{df.shape}")
        if st.checkbox("Types des variables"):
            st.write(df.dtypes.value_counts())
        if st.checkbox("Valeurs manquantes:"):
            st.write(df.isna().sum())
        if st.checkbox('Duplicité des lignes :'):
            st.write(df.duplicated().sum())
            #st.write(":red[Les lignes dupliquées ont été supprimées!]")

    elif pages == 'Statistique et Visualisations':
        if st.checkbox("Statistique descriptive"):
            st.dataframe(df[list_numeric].describe().T)
        if st.checkbox("Distribution"):
            colname = st.selectbox('Choisissez une variable', list_numeric, key = 'dist')
            plt.rcParams['figure.figsize'] = (14, 5)
            fig, ax = plt.subplots()
            sns.histplot(df[colname], bins = 30, kde = True, color = 'red')
            plt.title(f"Histogramme des données : {colname}", fontsize = 14)
            st.pyplot(fig)
        if st.checkbox("Boxplot"):
            colname = st.selectbox('Choisissez une variable', list_numeric, key = 'box')
            plt.rcParams['figure.figsize'] = (14, 5)
            fig, ax = plt.subplots()
            sns.boxplot(df[colname], showfliers = False, showmeans = True, color = 'blue')
            plt.ylabel(f"{colname}", fontsize = 16)
            plt.title(f"Distribution en boites : {colname}", fontsize = 16)
            st.pyplot(fig)
            if st.checkbox('Commentaire'):
                st.markdown("""
Dans ce dataset :
* La plupart des quantités de produits achetés pour chaque transaction sont comprises entre 2 et 4 pour une 
valeur moyenne au alentour de 800$, ce qui suggère que la grande majorité des clients de ce magasin en ligne 
            sont des particuliers et peu de grossistes.
* Les tax sont également plutôt élevées car on a une moyenne de 200$ de taxe pour chaque transaction ce qui est énorme.

Comme 
$$
df[TotalAmt] = df[Rate] x df[Qty] + df[Tax]
$$

Nous allons donc supprimer les colonnes **Rate**, **Tax** et **Qty** et conserver que **Total_Amt**.
                """)
            if st.checkbox('Visualisation des ventes dans le temps'):
                df_categ = ['Product_Sub_category', 'Product_Category']
                col = st.selectbox('Choisissez une variable', df_categ)
                plt.rcParams['figure.figsize'] = (15, 3)
                df_copy = df.copy()
                df_copy.set_index("Txn_date", inplace = True)
                fig, ax = plt.subplots()
                df_copy.groupby(col)["Total_Amt"].mean().plot(style = '--o')
                plt.xticks(rotation = 85)
                plt.title(f"Evolution des ventes par : {col}", fontsize = 13)
                st.pyplot(fig)

    elif pages == "Data préparation":
        link_lifetime = "https://lifetimes.readthedocs.io/en/latest/"
        st.markdown(f"""
Dans cette partie, on va préparer nos données pour prédire les valeurs de vie des clients.
Tout d'abord, on va traiter les valeurs aberrantes puis utiliser la librairie [lifetimes]({link_lifetime}) pour extraire les fréquences, les récences, l'âge T des clients sur le site et leur valeur monétaire.

On va également calculer :
* La probabilité que le client fasse une nouvelle transaction dans x jours.
* La probabilité que le client soit en vie ou non.
* La valeur monétaire attendue pour le client s'il fait une nouvelle transaction.
        """)
        st.write("Début du traitement...")
        if st.checkbox("Afficher le tableau des valeurs aberrantes"):
            st.dataframe(df_abr.head())
            st.write(f":red[Dimension: {df_abr.shape}]")
        if st.checkbox('Afficher le tableau resultant après traitement'):
            summar_data['monetary_value'] = np.round(summar_data['monetary_value'], 2)
            st.dataframe(summar_data.head(3))
        if st.checkbox("Commentaire"):
            st.markdown("""
Quelques vocabulaires utiles à connaître pour la suite, car on fera, plusieurs fois, appel à eux.

* **Frequency** : Nombre d'achats par client
* **Recency** : Nombre de jours écoulé depuis le dernier achat
* **T** : Durée totale de l'observation pour chaque client
            """)

    elif pages == 'Predictions':
        link_lifetime = "https://lifetimes.readthedocs.io/en/latest/lifetimes.html"
        if st.checkbox('Commentaire'):
            st.markdown(f"""
Les deux modèles à utiliser pour calculer les probabilités sont [ParetoNBDFitter]({link_lifetime}) et [BetaGeoFitter]({link_lifetime}).
            """)
        if st.checkbox('Choix du modèle'):
            model = choice_model(key_name =  "proba_1", key_coef = 'prob')
            if st.checkbox('Afficher les paramètre du modèle'):
                st.write(model.params_)
        if st.checkbox('Prédire la probabilité que le client soit en vie'):
            model = choice_model(key_name =  'key_code', key_coef = 'code')
            summar_data['proba_alive'] = model.conditional_probability_alive(summar_data['frequency'], 
                                                                             summar_data['recency'], summar_data['T'])
            if st.checkbox("Afficher les probas"):
                st.dataframe(summar_data.head(5))

        #if st.checkbox('Visualiser le graphique correspondant'):
            #plt.rcParams['figure.figsize'] = (6, 4)
           # fig, ax = plt.subplots()
            #plot_probability_alive_matrix(model, title='Probabilité que le client soit en vie\n par sa fréq. et recence', 
                                         # xlabel='Fréquence historique', ylabel='Recence')
            #st.pyplot(fig)

        if st.checkbox('Afficher la matrice des récences et fréquences'):
            recenc_value = [100, 800 , 1000, 1200]
            value_rec = st.select_slider('Choisissez la valeur de max_recency', options = recenc_value, value = 1000)
            freq_value = [100, 165, 200, 365]
            value_freq = st.select_slider('Choisissez la valeur de max_frequency', options = freq_value, value = 365)
            plt.rcParams['figure.figsize'] = (6, 4)
            fig, ax = plt.subplots()
            plot_frequency_recency_matrix(model, max_recency = value_rec, max_frequency = value_freq, title = "Matrice des freq. et recence")
            st.pyplot(fig)

        if st.checkbox("Calculer la probabilité d'une future trans. dans t jours"):
            fut_trans = predic_fut_trans(model = model)
            st.dataframe(fut_trans.head())

        if st.checkbox("Prédire la valeur monétaire d'une nouvelle transaction"):
            st.markdown("""##### :red[Prediction de la valeur monétaire]
:red[Dans cette sous-section, on va prédire la valeur monétaire d'une nouvelle transaction pour chaque 
client à l'aide du modèle :blue[GammaGammaFitter] de la librairie lifetimes.]
            """)
            val_monetaire = val_monetaire_trans()
            st.dataframe(val_monetaire.head())

    elif pages == "CVL Brute Prediction":
        if st.checkbox("Prédire le CLV brut"):
            df_cvl = predic_clv()
            st.dataframe(df_cvl.head())
            st.write(f':red[Dimension : {df_cvl.shape}]')

            if st.checkbox('Visualiser les dispersions des points'):
                df_var = ['frequency', 'recency', 'T', 'monetary_value',
                ]
        
                # Initialisation des colonnes
                col1, col2 = st.columns(2)
            
                
                for i, col in enumerate(stqdm(df_var, st_container=st.sidebar)):  
                    # Création du graphique avec Plotly Express
                    plotly_fig = px.scatter(
                        df_cvl,
                        x  =col,
                        y ="CLV_predict_Brut" ,
                        title=f'Nuage des points ({col}, CLV Brut)',
                        labels={col: col, 'CLV_predict Brut': 'CLV_predict_Brut'},
                        #color  = ,
                        
                        height=400,
                        width=350)
                    
                    plotly_fig.update_layout(title_x=0.2)
                    # Affichage du graphique dans la colonne appropriée
                    if i % 2 == 0:
                        col1.plotly_chart(plotly_fig)
                    else:
                        col2.plotly_chart(plotly_fig)
        
        if st.checkbox('Commentaire'):
            st.markdown("""
:green[On observe sur cette sortie une certaine cohérence entre la valeur financière brute
que le client pourraît rapporter à cette entreprise et les différentes fonctionnalités intrinséques
telles que la valeur monétaire, la fréquence d'achat et la récence des clients qui sont en outre
des données reélles extraites du dataset.
On peut donc en déduire que le modèle s'est plutôt adapté aux données.]
                        
:red[Ps : La qualité de la prédiction dépend fortement de la valeur du penalizer_coef.]    
                        """)

    elif pages == 'Calcul CVL Net':
        df_cvl = predic_clv()
        df_cvl["CLV_Net"] = df_cvl['CLV_predict_Brut'] * 0.05
        if st.checkbox("Afficher le Customer Lifetime Value Net"):
            st.dataframe(df_cvl.head())
        if st.checkbox("Calculer les statistiques et visualiser le boxplot"):
            # Initialisation des colonnes
            col1, col2 = st.columns(2)
        
            for i in stqdm(range(1, 3), st_container=st.sidebar):
                # Création du graphique avec Plotly Express
                plotly_fig = px.box(
                    df_cvl,
                    x = "CLV_Net",
                    title=f'Diagr. en boîte CLV_Net',
                    height=400,
                    width=350)
                
                plotly_fig.update_layout(title_x=0.2)
                # Affichage du graphique dans la colonne appropriée
                if i % 2 == 0:
                    col1.dataframe(df_cvl["CLV_Net"].describe())
                else:
                    col2.plotly_chart(plotly_fig)

        if st.checkbox('Commentaire:'):
            st.markdown("""
Nous avons enfin réussi à calculer la valeur de profit net de chaque client. Cette valeur correspond en effet à la valeur financière totale qu'un client apporte à l'entreprise dans les 30 prochains jours avec une marge de profit de 5% c'est-à-dire après déduction des coûts.
            """)

    elif pages == "Segmentation":
        st.markdown("""#### Note :                  
On va appliquer le model kmeans pour segmenter les clients. Cela necessite beaucoup de 
préparation notamment la recherche du nombre de cluster optimal.
* **La méthode du coude** est un outil graphique utile pour estimer le nombre optimal de clusters k pour une tâche donnée. Intuitivement, on peut dire que, si $k$ augmente,
la distance intra-cluster (« distorsion ») va diminuer. En effet, les échantillons seront plus proches des centroïdes auxquels ils sont affectés.

* **Distorsion** : elle est calculée comme la moyenne des distances au carré des centres de cluster des clusters respectifs. En règle générale, la métrique de distance euclidienne est utilisée.
                    
* **Inertie** : C’est la somme des distances au carré des échantillons par rapport à leur centre de cluster le plus proche.
        """)
        if st.checkbox('Afficher le dataset pour la segmentation') :
            st.dataframe(X.head(3))
            st.write(f":red[Dimmension :{X.shape}]")
        if st.checkbox('Visualiser la courbe de torsion'):
            distorsion_plot()
        if st.checkbox("Notion de coefficient de silhouette") :
            st.markdown(r"""
##### Coefficient de silhouette
Pour un point $x$ donné, le coefficient de silhouette $s(x)$ permet d'évaluer si ce point appartient au « bon » cluster : est-il proche des points du cluster auquel il appartient ? Est-il loin des autres points ? Pour répondre à la première question, on calcule la distance moyenne de $x$ à tous les autres points du cluster $C_k$ auquel il appartient :

$$
a(x) = \frac{1}{|C_k| - 1} \sum\limits_{\substack{u \in C_k \\ u \neq x}}{d(u, x)}
$$

Pour répondre à la deuxième, on calcule la plus petite valeur que pourrait prendre $a(x)$, si $x$ était assigné à un autre cluster :

$$
b(x) = \min\limits_{\substack{l \neq k}} \sum\limits_{\substack{u \in C_l}}{d(u, x)}
$$

Si $x$ a été correctement assigné, alors $a(x) < b(x)$. Le coefficient de silhouette est donné par :

$$
s(x) = \frac{b(x) - a(x)}{\max(a(x), b(x))}
$$

Il est donc compris entre -1 et 1, et d'autant plus proche de 1 que l'assignation de $x$ à son cluster est satisfaisante.

Les valeurs proches de 0 indiquent des clusters qui se chevauchent. Les valeurs négatives indiquent généralement qu'un échantillon a été affecté au mauvais groupe, car un groupe différent est plus similaire.

Pour évaluer un clustering, on peut calculer son **coefficient de silhouette moyen**.
""")
        if st.checkbox('Visualiser les silhouettes scores et Calinski_Harabash score') :
            plot_silhouettes_calinski()
        if st.checkbox("Visualiser la qualité des clusters"):
            plot_qualite_cluster()
        if st.checkbox('Choix final du nombre de cluster'):
            st.markdown("""
A l'image de ce qui est détaillé plus haut, on en déduit que le nombre optimal de cluster
adapté à notre dataset est $k = 3$.                                         
                        """)
    
    elif pages == "ClusterVisualisation":
        if st.checkbox('Afficher le radar multidimensionnel des clusters') :
            plot_radars(X_scaled_clusters, 'Cluster')
            if st.checkbox("Voir l'interpretation") :
                st.markdown("""
* **Cluster 0 :** Ce groupe est composé de clients qui visitent fréquemment le site, effectuent des transactions récentes
et ont une forte probabilité de rester actifs. Leur valeur à vie et leur propension 
à effectuer de nouvelles transactions sont élevées. Ce sont les clients les plus précieux
et il est important de les récompenser pour renforcer leur fidélité et les maintenir engagés.

* **Cluster 1 :** Ce groupe se compose de clients qui n'ont pas effectué de transactions récentes,
ne visitent pas souvent le site et ont une faible probabilité de rester en tant que clients actifs. 
Cependant, leur propension à effectuer de nouvelles transactions est relativement fiable. Leur valeur
à vie est très élevée, mais ils nécessitent une approche proactive pour les ramener sur le site plus
 souvent, peut-être par le biais de remises ou de promotions.

* **Cluster 2 :** Ce groupe est constitué de clients qui visitent peu fréquemment le site, 
effectuent des transactions récentes et ont une probabilité fiable de rester actifs et de
faire de nouvelles transactions. Cependant, leur valeur à vie est très faible. Il est probable
qu'ils soient des clients mécontents, donc il est crucial de les ramener pour ne pas les perdre.
        """)

    elif pages == 'Analyse des clusters':
        if st.checkbox('Voir le diagramme en dispersion'):
            st.markdown("""
                <h1 style='text-align: center; color: green;'> ------Représentation en 2D------- 
                            </h1>""", unsafe_allow_html=True
                )
            plot_dispersion_diagram(X)
            st.markdown("""
                <h1 style='text-align: center; color: red;'> ------Représentation en 3D------- 
                            </h1>""", unsafe_allow_html=True
                )
            plot_dispersion_diagram_3d(X)
            if st.checkbox("Voir l'interprétation") :
                st.markdown("""
                <h1 style='text-align: center; color: grey; font-size: 16px;'>Une première observation consiste à affirmer que les 3 clusters formés
                        sont bien visibles et se distinctes les uns des autres.
                        Secundo, on constate la présence de quelques points isolés, ce qui laisse penser à des valeurs aberrantes ou anormales.
                            Peut-être une attention particulière pour ces points qui ne suivent le schema général.
                            Enfin, la visualisation en 3D apporte plus de visibilité et nous pouvons affirmer les clients 
                            du clusters 1(rose) ont une valeur de vie beaucoup plus importante que ceux des clusters 0 et 2.
                            </h1>""", unsafe_allow_html=True
                )

        if st.checkbox('Afficher les boîtes à moutaches') :
            boxplot_diagram(X)
            if st.checkbox('Analyse Textuelle'):
                # Comparaison avec les autres clusters
                st.write("#### Comparaison du cluster 0 avec le cluster 1")
                st.markdown("""
                    Le Cluster 0 se distingue par sa récence sur le site et sa fréquence de transactions plus élevée, mais avec une valeur financière 
                    légèrement inférieure par rapport aux autres clusters.
                    En comparaison, le Cluster 1 pourrait avoir une valeur financière plus élevée malgré une récence et 
                    une fréquence de transactions potentiellement plus faibles.
                """)
                # Recommandations actionnables
                st.write("#### Recommandations actionnables généralisées")
                st.markdown("""
                    - Cibler les nouveaux clients avec des offres spéciales ou des incitations pour encourager des transactions répétées.
                    - Mettre en place des programmes de fidélité ou des récompenses pour encourager les clients des clusters 0 et 2 à
                        augmenter leur valeur financière.
                    - Personnaliser les stratégies de marketing et de vente pour mieux répondre aux besoins et aux comportements
                      des clients des groupes 0 et 2.
                """)
                
        if st.checkbox("Voir le diagramme en violon"):
            plot_violin_diagram(X)
            if st.checkbox('Interprétation :' ):
                st.markdown("""
Les diagrammes en violon sont utiles pour comparer différents groupes. Deux observations importantes ressortent :

- On constate que la valeur prédite du CLV (Customer Lifetime Value), qui est l'indice le plus significatif,
montre une forte densité de probabilité des données entre 5 et 500 dollars pour l'ensemble des groupes.

- Une deuxième remarque importante concerne le cluster 1, composé des clients haut de gamme,
qui affiche une valeur financière bien plus élevée, dépassant souvent les 1000 dollars. Ces clients,
considérés comme "Premium", devraient être récompensés en leur offrant des offres personnalisées. Néanmoins,
il est essentiel de ne pas négliger les clients du cluster 2, auxquels il faut également proposer des récompenses
afin de les inciter à rejoindre la classe des clients "Premium".

                            """)

        if st.checkbox("Visualiser les diagrammes de densité"):
            plot_density_diagram(X)
            if st.checkbox("Voir l'interpretation", key = 'com') :
                st.markdown("""
               Ici, nous allons nous concentrer sur la forme des clusters de la variable CLV. Les distributions des données dans les
                 trois groupes suivent une allure gaussienne, ce qui signifie que la moyenne et la médiane sont quasiment identiques.
                    Cependant, les données du cluster 1 en rose sont plutôt plus larges que celles des autres, 
                        suggérant une population plus importante.  
                
                    """)
    
    elif pages == "Conclusion":
        pass

if __name__ == '__main__':
    main()
