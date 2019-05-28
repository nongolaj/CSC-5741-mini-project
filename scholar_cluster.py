from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import collections
import string
from nltk.corpus import stopwords
from textblob import Word
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

dataset = pd.read_csv("scholar_extracts.csv")

dataset = dataset[pd.notnull(dataset['description'])]

pd.options.mode.chained_assignment = None
dataset['description'] = dataset['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# remove punctuations
dataset['description'] = dataset['description'].str.replace('[^\w\s]','')

#remove stopwords

stop = stopwords.words('english')
dataset['description'] = dataset['description'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

#remove duplicates
dataset.drop_duplicates(subset ="description", keep = False, inplace = True)


dataset['description'] = dataset['description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
var_cleaned_abstract=dataset['description'].tolist()



var_etd_vectoriser = TfidfVectorizer(use_idf=True)
var_etd_vectoriser_data = var_etd_vectoriser.fit_transform(dataset["description"])
var_etd_vectoriser_data_tfidf = pd.DataFrame(var_etd_vectoriser_data.toarray(), columns=var_etd_vectoriser.get_feature_names())
var_etd_vectoriser_data[0]
print (var_etd_vectoriser_data[0])
var_etd_vectoriser_data_tfidf.columns
len(var_etd_vectoriser_data_tfidf.columns)



cluster_range = range( 1,10)
cluster_errors = []
for num_clusters in cluster_range:
   clusters = KMeans( num_clusters )
   clusters.fit(var_etd_vectoriser_data)
   cluster_errors.append( clusters.inertia_ )
clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )

plt.figure(figsize=(12,6))
plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )



for n_cluster in range(2, 30):
   kmeans = KMeans(n_clusters=n_cluster).fit(var_etd_vectoriser_data)
   label = kmeans.labels_
   sil_coeff = silhouette_score(var_etd_vectoriser_data, label, metric='euclidean')
   print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
   



def cluster_texts(text,clusters=3):
   var_etd_vectoriser = TfidfVectorizer(use_idf=True)
   var_etd_vectoriser_data = var_etd_vectoriser.fit_transform(text)
   km_model = KMeans(n_clusters=clusters)
   km_model.fit(var_etd_vectoriser_data)
 
   clustering = collections.defaultdict(list)
 
   for idx, label in enumerate(km_model.labels_):
      clustering[label].append(text[idx])
      
 
   return clustering
 

clusters = cluster_texts(var_cleaned_abstract,18)
pprint(dict(clusters))