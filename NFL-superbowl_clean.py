
import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import string
import re
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
# Preprocesado y modelado
# ==============================================================================
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')


st.title('NFL Tweets mining')
df = pd.read_csv('NFL_CLEAN.csv')
df.head()
st.dataframe(df)
df.shape

df.info

df.drop_duplicates()

df.dropna(thresh=2)

df.dropna(subset=['location', 'followers'])

df.dropna(subset=['username', 'description','location','followers'])

df.to_csv ('NFL_CLEAN.csv')





fig = px.scatter(df, x="followers", y="following", color='totaltweets')
fig.show()

fig = px.scatter(df, x="totaltweets", y="followers", trendline="ols")
fig.show()

fig = px.scatter(df, x="totaltweets", y="following", trendline="ols")
fig.show()

fig = px.scatter(df, x="totaltweets", y="retweetcount", trendline="ols")
fig.show()


def limpiar_tokenizar(text):
    '''
    Esta función limpia y tokeniza el texto en palabras individuales.
    El orden en el que se va limpiando el texto no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation)
    '''
    
    # Se convierte todo el texto a minúsculas
    
    new_text = text.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    new_text = re.sub('http\S+', ' ', new_text)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    new_text = re.sub(regex , ' ', new_text)
    # Eliminación de números
    new_text = re.sub("\d+", ' ', new_text)
    # Eliminación de espacios en blanco múltiples
    new_text = re.sub("\\s+", ' ', new_text)
    # Tokenización por palabras individuales
    new_text = new_text.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    new_text = [token for token in new_text if len(token) > 1]
    return(new_text)

test = "This is a sample o text clean"
print(test)
print(limpiar_tokenizar(text=test))

# Se aplica la función de limpieza y tokenización a cada tweet
# ==============================================================================
df['text_tokenizado'] = df['text'].apply(lambda x: limpiar_tokenizar(x))
df[['text', 'text_tokenizado']].head()

# Unnest de la columna texto_tokenizado
# ==============================================================================
df_tidy = df.explode(column='text_tokenizado')
df_tidy = df_tidy.drop(columns='text')
df_tidy = df_tidy.rename(columns={'text_tokenizado':'token'})
df_tidy.head(3)

print('Palabras by user')

print('----------------------------')
print('Words by Location')
print('----------------------------')
df_tidy.groupby(by='location')['token'].nunique()

print('----------------------------')
print('Words by username')
print('----------------------------')
df_tidy.groupby(by='username')['token'].nunique()

#Mean
# ==============================================================================
temp_df = pd.DataFrame(df_tidy.groupby(by = ["username"])["token"].count())
temp_df.reset_index().groupby("username")["token"].agg(['mean', 'std'])

#Mean
# ==============================================================================
temp_df = pd.DataFrame(df_tidy.groupby(by = ["location"])["token"].count())
temp_df.reset_index().groupby("location")["token"].agg(['mean', 'std'])

#Mean
# ==============================================================================
temp_df = pd.DataFrame(df_tidy.groupby(by = ["hashtags"])["token"].count())
temp_df.reset_index().groupby("hashtags")["token"].agg(['mean', 'std'])

# Top 5 words by user
# ==============================================================================
df_tidy.groupby(['username','token'])['token'] \
 .count() \
 .reset_index(name='count') \
 .groupby('username') \
 .apply(lambda x: x.sort_values('count', ascending=False).head(5))

import nltk
nltk.download('stopwords')
# Obtención de listado de stopwords del inglés
# ==============================================================================
stop_words = list(stopwords.words('english'))
# Se añade la stoprword: amp, ax, ex
stop_words.extend(("amp", "xa", "xe"))
print(stop_words[:10])

# Filtrado para excluir stopwords
# ==============================================================================
df_tidy = df_tidy[~(df_tidy["token"].isin(stop_words))]

# Top 10 words group by hashtag (sin stopwords)
# ==============================================================================
fig, axs = plt.subplots(nrows=10, ncols=1,figsize=(20, 40))
for i, autor in enumerate(df_tidy.hashtags.unique()):
    df_temp = df_tidy[df_tidy.hashtags == autor]
    counts  = df_temp['token'].value_counts(ascending=False).head(10)
    counts.plot(kind='barh', color='firebrick', ax=axs[i])
    axs[i].invert_yaxis()
    axs[i].set_title(autor)
fig.tight_layout()
fig = px.scatter(df, x="totaltweets", y="hashtags", color='retweetcount')
fig.show()
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(df, x="following", y="location", color='totaltweets')
fig.show()
st.plotly_chart(fig, use_container_width=True)

fig = px.scatter(df, x="totaltweets", y="text", color='retweetcount')
fig.show()
st.pyplot(fig)
st.plotly_chart(fig, use_container_width=True)


df.columns

"""Select features

"""

cdf = df[['following','followers','totaltweets','retweetcount']]
cdf.head(9)

"""Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set. This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that have been used to train the model. Therefore, it gives us a better understanding of how well our model generalizes on new data.

We know the outcome of each data point in the testing dataset, making it great to test with! Since this data has not been used to train the model, the model has no knowledge of the outcome of these data points. So, in essence, it is truly an out-of-sample testing.

Let's split our dataset into train and test sets. Around 80% of the entire dataset will be used for training and 20% for testing. We create a mask to select random rows using the np.random.rand() function:
"""

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

plt.scatter(train.following, train.totaltweets,  color='blue')
plt.xlabel("following")
plt.ylabel("total_tweets")
plt.show()

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['following','followers','totaltweets']])
y = np.asanyarray(train[['retweetcount']])
regr.fit (x, y)
# The coefficients
print ('Coefficients: ', regr.coef_)

"""   Retweetcount=-8.17 following -3.10 followers+1.666*10-5 totaltweets in a account
   
"""

# Commented out IPython magic to ensure Python compatibility.
y_hat= regr.predict(test[['following','followers','totaltweets']])
x = np.asanyarray(test[['following','followers','totaltweets']])
y = np.asanyarray(test[['retweetcount']])
print("Residual sum of squares: %.2f")
#       % np.mean((y_hat - y) ** 2))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))



"""
Let $\hat{y}$ be the estimated target output, y the corresponding (correct) target output, and Var be the Variance (the square of the standard deviation). Then the explained variance is estimated as follows:

$\texttt{explainedVariance}(y, \hat{y}) = 1 - \frac{Var{ y - \hat{y}}}{Var{y}}$\
The best possible score is 1.0, the lower values are worse.
"""