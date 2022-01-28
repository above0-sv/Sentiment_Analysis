# Tratamiento de datos
# ==============================================================================
import numpy as np
import pandas as pd
import string
import re
# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#style.use('ggplot') or plt.style.use('ggplot')

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

st.title('Sentimental Analysis')
df = pd.read_csv('analisis_comments_tiktok.csv')
df.head()
def limpiar_tokenizar(comment):
    '''
    Esta función limpia y tokeniza el comment en palabras individuales.
    El orden en el que se va limpiando el comment no es arbitrario.
    El listado de signos de puntuación se ha obtenido de: print(string.punctuation)
    y re.escape(string.punctuation)
    '''
    # Se convierte todo el comment a minúsculas
    nuevo_comment = comment.lower()
    # Eliminación de páginas web (palabras que empiezan por "http")
    nuevo_comment = re.sub('http\S+', ' ', nuevo_comment)
    # Eliminación de signos de puntuación
    regex = '[\\!\\"\\#\\$\\%\\&\\\'\\(\\)\\*\\+\\,\\-\\.\\/\\:\\;\\<\\=\\>\\?\\@\\[\\\\\\]\\^_\\`\\{\\|\\}\\~]'
    nuevo_comment = re.sub(regex , ' ', nuevo_comment)
    # Eliminación de números
    nuevo_comment = re.sub("\d+", ' ', nuevo_comment)
    # Eliminación de espacios en blanco múltiples
    nuevo_comment = re.sub("\\s+", ' ', nuevo_comment)
    # Tokenización por palabras individuales
    nuevo_comment = nuevo_comment.split(sep = ' ')
    # Eliminación de tokens con una longitud < 2
    nuevo_comment = [token for token in nuevo_comment if len(token) > 1]
    
    return(nuevo_comment)

test = "Text mining"
print(test)
print(limpiar_tokenizar(comment=test))

# Se aplica la función de limpieza y tokenización a cada tweet
# ==============================================================================
df['comment_tokenizado'] = df['comment'].apply(lambda x: limpiar_tokenizar(x))
df[['comment', 'comment_tokenizado']].head()
# Unnest de la columna texto_tokenizado
# ==============================================================================
comment_tidy = df.explode(column='comment_tokenizado')
comment_tidy = comment_tidy.drop(columns='comment')
comment_tidy = comment_tidy.rename(columns={'comment_tokenizado':'token'})
comment_tidy.head(3)
# Unnest de la columna texto_tokenizado
# ==============================================================================
comment_tidy = df.explode(column='comment_tokenizado')
comment_tidy = comment_tidy.drop(columns='comment')
comment_tidy = comment_tidy.rename(columns={'comment_tokenizado':'token'})
comment_tidy.head(3)
# Palabras distintas utilizadas por cada autor
# ==============================================================================
print('----------------------------')
print('Palabras distintas por influencer')
print('----------------------------')
comment_tidy.groupby(by='influencer')['token'].nunique()
# Top 5 palabras más utilizadas por cada comment
# ==============================================================================
comment_tidy.groupby(['influencer','token'])['token'] \
 .count() \
 .reset_index(name='count') \
 .groupby('influencer') \
 .apply(lambda x: x.sort_values('count', ascending=False).head(5))
# Obtención de listado de stopwords del inglés
# ==============================================================================
stop_words = list(stopwords.words('english'))
# Se añade la stoprword: amp, ax, ex
stop_words.extend(("amp", "xa", "xe"))
print(stop_words[:10])
# Filtrado para excluir stopwords
# ==============================================================================
comment_tidy = comment_tidy[~(comment_tidy["token"].isin(stop_words))]
# Top 10 palabras por autor (sin stopwords)
# ==============================================================================
fig, axs = plt.subplots(nrows=4, ncols=1,figsize=(20, 30))
for i, autor in enumerate(comment_tidy.influencer.unique()):
    df_temp = comment_tidy[comment_tidy.influencer == autor]
    counts  = df_temp['token'].value_counts(ascending=False).head(10)
    counts.plot(kind='barh', color='firebrick', ax=axs[i])
    axs[i].invert_yaxis()
    axs[i].set_title(autor)

fig.tight_layout()
st.pyplot(fig)