from schrutepy import schrutepy
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import nltk
from nltk.corpus import stopwords
from PIL import Image
import numpy as np
import collections
import pandas as pd
import streamlit as st
import nltk


st.title('Sentimental Analysis')
df = pd.read_csv('analisis_comments_tiktok.csv')
df.head()
df.shape

nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

df['scores'] = df['comment'].apply(lambda comment: sid.polarity_scores(comment))

df.head()

df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

df.head()

df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')

df.head()

st.dataframe(df,200,10)

from plotly.offline import init_notebook_mode,iplot
import plotly.graph_objects as go
import cufflinks as cf
init_notebook_mode(connected=True)

#labels
lab = df["comp_score"].value_counts().keys().tolist()
#values
val = df["comp_score"].value_counts().values.tolist()
trace = go.Pie(labels=lab, 
                values=val, 
                marker=dict(colors=['red']), 
                # Seting values to 
                hoverinfo="value"
              )
data = [trace]
layout = go.Layout(title="Sentiment Distribution")

fig = go.Figure(data = data,layout = layout)
iplot(fig)

# defining data
trace = go.Histogram(x=df['platform'],nbinsx=40,histnorm='percent')
data = [trace]
# defining layout
layout = go.Layout(title="platform Distribution")
# defining figure and plotting
fig = go.Figure(data = data,layout = layout)
iplot(fig)
st.plotly_chart(fig, use_container_width=True)

# defining data
trace = go.Histogram(x=df['post_type'],nbinsx=40,histnorm='percent')
data = [trace]
# defining layout
layout = go.Layout(title="post_type Distribution")
# defining figure and plotting
fig = go.Figure(data = data,layout = layout)
iplot(fig)
st.plotly_chart(fig, use_container_width=True)

# defining data
trace = go.Histogram(x=df['compound'],nbinsx=40,histnorm='percent')
data = [trace]
# defining layout
layout = go.Layout(title="compound Distribution")
# defining figure and plotting
fig = go.Figure(data = data,layout = layout)
iplot(fig)
st.plotly_chart(fig, use_container_width=True)