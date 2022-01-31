import pandas as pd
import streamlit as st
import plotly as plt
st.write('Above Zero Chegg Sentimental Analysis report')
df=pd.read_csv('chegg_final.csv')
df.head() 
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
st.markdown('Total Sentiment Distribution')
layout = go.Layout(title="Sentiment Distribution")
fig = go.Figure(data = data,layout = layout)
iplot(fig)
st.plotly_chart(fig, use_container_width=True)
# defining data
trace = go.Histogram(x=df['platform'],nbinsx=40,histnorm='percent')
data = [trace]
# defining layout
st.markdown('The platform used by the users was Youtube')
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
st.markdown('The distribution is [-1,-0.5] negative')
st.markdown('The distribution is ]-0.5,0.5] neutral')
st.markdown('The distribution is [0.5,1.0] positive')
# defining data
trace = go.Histogram(x=df['influencer'],nbinsx=40,histnorm='percent')
data = [trace]
# defining layout
layout = go.Layout(title="Influencer Distribution")
# defining figure and plotting
fig = go.Figure(data = data,layout = layout)
iplot(fig)
st.plotly_chart(fig, use_container_width=True)