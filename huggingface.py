import requests
import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import plotly.express as px
import os
import dotenv 
from dotenv import load_dotenv
load_dotenv()



API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": "Bearer " + st.secrets("HUGGINGFACE_API")}

st.title("Seniment Model from Huggingface")
st.write("Model - j-hartmann/emotion-english-distilroberta-base")

text = st.text_input(label="Enter your sentence")

def query(text):
	response = requests.post(API_URL, headers=headers, json=text)
	return response.json()
	
output = query({
	"inputs": text,
})

json_data = output

labels = [item['label'] for item in json_data[0]]
scores = [item['score'] for item in json_data[0]]

data = {'Label': labels, 'Score': scores}
df = pd.DataFrame(data)

st.dataframe(df)

# CREATE BAR GRAPH
st.bar_chart(df, x='Label',y='Score')

# # Create a pie chart using matplotlib
# fig, ax = plt.subplots()
# ax.pie(df['Score'], labels=df['Label'], autopct='%1.1f%%',)
# ax.set_aspect('equal')
# ax.set_title("Scores Distribution")

# # Display the pie chart in Streamlit
# st.pyplot(fig)


# CREATE PIE CHART-----
fig = px.pie(df, values='Score', names='Label', title='Scores Distribution')
st.plotly_chart(fig)




