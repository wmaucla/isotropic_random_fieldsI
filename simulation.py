import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from numpy.random import normal, uniform
from scipy.special import jv


# ========================== Core App Code ==========================

st.title('Series Representations and Simulation of Isotropic Random Fields in the Euclidean Space')

st.write("As given in the paper, we simulate equation 3.7")
st.latex(r''' Z(x) = \sqrt(2) * \sum_{n=0}^{\infty}J_n(rV_n)  cos(n\theta+2 \pi U_n), x=(rcos\theta, rsin\theta) ''')

# Settings 
st.subheader('Simulation Settings')
m = st.slider('m', min_value=5, max_value=100, value=20) 
r = st.slider('r', min_value=0, max_value=20, value=(3, 6)) 
simulation_number = st.number_input('Number n', value=200)

# Functions - For reference please see notebook  
def func_v(x):
    return 2 * np.sqrt(-np.log(x))

def generate_function(n, r, theta):
    n_random_uniform = uniform(size=(n+1))
    n_random_v_uniform = uniform(size=(n+1))
    
    return np.sqrt(2) * sum(
            [
                jv(i, r*func_v(n_random_v_uniform[i])) * \
                np.cos(i*theta+2*np.pi*n_random_uniform[i])
                for i in range(0, n+1, 1)
            ]
        )


r_list = np.linspace(r[0], r[1], m)
theta_list = np.linspace(0, np.pi * 2, m)

R, T = np.meshgrid(r_list, theta_list)
X, Y = R*np.cos(T), R*np.sin(T) # express in polar coordinates

# now calculate z for each r, theta

z_list = []
for theta in theta_list:
    for r in r_list:
        z_list.append(generate_function(simulation_number, r, theta))
        
Z = np.reshape(z_list, (m, m))

st.subheader('Create Visualization')
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_traces(contours_z=dict(
    show=True, usecolormap=True,
    highlightcolor="limegreen",
    project_z=True))
fig.update_layout(title='Isotrophic Random Fields', width=600, height=600, autosize=True,
                  margin=dict(l=30, r=50, b=65, t=30))
st.plotly_chart(fig, use_container_width=True)



