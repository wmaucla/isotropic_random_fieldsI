import itertools
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from numpy.random import normal, uniform
from scipy.special import jv


# ========================== Core App Code ==========================

st.title('Series Representations and Simulation of Isotropic Random Fields in the Euclidean Space')

st.write("For the example in table 1, where we simulate the equation:")
st.latex(r''' Z(x) = \sqrt(2) * \sum_{n=0}^{\infty}J_n(rV_n)  cos(n\theta+2 \pi U_n), x=(rcos\theta, rsin\theta) ''')

# Settings 
st.subheader('Simulation Settings')
m = st.slider('m', min_value=5, max_value=200, value=20) 
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

def calculate_all_zs(theta):
    return [generate_function(simulation_number, r, theta) for r in r_list]

with mp.Pool(mp.cpu_count()) as mpool:
    z_list = mpool.starmap(calculate_all_zs, zip(theta_list))
        
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


st.subheader('Example 3.3 (continued), Case I')
st.write("The following image comes from the paper, with the aforementioned reference")

def example_1(i, j):
    """
    This function takes in two positional arguments, i and j, which represents the corresponding 
    values in a m by m matrix (based on the value set in cell 4)
    """
    r = R[i, j]
    theta = T[i, j]
    
    z_output = []
    for i in range(simulation_number):
        y1 = -np.log(uniform(low=0, high=1, size=1))
        y2 = uniform(low=np.log(2), high=np.log(8), size=1)
        v1 = (y1 ** 0.5) * np.exp(-y2/2)
        w1 = uniform(low=0, high=1, size=1)
        u1 = uniform(low=0, high=1, size=1)
        Zi = jv(i, 2*r*np.sqrt(-v1 * np.log(w1))) * np.cos(i*theta+2*np.pi*u1)
        z_output.append(Zi)
    return np.sqrt(2) * np.sum(z_output)

def generate_z(function):
    z_list = []
    with mp.Pool(mp.cpu_count()) as p:
        z_list = p.starmap(function, itertools.product(range(m), range(m)))
    
    Z = np.reshape(z_list, (m, m))
    return Z

Z = generate_z(example_1)
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_traces(contours_z=dict(
    show=True, usecolormap=True,
    highlightcolor="limegreen"))
fig.update_layout(title='', autosize=False,
                  width=1000, height=1000,
                  margin=dict(l=65, r=50, b=70, t=80), scene=dict(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        zaxis=dict(showticklabels=False),
    ))
st.plotly_chart(fig, use_container_width=True)

st.write("Additional examples are omitted and should reference the repo for additional code")