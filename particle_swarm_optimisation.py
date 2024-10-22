import streamlit as st
import tensorflow as tf
from math import pi, e
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from io import BytesIO


def particle_swarm(current_position, current_value, best_position, best_value, current_velocity, iniertia_weight, global_accel, personal_accel, global_best_position, lim):
    new_velocity = 0
    new_velocity += iniertia_weight * current_velocity
    new_velocity += personal_accel * tf.random.uniform((n_particles, 2)) * (best_position - current_position)
    new_velocity += global_accel * tf.random.uniform((n_particles, 2)) * (global_best_position - current_position)
    
    new_position = current_position + new_velocity
    new_position = tf.clip_by_value(new_position, -lim, lim)
    new_value = f(new_position)
    
    best_position = tf.where(tf.expand_dims(new_value < current_value, axis=1), new_position, best_position)
    best_value = tf.where(new_value < current_value, new_value, best_value)
    
    global_best_position = tf.where(tf.reduce_min(new_value) < global_best_value, new_position[tf.argmin(new_value)], global_best_position)
    global_best_value.assign(tf.reduce_min(best_value))
    return new_position, best_position, new_value, best_value, new_velocity, global_best_position, global_best_value



# Define the test functions


@tf.function
def f1(X):
    m = 10
    dtype = X.dtype
    C = tf.constant([[4.0, 4.0], [1.0, 1.0], [8.0, 8.0], [6.0, 6.0], [3.0, 7.0], [2.0, 9.0], [5.0, 5.0], [8.0, 1.0], [6.0, 2.0], [7.0, 3.6]], dtype=dtype)
    beta = tf.constant([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5], dtype=dtype)
    result = tf.constant(0.0, dtype=dtype)
    for i in range(m):
        result += 1.0 / (beta[i] + tf.reduce_sum(tf.square(X - C[i]), axis=1))
    return -result

@tf.function
def f2(X):
    return tf.reduce_sum(tf.square(X), axis=1)

@tf.function
def f3(X):
    return tf.reduce_sum(tf.sin(X), axis=1)

# Add LaTeX equations for the labels


# Streamlit part to select the function
st.title("Particle Swarm Optimization Animation")
function_option = st.selectbox("Select a test function", ("Function 1:", "Function 2:", "Function 3:"))
st.latex(r'''
f_1(X) = -\sum_{i=1}^{10} \frac{1}{\beta_i + \sum_{j=1}^{2} (X_j - C_{ij})^2}
''')
st.latex(r'''
f_2(X) = \sum_{j=1}^{2} X_j^2
''')
st.latex(r'''
f_3(X) = \sum_{j=1}^{2} \sin(X_j)
''')
# Extract the function name from the selected option
function_option = function_option.split(":")[0]

if function_option == "Function 1":
    f = f1
elif function_option == "Function 2":
    f = f2
else:
    f = f3

# Create a meshgrid for plotting
st.text("Contour plot of your function")
lim = 4
X1, X2 = tf.meshgrid(tf.linspace(-lim, lim, 100), tf.linspace(-lim, lim, 100))
X_grid = tf.stack([tf.reshape(X1, [-1]), tf.reshape(X2, [-1])], axis=1)
Y_grid = f(X_grid)
Y_grid = tf.reshape(Y_grid, X1.shape)

# Plot the contour plot
fig, ax = plt.subplots()
contour = ax.contourf(X1, X2, Y_grid, levels=50, cmap='viridis')
fig.colorbar(contour)
st.pyplot(fig)
# Add sliders for the parameters
st.sidebar.title("PSO Parameters")
n_particles = st.sidebar.slider("Number of Particles", min_value=10, max_value=100, value=20, step=1)
n_frame = st.sidebar.slider("Number of Frames", min_value=10, max_value=100, value=30, step=1)
iniertia_weight = tf.Variable(st.sidebar.slider("Inertia Weight", min_value=0.1, max_value=1.0, value=0.95, step=0.05))
global_accel = tf.constant(st.sidebar.slider("Global Acceleration", min_value=0.1, max_value=1.0, value=0.5, step=0.05))
personal_accel = tf.constant(st.sidebar.slider("Personal Acceleration", min_value=0.1, max_value=2.0, value=1.0, step=0.1))
# Optimize button
if st.button('Optimize'):
    # Setting up the particles
    
    current_position = tf.random.uniform((n_particles, 2), -lim, lim)
    rand_vector = tf.random.uniform((n_particles, 2), minval=-1, maxval=2, dtype=tf.int32)
    rand_vector = tf.where(rand_vector == 2, 0, rand_vector)  # Ensure values are -1, 0, or 1
    rand_vector = tf.tensor_scatter_nd_update(rand_vector, [[tf.random.uniform([], 0, n_particles, dtype=tf.int32), tf.random.uniform([], 0, 2, dtype=tf.int32)]], [0])
    rand_vector = tf.cast(rand_vector, tf.float32) * lim
    current_position += rand_vector
    current_position = tf.clip_by_value(current_position, -lim, lim)
    best_values = []

    best_position = tf.Variable(current_position)
    current_value = tf.Variable(f(current_position))
    best_value = tf.Variable(current_value)
    current_velocity = tf.random.uniform((n_particles, 2), -0.1, 0.1)
    global_best_value = tf.Variable(tf.reduce_min(best_value))
    global_best_position = tf.Variable(best_position[tf.argmin(best_value)])


    fig = plt.figure(figsize=(10, 16))
    ax3d = fig.add_subplot(211, projection='3d')
    ax2d = fig.add_subplot(212)

    def init():
        ax3d.plot_surface(X1, X2, Y_grid, cmap='viridis', alpha=0.6)
        scat = ax3d.scatter(current_position[:, 0], current_position[:, 1], current_value, color='blue')
        ax2d.set_xlim(0, 30)
        ax2d.set_ylim(tf.reduce_min(best_value).numpy() - 1, tf.reduce_max(best_value).numpy() + 1)
        line, = ax2d.plot([], [], 'r-')
        return scat, line

    def update(frame):
        global current_position, best_position, current_value, best_value, current_velocity, global_best_position, best_values,n_frame,iniertia_weight,global_accel,personal_accel

        if frame == 0:
            best_values = []
            current_position = tf.random.uniform((n_particles, 2), -lim, lim)
            rand_vector = tf.random.uniform((n_particles, 2), minval=-1, maxval=2, dtype=tf.int32)
            rand_vector = tf.where(rand_vector == 2, 0, rand_vector)  # Ensure values are -1, 0, or 1
            rand_vector = tf.tensor_scatter_nd_update(rand_vector, [[tf.random.uniform([], 0, n_particles, dtype=tf.int32), tf.random.uniform([], 0, 2, dtype=tf.int32)]], [0])
            rand_vector = tf.cast(rand_vector, tf.float32) * lim
            current_position += rand_vector
            current_position = tf.clip_by_value(current_position, -lim, lim)
            best_position = tf.Variable(current_position)
            current_value = tf.Variable(f(current_position))
            best_value = tf.Variable(current_value)
            current_velocity = tf.random.uniform((n_particles, 2), -0.1, 0.1)
            global_best_position = tf.Variable(best_position[tf.argmin(best_value)])
            color = 'blue'
        else:
            color = 'red'

        ax3d.clear()
        ax3d.plot_surface(X1, X2, Y_grid, cmap='viridis', alpha=0.6)
        scat = ax3d.scatter(current_position[:, 0], current_position[:, 1], current_value, color=color, s=25)
        ax3d.set_xlabel('X1')
        ax3d.set_ylabel('X2')
        ax3d.set_zlabel('Function value')
        ax3d.set_title(f'3D surface plot of the particles: iter = {frame}')

        current_position, best_position, current_value, best_value, current_velocity, global_best_position, global_best_value = particle_swarm(
            current_position, current_value, best_position, best_value, current_velocity, iniertia_weight, global_accel, personal_accel, global_best_position, lim
        )

        best_values.append(tf.reduce_min(best_value).numpy())
        ax2d.clear()
        ax2d.plot(range(frame + 1), best_values, 'r-')
        ax2d.set_xlim(0, n_frame)
        ax2d.set_ylim(tf.reduce_min(best_value).numpy() - 1, tf.reduce_max(best_value).numpy() + 1)
        ax2d.set_xlabel('Iteration')
        ax2d.set_ylabel('Best Value')
        ax2d.set_title('Best Value vs Iteration')

        return scat,

    ani = FuncAnimation(fig, update, frames=n_frame, init_func=init, blit=True, repeat=False)

    ani_html = ani.to_jshtml()
    st.components.v1.html(f"<div style='display: flex; justify-content: left;'>{ani_html}</div>", height=3000, width=1000)

