import numpy as np
import plotly.graph_objects as go

def render_3d_mesh_plotly(smplx_vertices):
    """
    Render a 3D mesh from vertices using plotly.

    Parameters:
    - vertices: NumPy array of shape (N, 3), where N is the number of vertices.

    Returns:
    - None (opens the 3D interactive plot in a browser).
    """
    fig = go.Figure(data=[go.Scatter3d(x=smplx_vertices[:, 0], y=smplx_vertices[:, 1], z=smplx_vertices[:, 2],
                                     mode='markers', marker=dict(size=3))])

    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

vertex = np.load('/Users/anekha/Documents/GitHub/Human_Body_Shape_Estimation/012.npy')
render_3d_mesh_plotly(vertex)
