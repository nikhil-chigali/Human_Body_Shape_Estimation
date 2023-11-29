import numpy as np
import k3d

def render_smooth_3d_mesh(vertices):
    """
    Render a smooth 3D mesh from vertices using k3d.

    Parameters:
    - vertices: NumPy array of shape (N, 3), where N is the number of vertices.

    Returns:
    - plot: k3d Plot object.
    """
    plot = k3d.plot()

    # Reduce point size, set light blue color, and adjust opacity for smoother appearance
    points = k3d.points(vertices.astype(np.float32), point_size=0.02, shader='3dSpecular', color=0xADD8E6, opacity=0.8)


    # Add the points to the plot
    plot += points

    # Set camera position and orientation for better view
    plot.camera = [1.5, 1.5, 1.5, 0, 0, 0, 0, 1, 0]

    return plot


#example 3D mesh vertices of shape (10475, 3)

vertex = np.load('/Users/anekha/Documents/GitHub/Human_Body_Shape_Estimation/012.npy')



# Display the renderer in Visual Studio Code
plot = render_smooth_3d_mesh(vertex)
plot.display()
