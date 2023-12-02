def extract_body_parts(vertices):
    """
    Extract coordinates of labeled extremities from 3D mesh vertices.

    Parameters:
    - vertices: NumPy array of shape (N, 3), where N is the number of vertices.

    Returns:
    - body_parts: Dictionary containing labeled extremities and their coordinates.
    """
    # A predefined mapping of vertices to body parts
    body_part_mapping = {
        'head': 0,        # Replace with the actual vertex index for the head
        'left_shoulder': 1,
        'right_shoulder': 2,
        # Add more body parts as needed
    }

    body_parts = {}
    for part, vertex_index in body_part_mapping.items():
        body_parts[part] = vertices[vertex_index]

    return body_parts

#'smplx_vertices' is the 3D mesh vertices of shape (10475, 3)
labeled_body_parts = extract_body_parts(smplx_vertices)

# Print the coordinates of labeled extremities
for part, coordinates in labeled_body_parts.items():
    print(f'{part}: {coordinates}')
