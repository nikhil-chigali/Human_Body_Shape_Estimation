def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two 3D points.

    Parameters:
    - point1: NumPy array of shape (3,) representing the coordinates of the first point.
    - point2: NumPy array of shape (3,) representing the coordinates of the second point.

    Returns:
    - distance: Euclidean distance between the two points.
    """
    return np.linalg.norm(point1 - point2)

# Example:
left_shoulder = labeled_body_parts['left_shoulder']
right_shoulder = labeled_body_parts['right_shoulder']

distance_between_shoulders = calculate_distance(left_shoulder, right_shoulder)
print(f'Distance between shoulders: {distance_between_shoulders} units')
