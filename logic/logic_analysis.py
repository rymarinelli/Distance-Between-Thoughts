import numpy as np
from scipy.spatial.distance import cosine

def compute_consecutive_distances(embeddings):
    """
    Computes cosine distances between each step and the next.
    """
    return [cosine(embeddings[i], embeddings[i+1]) for i in range(len(embeddings) - 1)]

def logic_error_magnitude(embeddings, steps):
    """
    Returns (index, angle in degrees) for each step with phrases like 'wait, no' etc.
    """
    errors = []
    for i in range(1, len(steps) - 1):
        if any(x in steps[i].lower() for x in ["wait, no", "wait, but", "but, wait"]):
            try:
                vec1 = embeddings[i] - embeddings[i-1]
                vec2 = embeddings[i+1] - embeddings[i]
                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
                errors.append((i, np.degrees(angle)))
            except:
                continue
    return errors
