import numpy as np

A = np.array([1,2,3])
B = np.array([-4,-5,-6])
C = np.array([5,5,5])

#calulate the cosine similarity
cosine_similarity = np.dot(A,B) / (np.linalg.norm(A) * np.linalg.norm(B))

print("Cosine similarity between A and B:-",cosine_similarity)

# Calculate the cosine similarity
cosine_similarity = np.dot(A, C) / (np.linalg.norm(A) * np.linalg.norm(C))

print("Cosine similarity between A and C:", cosine_similarity)