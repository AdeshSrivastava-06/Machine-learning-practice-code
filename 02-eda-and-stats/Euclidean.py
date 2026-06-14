"""the concept of euclidean is used in KNN"""
"""Distance from origin->Euclidean norm"""

import numpy as np

#define an n-dimensional array
A = np.array([1,2,3,4,5])

#calculate the Euclidean dis from the origin
distance = np.linalg.norm(A)

print("Euclidean distance from the origin:--",distance)


"""distance from one point(X1,Y1) to (X2,Y2) is euclidean distance"""

A = np.array([1,2,3,4,5])
B = np.array([6,7,8,9,10])

#calculate the difference vector 
difference = A - B
distance = np.linalg.norm(difference)

print("Euclidean distance between A and B:--",distance)

"""visulization"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import euclidean

# 1. Generate 5 random 3D vectors
vectors = np.random.rand(5, 3)

# 2. Assign a random class (0 or 1) to each vector
classes = np.random.randint(0, 2, 5)

# 3. Plot the vectors on a 3D Matplotlib graph
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for vec, cls in zip(vectors, classes):
    ax.quiver(0, 0, 0, vec[0], vec[1], vec[2], color=("r" if cls == 1 else "b"))



# 4. Get user input for a query point (3D vector)
query_vector = np.array(list(map(float, input("Enter the query point (3D vector) separated by space: ").split())))

# 5. Calculate the distance from the query vector to the 5 vectors and find the nearest neighbor
distances = [euclidean(query_vector, vec) for vec in vectors]
nearest_neighbor_index = np.argmin(distances)

# 6. Output the class of the nearest neighbor
print("The class of the nearest neighbor is:", classes[nearest_neighbor_index])

# 7. Plot the query vector with a different color
ax.quiver(0, 0, 0, query_vector[0], query_vector[1], query_vector[2], color="g")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")

plt.show()