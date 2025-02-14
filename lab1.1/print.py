import numpy as np
import matplotlib.pyplot as plt

MAP_SIZE = 1000
grid = np.array([[' ' for i in range(MAP_SIZE)] for j in range(MAP_SIZE)])


# np.empty(shape=(MAP_SIZE, MAP_SIZE), dtype=str)
pos = np.array([MAP_SIZE//2, MAP_SIZE//2], dtype=int)

grid[*pos] = 'x'

# Define color mapping
color_map = {' ': 0, 'x': 1, 'o': 2}  # Assign numerical values

# Convert char array to numerical values
num_array = np.vectorize(color_map.get)(grid)

# Plot and save the image
plt.figure(figsize=(4, 4))
plt.imshow(num_array, cmap='plasma', interpolation='nearest')
plt.axis('off')  # Hide axes
plt.savefig("char_array_image.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()
