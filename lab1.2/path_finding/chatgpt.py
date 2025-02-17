import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Define a 5x5 grid with weighted connections (0 = obstacle, >0 = cost)
grid = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],  # 0 represents an obstacle (cannot be traversed)
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
    ]
)

# Define start and goal positions
start = (0, 0)
goal = (4, 4)


def a_star(grid, start, goal):

    rows, cols = grid.shape

    # Convert grid to graph (adjacency matrix)
    def grid_to_graph(grid):
        nodes = rows * cols
        graph = np.full(
            (nodes, nodes), np.inf
        )  # Initialize adjacency matrix with infinite cost

        def node_index(r, c):
            return r * cols + c  # Convert (row, col) to node index

        for r in range(rows):
            for c in range(cols):
                if grid[r, c] == 0:
                    continue  # Skip obstacles

                index = node_index(r, c)

                # Define possible movement directions (4-way)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] > 0:
                        neighbor_index = node_index(nr, nc)
                        graph[index, neighbor_index] = grid[nr, nc]  # Use weight

        return csr_matrix(
            graph
        )  # Convert to compressed sparse row format for efficiency

    graph = grid_to_graph(grid)

    start_index = start[0] * cols + start[1]
    goal_index = goal[0] * cols + goal[1]

    # Compute shortest path
    dist_matrix, predecessors = shortest_path(
        graph, directed=False, return_predecessors=True
    )

    # Reconstruct path from predecessors
    def reconstruct_path(predecessors, start, goal):
        path = []
        current = goal
        while current != start:
            path.append(current)
            current = predecessors[start, current]
            if current == -9999:  # No path found
                return None
        path.append(start)
        return path[::-1]  # Reverse path

    path = reconstruct_path(predecessors, start_index, goal_index)

    # Convert path indices back to grid coordinates
    if path:
        path_coords = [np.array([p // cols, p % cols], dtype=int) for p in path]
        print("Shortest Path:", path_coords)
        return path_coords
    else:
        print("No path found!")
        return []
