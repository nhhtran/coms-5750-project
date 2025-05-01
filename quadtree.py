import math
from astar import Node, Heap

class QuadTreeNode:
    """
    Pure-Python quadtree node for 2D list-based grids.
    """
    def __init__(self, x, y, w, h, grid):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.grid = grid
        self.children = []
        self.is_leaf = True

    def subdivide(self):
        """Recursively subdivide until the region is homogeneous or atomic (1x1)."""
        first = self.grid[self.y][self.x]
        homogeneous = True
        for i in range(self.y, self.y + self.h):
            for j in range(self.x, self.x + self.w):
                if self.grid[i][j] != first:
                    homogeneous = False
                    break
            if not homogeneous:
                break
        # Stop if homogeneous or region is atomic (1x1)
        if homogeneous or (self.w == 1 and self.h == 1):
            self.is_leaf = True
        else:
            self.is_leaf = False
            hw = self.w // 2
            hh = self.h // 2
            quadrants = [
                (self.x,       self.y,       hw,      hh),
                (self.x + hw,  self.y,       self.w-hw, hh),
                (self.x,       self.y + hh,  hw,      self.h-hh),
                (self.x + hw,  self.y + hh,  self.w-hw, self.h-hh),
            ]
            for (nx, ny, nw, nh) in quadrants:
                if nw > 0 and nh > 0:
                    child = QuadTreeNode(nx, ny, nw, nh, self.grid)
                    child.subdivide()
                    self.children.append(child)

    def get_leaves(self):
        if self.is_leaf:
            return [self]
        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())
        return leaves


def are_adjacent(a, b):
    """Check if two quadtree nodes share a side"""
    # Vertical side
    if a.x + a.w == b.x or b.x + b.w == a.x:
        if max(a.y, b.y) < min(a.y + a.h, b.y + b.h):
            return True
    # Horizontal side
    if a.y + a.h == b.y or b.y + b.h == a.y:
        if max(a.x, b.x) < min(a.x + a.w, b.x + b.w):
            return True
    return False


def build_adjacency(leaves):
    """Build adjacency map with Euclidean costs between region centers"""
    adjacency = {i: [] for i in range(len(leaves))}
    centers = [(leaf.x + leaf.w/2.0, leaf.y + leaf.h/2.0) for leaf in leaves]
    for i, leaf in enumerate(leaves):
        for j in range(i+1, len(leaves)):
            if are_adjacent(leaf, leaves[j]):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                cost = math.hypot(dx, dy)
                adjacency[i].append((j, cost))
                adjacency[j].append((i, cost))
    return adjacency, centers


def find_leaf_index(leaves, point):
    x, y = point
    for idx, leaf in enumerate(leaves):
        if leaf.x <= x < leaf.x + leaf.w and leaf.y <= y < leaf.y + leaf.h:
            return idx
    raise ValueError(f"Point {point} not contained in any quadtree leaf")


def _astar_regions(adjacency, centers, start_idx, end_idx):
    start_center = centers[start_idx]
    end_center = centers[end_idx]
    start_h = math.hypot(start_center[0] - end_center[0], start_center[1] - end_center[1])
    nodes = {start_idx: Node(start_idx, None, start_h, 0)}
    queue = Heap()
    queue.push(nodes[start_idx])
    closed = set()

    while queue.size > 0:
        current = queue.pop()
        ci = current.element
        if ci == end_idx:
            path = []
            node = ci
            while node is not None:
                path.append(centers[node])
                node = nodes[node].parent
            return path[::-1], closed
        closed.add(ci)
        for ni, cost in adjacency[ci]:
            if ni in closed:
                continue
            g = current.gcost + cost
            h = math.hypot(centers[ni][0] - end_center[0], centers[ni][1] - end_center[1])
            f = g + h
            if ni in nodes:
                if f < nodes[ni].fcost:
                    nodes[ni].fcost, nodes[ni].gcost = f, g
                    nodes[ni].parent = ci
                    queue.up(nodes[ni])
            else:
                node = Node(ni, ci, f, g)
                nodes[ni] = node
                queue.push(node)
    return [], closed


def astar_quadtree(grid, obstacleIds, start, end):
    """Quad-tree accelerated A* on a 2D Python list grid"""
    height = len(grid)
    width = len(grid[0])
    root = QuadTreeNode(0, 0, width, height, grid)
    root.subdivide()
    leaves = root.get_leaves()
    # Only keep leaves fully free of obstacles
    free_leaves = []
    for leaf in leaves:
        free = True
        for i in range(leaf.y, leaf.y + leaf.h):
            for j in range(leaf.x, leaf.x + leaf.w):
                if grid[i][j] in obstacleIds:
                    free = False
                    break
            if not free:
                break
        if free:
            free_leaves.append(leaf)
    adjacency, centers = build_adjacency(free_leaves)
    si = find_leaf_index(free_leaves, start)
    ei = find_leaf_index(free_leaves, end)
    return _astar_regions(adjacency, centers, si, ei)

# ---------------- NumPy version (similar logic) ----------------
import numpy as np

class QuadTreeNodeNP(QuadTreeNode):
    def __init__(self, x, y, w, h, arr):
        super().__init__(x, y, w, h, arr.tolist())
        self.arr = arr

    def subdivide(self):
        if self.w == 1 and self.h == 1:
            self.is_leaf = True
            return
        region = self.arr[self.y:self.y+self.h, self.x:self.x+self.w]
        first = region[0, 0]
        if (region == first).all():
            self.is_leaf = True
        else:
            self.is_leaf = False
            hw = self.w // 2
            hh = self.h // 2
            quadrants = [
                (self.x,       self.y,       hw,      hh),
                (self.x + hw,  self.y,       self.w-hw, hh),
                (self.x,       self.y + hh,  hw,      self.h-hh),
                (self.x + hw,  self.y + hh,  self.w-hw, self.h-hh),
            ]
            for (nx, ny, nw, nh) in quadrants:
                if nw > 0 and nh > 0:
                    child = QuadTreeNodeNP(nx, ny, nw, nh, self.arr)
                    child.subdivide()
                    self.children.append(child)


def astar_quadtree_np(arr, obstacleIds, start, end):
    """Quad-tree accelerated A* on a 2D NumPy array grid"""
    height, width = arr.shape
    root = QuadTreeNodeNP(0, 0, width, height, arr)
    root.subdivide()
    leaves = root.get_leaves()
    # Only keep leaves fully free of obstacles
    free_leaves = []
    for leaf in leaves:
        sub = arr[leaf.y:leaf.y+leaf.h, leaf.x:leaf.x+leaf.w]
        if not np.isin(sub, list(obstacleIds)).any():
            free_leaves.append(leaf)
    adjacency, centers = build_adjacency(free_leaves)
    si = find_leaf_index(free_leaves, start)
    ei = find_leaf_index(free_leaves, end)
    return _astar_regions(adjacency, centers, si, ei)

# ---------------- Sample usage ----------------
if __name__ == "__main__":
    grid = [
        [0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,0,1,1,1],
        [0,0,0,0,0,0,0,0,0,0],
    ]
    obstacles = {1}
    start=(0,0); end=(9,4)
    path, closed = astar_quadtree(grid, obstacles, start, end)
    print("QuadTree A* path (pure Python):", path)
    arr = np.array(grid)
    path_np, closed_np = astar_quadtree_np(arr, obstacles, start, end)
    print("QuadTree A* path (NumPy):", path_np)
