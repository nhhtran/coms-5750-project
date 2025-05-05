import numpy as np
import cv2
from astar import astar
import timeit

def draw_path(grid, path, color):
    ipath = []

    for p in path:
        ipath.append((int(p[0]), int(p[1])))

    for i in range(1, len(ipath)):
        cv2.line(grid, ipath[i - 1], ipath[i], color, 1)


def random_walk_path(grid, max_waypoints, empty, start, end):
    waypoints = []
    waypoints.append(start)
    domain = int(len(grid) / max_waypoints)
    indomain = 1
    for _ in range(max_waypoints):
        waypoints.append(np.random.randint(0, indomain * domain, (2)))
        indomain += 1
    waypoints.append(end)

    draw_path(grid, waypoints, empty)


def gen_square_problem_grid(grid_size, start, end):
    empty = 1
    obstacle = 0

    prob_empty = 0.3

    ngrid = np.random.rand(grid_size, grid_size) < prob_empty
    ngrid = ngrid.astype(np.uint8)

    max_waypoints = grid_size // 10

    random_walk_path(ngrid, max_waypoints, empty, start, end)

    return ngrid


if __name__ == "__main__":
    np.random.seed(42)
    
    
    # Test cases
    r = 10
    for i in range(100, 5000, 100):
        n = 0
        print(i, end=" ")
        for j in range(r):
            grid_size = i

            start = (0,0)
            end = (grid_size - 1, grid_size - 1)
            obstacles = {0}
            
            #ngrid = cv2.imread("blocky_path1.png", cv2.IMREAD_GRAYSCALE) / 255

            ngrid = gen_square_problem_grid(grid_size, start, end)
            
            #path_np, closed_np = astar_quadtree_np(ngrid, obstacles, start, end)
            #print("QuadTree A* path (NumPy):", path_np)
            #t = timeit.Timer("astar(ngrid, obstacles, start, end)", globals=globals())
            path, closed = astar(ngrid, obstacles, start, end)
            n += len(closed)
            print(len(closed), end=" ")

        print()
    

    '''
    
    grid_size = 200

    start = (0,0)
    end = (grid_size - 1, grid_size - 1)
    obstacles = {0}
    
    #ngrid = cv2.imread("blocky_path1.png", cv2.IMREAD_GRAYSCALE) / 255

    ngrid = gen_square_problem_grid(grid_size, start, end)
    
    #path_np, closed_np = astar_quadtree_np(ngrid, obstacles, start, end)
    #print("QuadTree A* path (NumPy):", path_np)
    path_np, closed_np = astar(ngrid, obstacles, start, end)


    ngrid = np.array(ngrid) * 255
    ngrid = ngrid.astype(np.uint8)
    
    ngrid = cv2.cvtColor(ngrid,cv2.COLOR_GRAY2RGB)
    draw_path(ngrid, path_np, (0, 0, 255))

    # Optional: Scale up the image for better visibility (e.g., each "pixel" becomes 20x20)
    scale = 4
    ngrid = cv2.resize(ngrid, (ngrid.shape[1]*scale, ngrid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    
    print(len(closed_np))
    
    # Display the image
    cv2.imshow('astar found path', ngrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''    
