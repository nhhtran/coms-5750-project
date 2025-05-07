import numpy as np
import timeit

def draw_path(grid, path, color, t1, t2):
    ipath = []

    for p in path:
        ipath.append((int(p[0]), int(p[1])))

    for i in range(1, len(ipath)):
        t = np.random.randint(t1, t2)   
        cv2.line(grid, ipath[i - 1], ipath[i], color, t)


def random_walk_path(grid, max_waypoints, empty, start, end, t1, t2):
    waypoints = []
    waypoints.append(start)
    domain = int(len(grid) / max_waypoints)
    indomain = 1
    for _ in range(max_waypoints):
        waypoints.append(np.random.randint(0, indomain * domain, (2)))
        indomain += 1
    waypoints.append(end)

    draw_path(grid, waypoints, empty, t1, t2)


def gen_square_problem_grid(grid_size, start, end):
    empty = 1
    obstacle = 0

    ngrid = np.zeros((grid_size, grid_size)) 
    ngrid = ngrid.astype(np.uint8)

    max_waypoints = grid_size // 10

    t1 = grid_size // 40
    t2 = grid_size // 15
    
    random_walk_path(ngrid, max_waypoints, empty, start, end, t1, t2)

    return ngrid


if __name__ == "__main__": 
    import cv2
    import time
    from quadtree_orcld import build_graph, build_tree
    from quadtree_astar import astar as qastar
    from astar import astar
    import math

    np.random.seed(42)
    
    # Test cases
    '''
    r = 10
    for i in range(100, 5000, 100):
        aspeed = 0
        print(i, end=" ")
        for j in range(r):
            grid_size = i

            start = (0,0)
            end = (grid_size - 1, grid_size - 1)
            ngrid = gen_square_problem_grid(grid_size, start, end)

            #depth = math.ceil(math.log2(max(ngrid.shape[0] - 1, ngrid.shape[1] - 1)))
            #quadtree = build_tree(ngrid, start, end, depth, ngrid.shape[0] - 1, ngrid.shape[1] - 1)
            #graph = build_graph(quadtree)
            
            
            t = timeit.timeit("astar(ngrid, {0}, start, end)", globals=globals(), number=1) 
            
            #print(len(closed), end=" ")
            aspeed += t

        print(aspeed / r)
    
    '''
    ngrid = gen_square_problem_grid(500, (0, 0), (499, 499))
    #_, bgrid = cv2.threshold(ngrid, 10, 255, cv2.THRESH_BINARY)
    
    #start = (ngrid.shape[0] - 1,0)
    #end = (0, ngrid.shape[1] - 1)

    #depth = math.ceil(math.log2(max(ngrid.shape[0] - 1, ngrid.shape[1] - 1)))
    #quadtree = build_tree(ngrid, start, end, depth, ngrid.shape[0] - 1, ngrid.shape[1] - 1)
    #graph = build_graph(quadtree)        
    #(path_np, closed) = qastar(graph, start, end)
    
    #(path_np, closed) = astar(bgrid, {0}, start, end)
    
    #print(len(path_np))

    ngrid = np.array(ngrid) * 255
    ngrid = ngrid.astype(np.uint8)
    
    ngrid = cv2.cvtColor(ngrid,cv2.COLOR_GRAY2RGB)
    #draw_path(ngrid, path_np, (0, 0, 255), 1, 2)

    cv2.imshow('astar found path', ngrid)
    # Optional: Scale up the image for better visibility (e.g., each "pixel" becomes 20x20)
    scale = 2
    ngrid = cv2.resize(ngrid, (ngrid.shape[1]*scale, ngrid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    
    # Display the image
    cv2.imwrite("example_test.png", ngrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
