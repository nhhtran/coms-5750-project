class QuadTree:
    def __init__(self):
        self.bbox = None
        self.node_id = None
        self.data = []
        self.nodes = []

def make_box(minx, miny, maxx, maxy):
    return [minx, miny, maxx, maxy]

def midpoint(p1x, p1y, p2x, p2y):
    return ((p1x + p2x) / 2, (p1y + p2y) / 2)

def extend_box(box, p1x, p1y):
    if p1x < box[0]: box[0] = p1x
    if p1y < box[1]: box[1] = p1y
    if p1x > box[2]: box[2] = p1x
    if p1y > box[3]: box[3] = p1y

def make_bbox(points):
    box = make_box(points[0][0], points[0][1], points[0][0], points[0][1])
    (extend_box(box, points[i][0], points[i][1]) for i in range(1, len(points)))
    return box
    
def subdivide(quadtree, bbox, points, depth_limit):
    if (len(points) == 0): return None
    
    node_id = len(quadtree.nodes)
    quadtree.nodes.append([])

    quadtree.data.append(bbox)

    #leaf nodes
    if (len(points) == 1 or depth_limit == 0):  
        return node_id
    

    cent = midpoint(bbox[0], bbox[1], bbox[2], bbox[3])
    
    tl = [p for p in points if p[0] < cent[0] and p[1] < cent[1]]
    tr = [p for p in points if p[0] > cent[0] and p[1] < cent[1]]
    bl = [p for p in points if p[0] < cent[0] and p[1] > cent[1]]
    br = [p for p in points if p[0] > cent[0] and p[1] > cent[1]]

    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(cent[0], bbox[1], bbox[2], cent[1]), tr, depth_limit - 1))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(bbox[0], cent[1], cent[0], bbox[3]), bl, depth_limit - 1))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(cent[0], cent[1], bbox[2], bbox[3]), br, depth_limit - 1))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(bbox[0], bbox[1], cent[0], cent[1]), tl, depth_limit - 1))

    return node_id

def build_tree(points, max_depth, minx, miny, maxx, maxy):
    quadtree = QuadTree()
    quadtree.node_id = subdivide(quadtree, make_box(minx, miny, maxx, maxy), points, max_depth)
    return quadtree

def draw_rect_from_quadtree(grid, quadtree):
    for r in quadtree.data:
        grid = cv2.rectangle(grid, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 0), 1)
        #break

if __name__ == "__main__":
    import cv2
    import numpy as np

    grid_size = 200
    num_points = 1000

    points = np.random.randint(0, grid_size, (num_points, 2))
    
    grid = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
    
    pixel_color = np.array([255, 255, 255], dtype=np.uint8)
    
    for p in points:
        grid[p[0]][p[1]] = pixel_color

    quadtree = build_tree(points, 5, 0, 0, grid_size - 1, grid_size - 1)
    
    draw_rect_from_quadtree(grid, quadtree)

    scale = 4
    grid = cv2.resize(grid, (grid.shape[1]*scale, grid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('quadtree', grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    print("Ok")