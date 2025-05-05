# TODO: Bad hardcode values
OBSTACLE_ID = 0
MAX_DEPTH = 10
MIN_SUBDIVIDE_AREA = 20

class RegionData:
    def __init__(self, bbox, is_free):
        self.bbox = bbox
        self.free = is_free


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

# -------------------- SUBDIVISION SATISFACTION CONDITIONS ------------------------- #

def is_bound_closed(grid, bbox):
    # if bounds are all obstacle, return true.
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])

    for x in range(xmin, xmax + 1):
        if (grid[ymin][x] != OBSTACLE_ID):
            return False

    for x in range(xmin, xmax + 1):
        if (grid[ymax][x] != OBSTACLE_ID):
            return False

    for y in range(ymin + 1, ymax):
        if (grid[y][xmin] != OBSTACLE_ID):
            return False

    for y in range(ymin + 1, ymax):
        if (grid[y][xmax] != OBSTACLE_ID):
            return False

    return True


def is_region_free(grid, bbox):
    # if region is free of obstacles, return true
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if (grid[y][x] == OBSTACLE_ID):
                return (x, y)
    return (-1, -1)


def is_area_minimum(bbox):
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])

    return ((xmax - xmin) * (ymax - ymin)) <= MIN_SUBDIVIDE_AREA

# ---------------------------------------------------------------------------------- #

def subdivide(quadtree, bbox, grid, depth_limit, cached_points):
    r = (int(bbox[0]), int(bbox[1])) == (int(bbox[2]), int(bbox[3])) 
    if (r): return None
    
    node_id = len(quadtree.nodes)
    quadtree.nodes.append([])

    # SUBDIVISION SATISFIED CONDITION CHECK
    cach_point = (-1, -1)

    if (depth_limit == 0 or is_area_minimum(bbox)):
        is_obs = (len(cached_points) > 0) or (is_region_free(grid, bbox) != (-1, -1)) 
        quadtree.data.append(RegionData(bbox, not is_obs))
        return node_id
    elif (is_bound_closed(grid, bbox)): # TODO: CHECK TO SEE IF START AND END POINTS ARE IN THERE
        quadtree.data.append(RegionData(bbox, False))
        return node_id
    else:
        if (len(cached_points) == 0):
            cach_point = is_region_free(grid, bbox)
            if (cach_point == (-1, -1)):
                quadtree.data.append(RegionData(bbox, True))
                return node_id

    if (cach_point != (-1, -1)):
        cached_points.append(cach_point)
    # -------------------------------------

    cent = midpoint(bbox[0], bbox[1], bbox[2], bbox[3])
    
    tl = [p for p in cached_points if p[0] < cent[0] and p[1] < cent[1]]
    tr = [p for p in cached_points if p[0] > cent[0] and p[1] < cent[1]]
    bl = [p for p in cached_points if p[0] < cent[0] and p[1] > cent[1]]
    br = [p for p in cached_points if p[0] > cent[0] and p[1] > cent[1]]

    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(cent[0], bbox[1], bbox[2], cent[1]), grid, depth_limit - 1, tr))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(bbox[0], cent[1], cent[0], bbox[3]), grid, depth_limit - 1, bl))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(cent[0], cent[1], bbox[2], bbox[3]), grid, depth_limit - 1, br))
    quadtree.nodes[node_id].append(subdivide(quadtree, make_box(bbox[0], bbox[1], cent[0], cent[1]), grid, depth_limit - 1, tl))

    return node_id


def build_tree(grid, max_depth, minx, miny, maxx, maxy):
    quadtree = QuadTree()
    #quadtree.node_id = subdivide(quadtree, make_box(minx, miny, maxx, maxy), grid, max_depth, [])
    stack = [(make_box(minx, miny, maxx, maxy), max_depth, [])]

    while (len(stack) > 0):
        param = stack.pop()
        bbox, depth, cached_points = param[0], param[1], param[2]
        if (int(bbox[0]), int(bbox[1])) == (int(bbox[2]), int(bbox[3])): continue
        
        #node_id = len(quadtree.nodes)
        #quadtree.nodes.append([])

        # SUBDIVISION SATISFIED CONDITION CHECK
        cach_point = (-1, -1)

        if (depth == 0 or is_area_minimum(bbox)):
            is_obs = (len(cached_points) > 0) or (is_region_free(grid, bbox) != (-1, -1)) 
            quadtree.data.append(RegionData(bbox, not is_obs))
            #return node_id
            continue
        elif (is_bound_closed(grid, bbox)): # TODO: CHECK TO SEE IF START AND END POINTS ARE IN THERE
            quadtree.data.append(RegionData(bbox, False))
            #return node_id
            continue
        else:
            if (len(cached_points) == 0):
                cach_point = is_region_free(grid, bbox)
                if (cach_point == (-1, -1)):
                    quadtree.data.append(RegionData(bbox, True))
                    #return node_id
                    continue

        if (cach_point != (-1, -1)):
            cached_points.append(cach_point)
        # -------------------------------------

        cent = midpoint(bbox[0], bbox[1], bbox[2], bbox[3])
        
        tl = [p for p in cached_points if p[0] < cent[0] and p[1] < cent[1]]
        tr = [p for p in cached_points if p[0] > cent[0] and p[1] < cent[1]]
        bl = [p for p in cached_points if p[0] < cent[0] and p[1] > cent[1]]
        br = [p for p in cached_points if p[0] > cent[0] and p[1] > cent[1]]

        stack.append((make_box(cent[0], bbox[1], bbox[2], cent[1]), depth - 1, tr))
        stack.append((make_box(bbox[0], cent[1], cent[0], bbox[3]), depth - 1, bl))
        stack.append((make_box(cent[0], cent[1], bbox[2], bbox[3]), depth - 1, br))
        stack.append((make_box(bbox[0], bbox[1], cent[0], cent[1]), depth - 1, tl))
    
    return quadtree


def draw_rect_from_quadtree(grid, quadtree):
    for region in quadtree.data:
        r = region.bbox
        color = (255, 255, 0) if region.free else (50, 20, 255)
        grid = cv2.rectangle(grid, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, -1)
        grid = cv2.rectangle(grid, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 255), 1)


if __name__ == "__main__":
    import cv2
    import numpy as np
    import timeit

    ggrid = cv2.imread("blocky_path1.png", cv2.IMREAD_GRAYSCALE)
    _, bgrid = cv2.threshold(ggrid, 10, 255, cv2.THRESH_BINARY)
    
    quadtree = build_tree(bgrid, 10, 0, 0, bgrid.shape[0] - 1, bgrid.shape[1] - 1)

    cgrid = cv2.cvtColor(ggrid,cv2.COLOR_GRAY2RGB)

    draw_rect_from_quadtree(cgrid, quadtree)

    scale = 2
    cgrid = cv2.resize(cgrid, (cgrid.shape[1]*scale, cgrid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow('quadtree', cgrid)
    cv2.imwrite("quadtree_environment_partition.png", cgrid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    t = timeit.timeit("build_tree(bgrid, 10, 0, 0, bgrid.shape[0] - 1, bgrid.shape[1] - 1)", globals=globals(), number=15)
    print(t)

    print("Ok")