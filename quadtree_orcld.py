from collections import deque
import math
import numpy as np

# TODO: Bad hardcode values
OBSTACLE_ID = 0
MAX_DEPTH = 10
MIN_SUBDIVIDE_AREA = 1


def midpoint(p1x, p1y, p2x, p2y):
    return ((p1x + p2x) / 2, (p1y + p2y) / 2)


class RegionData:
    def __init__(self, level, bbox, is_free, lcld, num):
        self.bbox = bbox
        self.free = is_free
        self.num = num
        self.lcld = lcld 
        self.level = level
        self.center = midpoint(bbox[0], bbox[1], bbox[2], bbox[3])


class SubdivisionData:
    def __init__(self, level, bbox, cached_points, parent_lcld, child_pos, num):
        self.bbox = bbox
        self.cached_points = cached_points
        self.num = num
        self.parent_lcld = parent_lcld # L code level diff in directions [N, W, S, E]
        self.child_pos = child_pos
        self.child_lcld = [0, 0, 0, 0]
        self.level = level


class QuadTree:
    def __init__(self, encode_dirs):
        self.region_data = {}
        self.is_enclosed = False
        self.encode_dirs = encode_dirs
        self.subdivision_data = {} # num : data
        self.r = None
        self.tx = None
        self.ty = None


def make_box(minx, miny, maxx, maxy):
    return [math.ceil(minx), math.ceil(miny), math.floor(maxx), math.floor(maxy)]


def is_point_in_bbox(point, bbox):
    return point[0] >= bbox[0] and point[0] <= bbox[2] and point[1] >= bbox[1] and point[1] <= bbox[3]


def dilated_integer_addition(nq, ni, tx, ty):
    return (((nq | ty ) + (ni & tx )) & tx ) | (((nq | tx ) + (ni & ty )) & ty )


def interleave(x, y, size):
    encoded = np.uint32(0)
    mask = np.uint32(2**(size-1))
    for i in range(size):
        encoded = (encoded << 1) + ((y & mask) != 0)
        encoded = (encoded << 1) + ((x & mask) != 0)
        x, y = x << 1, y << 1
    return encoded


def compute_8directions(r):
    NW = interleave(~np.uint32(0),  np.uint32(1), r)
    N  = interleave( np.uint32(0),  np.uint32(1), r)
    NE = interleave( np.uint32(1),  np.uint32(1), r)
    W  = interleave(~np.uint32(0),  np.uint32(0), r)
    E  = interleave( np.uint32(1),  np.uint32(0), r)
    SW = interleave(~np.uint32(0), ~np.uint32(0), r)
    S  = interleave( np.uint32(0), ~np.uint32(0), r)
    SE = interleave( np.uint32(1), ~np.uint32(0), r)
    
    return [N, W, S, E, NW, NE, SE, SW]
    

# -------------------- SUBDIVISION SATISFACTION CONDITIONS ------------------------- #

def is_bound_closed(grid, bbox):
    # if bounds are all obstacle, return true.
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]

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
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    for y in range(ymin, ymax + 1):
        for x in range(xmin, xmax + 1):
            if (grid[y][x] == OBSTACLE_ID):
                return (x, y)
    return (-1, -1)


def is_area_minimum(bbox):
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    return ((xmax - xmin) * (ymax - ymin)) <= MIN_SUBDIVIDE_AREA

# ---------------------------------------------------------------------------------- #
def find_neighbor(quadtree, quadrant_num, direction):
    level_diff = quadtree.region_data[quadrant_num].lcld[direction]
    
    if (level_diff is not None):
        r = quadtree.r
        tx = quadtree.tx
        ty = quadtree.ty
        nd = quadtree.encode_dirs[direction]
        nq = np.uint32(quadrant_num)
        l = quadtree.region_data[quadrant_num].level
        if (level_diff < 0):
            s = np.uint32((r - l + abs(level_diff)) << 1)
            return dilated_integer_addition((nq >> s) << s, nd << s, tx, ty)
        else:
            return dilated_integer_addition(nq, nd << ((r - l) << 1), tx, ty)            

    return None


def test(quadtree):
    s = "111000" # 320
    np = int(s, 2)
    a = find_neighbor(quadtree, np, 2)
    #mq = find_neighbor(quadtree, 322, 2)
    #print("mq", mq)
    print(bin(a))


def build_tree(grid, start, end, max_depth, minx, miny, maxx, maxy):
    width = maxx - minx + 1
    height = maxy - miny + 1
    depth = min(max_depth, math.ceil(math.log2(max(width, height))))

    tx = np.uint32(1)
    ty = np.uint32(2)
    for _ in range(depth - 1):
        tx = (tx << 2) + 1
        ty = (ty << 2) + 2

    quadtree = QuadTree(compute_8directions(depth))
    quadtree.r = depth
    quadtree.tx = tx
    quadtree.ty = ty

    queue    = deque()

    #quadtree.node_id = subdivide(quadtree, make_box(minx, miny, maxx, maxy), grid, max_depth, [])
    s = SubdivisionData(np.uint32(0), make_box(minx, miny, maxx, maxy), [], [None] * 4, (), np.uint32(0))
    s.child_lcld = [None] * 4
    queue.append(s)
    
    while (len(queue) > 0):
        s = queue.popleft()
        if ((s.bbox[2] <= s.bbox[0]) or (s.bbox[3] <= s.bbox[1])): continue

        for d in s.child_pos:
            s.child_lcld[d] = s.parent_lcld[d] - 1 if (s.parent_lcld[d] is not None) else None
            
        # SUBDIVISION SATISFIED CONDITION CHECK
        is_subdivided = True
        cach_point = (-1, -1)
        if (depth == s.level or is_area_minimum(s.bbox)):
            is_obs = (len(s.cached_points) > 0) or (is_region_free(grid, s.bbox) != (-1, -1)) 
            quadtree.region_data[s.num << ((depth - s.level) << 1)] = RegionData(s.level, s.bbox, not is_obs, s.child_lcld, s.num << ((depth - s.level) << 1))
            is_subdivided = False
        elif (is_bound_closed(grid, s.bbox)): # TODO: CHECK TO SEE IF START AND END POINTS ARE IN THERE
            is_start_in = is_point_in_bbox(start, s.bbox)
            is_end_in   = is_point_in_bbox(end,   s.bbox)
            if (is_start_in != is_end_in):
                quadtree.is_enclosed = True
                return quadtree
            elif (is_start_in):
                queue = deque()
                queue.append(SubdivisionData(np.uint32(0), make_box(s.bbox[0] + 1, s.bbox[1] + 1, s.bbox[2] - 1, s.bbox[3] - 1), [], [None] * 4, (), np.uint32(0)))
                quadtree.region_data.clear()
            else:
                quadtree.region_data[s.num << ((depth - s.level) << 1)] = RegionData(s.level, s.bbox, False, s.child_lcld, s.num << ((depth - s.level) << 1))
            
            is_subdivided = False
        else:
            if (len(s.cached_points) == 0):
                cach_point = is_region_free(grid, s.bbox)
                if (cach_point == (-1, -1)):
                    quadtree.region_data[s.num << ((depth - s.level) << 1)] = RegionData(s.level, s.bbox, True, s.child_lcld, s.num << ((depth - s.level) << 1))
                    is_subdivided = False

        if (is_subdivided):
            for d in s.child_pos:
                opd = d + 2 if d + 2 < 4 else d - 2
                sibling_num = dilated_integer_addition(s.num, quadtree.encode_dirs[opd], tx, ty)
                quadtree.subdivision_data[sibling_num].child_lcld[d] += 1
                '''
                try:
                    quadtree.subdivision_data[sibling_num].child_lcld[d] += 1
                except:
                    k = dilated_integer_addition(np.uint32(1), quadtree.encode_dirs[1], tx, ty)
                    print("s.num", s.num, bin(s.num))
                    print("opd", opd)
                    print("sibling_num", sibling_num, bin(sibling_num))
                    print("k", k)
                    exit()
                '''
        else:
            continue

        if (cach_point != (-1, -1)):
            s.cached_points.append(cach_point)

        # -------------------------------------

        cent = midpoint(s.bbox[0], s.bbox[1], s.bbox[2], s.bbox[3])
        
        tl = [p for p in s.cached_points if p[0] < cent[0] and p[1] < cent[1]]
        tr = [p for p in s.cached_points if p[0] > cent[0] and p[1] < cent[1]]
        bl = [p for p in s.cached_points if p[0] < cent[0] and p[1] > cent[1]]
        br = [p for p in s.cached_points if p[0] > cent[0] and p[1] > cent[1]]
        #(self, bbox, cached_points, parent_lcld, child_pos, num)

        quadtree.subdivision_data[(s.num << 2) + 0] = SubdivisionData(s.level + 1, make_box(s.bbox[0], cent[1], cent[0], s.bbox[3]), bl, s.child_lcld, (2, 1), (s.num << 2) + 0)
        quadtree.subdivision_data[(s.num << 2) + 1] = SubdivisionData(s.level + 1, make_box(cent[0], cent[1], s.bbox[2], s.bbox[3]), br, s.child_lcld, (2, 3), (s.num << 2) + 1)
        quadtree.subdivision_data[(s.num << 2) + 2] = SubdivisionData(s.level + 1, make_box(s.bbox[0], s.bbox[1], cent[0], cent[1]), tl, s.child_lcld, (0, 1), (s.num << 2) + 2)
        quadtree.subdivision_data[(s.num << 2) + 3] = SubdivisionData(s.level + 1, make_box(cent[0], s.bbox[1], s.bbox[2], cent[1]), tr, s.child_lcld, (0, 3), (s.num << 2) + 3)

        queue.append(quadtree.subdivision_data[(s.num << 2) + 0])
        queue.append(quadtree.subdivision_data[(s.num << 2) + 1])
        queue.append(quadtree.subdivision_data[(s.num << 2) + 2])
        queue.append(quadtree.subdivision_data[(s.num << 2) + 3])
    
        #del quadtree.subdivision_data[s.num]

    return quadtree


class Node:
    def __init__(self, px, py, bbox, num):
        self.px = px
        self.py = py
        self.bbox = bbox
        self.num = num
    
    def get(self):
        return (self.px, self.py)

    def num_id(self):
        return self.num


def distance_nodes(node1, node2):
    return math.sqrt((node1.px - node2.px) * (node1.px - node2.px) + (node1.py - node2.py) * (node1.py - node2.py))


def build_graph(quadtree):
    V, E, W = {}, {}, {}
    
    for region in quadtree.region_data.values():
        if (region.free):
            V[region.num] = Node(region.center[0], region.center[1], region.bbox, region.num)
            E[region.num] = []
            W[region.num] = []

    for region in quadtree.region_data.values():
        for d in range(4):
            if (region.free and region.lcld[d] is not None and region.lcld[d] < 1):
                neighbor_region_num = find_neighbor(quadtree, region.num, d)
               
                if (quadtree.region_data[neighbor_region_num].free):
                    E[region.num].append(neighbor_region_num)
                    w = distance_nodes(V[region.num], V[neighbor_region_num])
                    W[region.num].append(w)
                    if (region.lcld[d] < 0):
                        E[neighbor_region_num].append(region.num)
                        W[neighbor_region_num].append(w)

    return (V, E, W)


def draw_rect_from_quadtree(grid, quadtree, start, end):
    print("num_regions:", len(quadtree.region_data))
    for region in quadtree.region_data.values():
        r = region.bbox
        color = (255, 255, 0) if region.free else (50, 20, 255)
        #grid = cv2.rectangle(grid, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), color, -1)
        grid = cv2.rectangle(grid, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 0, 255), 1)
        c = midpoint(r[0], r[1], r[2], r[3])
        w = r[2] - r[0]
        h = r[3] - r[1]

        lcld = []
        for l in region.lcld:
            if (l is None):
                lcld.append("#")
            else:
                lcld.append(l)

        #grid = cv2.putText(grid, str(bin(region.num)), (int(r[0]), int(r[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
        grid = cv2.putText(grid, str(lcld[0]), (int(c[0]), int(r[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
        grid = cv2.putText(grid, str(lcld[1]), (int(r[0]), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
        grid = cv2.putText(grid, str(lcld[2]), (int(c[0]), int(r[3])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
        grid = cv2.putText(grid, str(lcld[3]), (int(r[2] - 7), int(c[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
     
    grid[start[1]][start[0]] = (70, 150, 150)
    grid[end[1]][end[0]]     = (70, 150, 150)


def draw_edges(grid, graph):
    V = graph[0]
    E = graph[1]
    W = graph[2]
    k = {}
    for num in V.keys():
        for i in range(len(E[num])):
            edge_num = E[num][i]
            center = midpoint(V[num].px, V[num].py, V[edge_num].px, V[edge_num].py)
            center2 = midpoint(V[num].px, V[num].py, center[0], center[1])
            cv2.circle(grid, (int(center2[0]), int(center2[1])), 2, (255, 0, 255), 1)
            cv2.line(grid, (int(V[num].px), int(V[num].py)), (int(V[edge_num].px), int(V[edge_num].py)), (255, 0, 255), 1)
            grid = cv2.putText(grid, str(round(W[num][i], 2)), (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (199, 70, 70), 1, cv2.LINE_AA)
            cv2.circle(grid, (int(V[num].px), int(V[num].py)), 5, (0, 0, 255), -1)
            

if __name__ == "__main__":
    import cv2
    import timeit

    ggrid = cv2.imread("blocky_path2.png", cv2.IMREAD_GRAYSCALE)
    _, bgrid = cv2.threshold(ggrid, 10, 255, cv2.THRESH_BINARY)
    
    start, end = (255, 0), (0, 255)

    depth = math.ceil(math.log2(max(bgrid.shape[0] - 1, bgrid.shape[1] - 1)))
    quadtree = build_tree(bgrid, start, end, depth, 0, 0, bgrid.shape[0] - 1, bgrid.shape[1] - 1)
    cgrid = cv2.cvtColor(ggrid,cv2.COLOR_GRAY2RGB)
    graph = build_graph(quadtree)
    draw_edges(cgrid, graph)
    
    scale = 2
    cgrid = cv2.resize(cgrid, (cgrid.shape[1]*scale, cgrid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    #cv2.imshow('quadtree', cgrid)
    #cv2.imwrite("quadtree_environment_partition.png", cgrid)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    t = timeit.timeit("build_tree(bgrid, start, end, 3, 0, 0, bgrid.shape[0] - 1, bgrid.shape[1] - 1)", globals=globals(), number=15)
    print(t)
    