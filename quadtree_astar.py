import math

class Node:
    def __init__(self, element, parent, fcost, gcost):
        self.element = element
        self.fcost = fcost
        self.gcost = gcost
        self.parent = parent

    def getValue(self):
        return self.fcost

    def compare(self, other):
        return self.fcost < other.fcost
    
    def __str__(self):
        return str(self.element)

    def __repr__(self):
        return str(self.element)

    def __hash__(self):
        return hash(self.element)

    def __eq__(self, other):
        return self.element == other.element

class Heap:
    def __init__(self):
        self.h = []
        self.t = {}
        self.size = 0
    
    def push(self, node):
        if (node in self.t):
            raise Exception(str(node) + " already exists in heap.")
            
        if (self.size < len(self.h)):
            self.h[self.size] = node
        else:
            self.h.append(node)

        self.t[self.h[self.size]] = self.size

        self.size += 1
        
        cindex = self.size - 1
        pindex = (cindex - 1) // 2
        while (cindex != 0 and node.compare(self.h[pindex])):
            parent = self.h[pindex]
            self.h[pindex] = node
            self.t[self.h[pindex]] = pindex

            self.h[cindex] = parent
            self.t[self.h[cindex]] = cindex
            
            cindex = pindex
            pindex = (cindex - 1) // 2
    
    def pop(self):
        if (self.size == 0):
            return None

        node = self.h[0]
        self.h[0] = self.h[self.size - 1]
        del self.t[node]
        
        self.size -= 1

        pindex = 0
        while(2 * pindex + 1 < self.size):
            lindex = 2 * pindex + 1
            rindex = 2 * pindex + 2
            parent = self.h[pindex]

            cindex = lindex
            if (rindex < self.size and self.h[rindex].compare(self.h[lindex])):
                cindex = rindex

            if (self.h[cindex].compare(parent)):
                
                self.h[pindex] = self.h[cindex]
                self.t[self.h[pindex]] = pindex
                
                self.h[cindex] = parent
                self.t[self.h[cindex]] = cindex
                
                pindex = cindex
            else:
                break

        return node

    def up(self, node):
        cindex = self.t[node]
        pindex = (cindex - 1) // 2
        while (cindex != 0 and node.compare(self.h[pindex])):
            parent = self.h[pindex]
            self.h[pindex] = node
            self.t[self.h[pindex]] = pindex

            self.h[cindex] = parent
            self.t[self.h[cindex]] = cindex
            
            cindex = pindex
            pindex = (cindex - 1) // 2
        
    def top(self):
        if (self.size == 0):
            return None
        return self.h[0]

    def __str__(self):
        s = "h = ["
        if (self.size > 0):
            for i in range(self.size - 1):
                s += str(self.h[i]) + ", "
            s += str(self.h[self.size - 1])
        s += "] \n"
        s += "t = " + str(self.t)
        return s


def find_quadrant(V, pos):
    quad_num = None
    for i in V.keys():
        node = V[i]
        if (pos[0] <= node.bbox[2] and pos[0] >= node.bbox[0] and pos[1] <= node.bbox[3] and pos[1] >= node.bbox[1]):
            quad_num = i
            break
    return quad_num


def floatcoord2int(coord):
    return (int(coord[0]), int(coord[1]))


def astar(graph, startPos, endPos):
    V, E, W = graph[0], graph[1], graph[2]

    queue = Heap()
    closed = set()
    nodes = {}
    
    startQuadNum = find_quadrant(V, startPos)
    endQuadNum   = find_quadrant(V, endPos)
    
    node = Node(V[startQuadNum], None, 0, 0)
    nodes[startQuadNum] = node
    queue.push(node)

    def hcost(posA, posB):
        dY = posA[1] - posB[1]
        dX = posA[0] - posB[0]
        return math.sqrt(dY * dY + dX * dX)

    while(queue.size > 0):
        current = queue.pop()
        currentNum = current.element.num_id()
        closed.add(currentNum)

        # if the end is None
        if (currentNum == endQuadNum):
            path = []
            path.append(endPos)
            path.append(floatcoord2int(current.element.get()))
            trace = current.parent
            while (not(trace is None)):
                path.append(floatcoord2int(trace.element.get()))
                trace = trace.parent
            path.append(floatcoord2int(startPos))
            path.reverse()
            return (path, closed)

        # Getting the neighbors of the current node

        for i in range(len(E[currentNum])):
            neighborNum = E[currentNum][i]

            if (neighborNum in closed):
                continue

            gcost = current.gcost + W[currentNum][i]
            fcost = hcost(V[neighborNum].get(), endPos) + gcost

            if (neighborNum in nodes):
                node = nodes[neighborNum]
                if (fcost < node.fcost):
                    node.fcost = fcost
                    node.gcost = gcost
                    node.parent = current
                    queue.up(node)
            else:
                node = Node(V[neighborNum], current, fcost, gcost)
                nodes[neighborNum] = node
                queue.push(node)
                
    return ([], closed)
    

def draw_path(grid, path):
    for i in range(1, len(path)):
        cur_p = path[i]
        pre_p = path[i - 1]
        cv2.line(grid, (int(cur_p[0]), int(cur_p[1])), (int(pre_p[0]), int(pre_p[1])), (0, 155, 50), 1)

    cur_p = path[0]
    cv2.circle(grid, (int(cur_p[0]), int(cur_p[1])), 5, (0, 0, 255), -1)
    cur_p = path[len(path) - 1]
    cv2.circle(grid, (int(cur_p[0]), int(cur_p[1])), 5, (255, 0, 0), -1)




if __name__ == "__main__":
    print("Running quadtree_astar.py")
    
    import cv2
    import timeit
    from quadtree_orcld import build_graph, build_tree

    ggrid = cv2.imread("blocky_path1.png", cv2.IMREAD_GRAYSCALE)
    _, bgrid = cv2.threshold(ggrid, 10, 255, cv2.THRESH_BINARY)
    
    start, end = (bgrid.shape[0] - 1, 0), (0, bgrid.shape[1] - 1)

    depth = math.ceil(math.log2(max(bgrid.shape[0] - 1, bgrid.shape[1] - 1)))
    quadtree = build_tree(bgrid, start, end, depth, bgrid.shape[0] - 1, bgrid.shape[1] - 1)
    if (not quadtree.is_enclosed):
        graph = build_graph(quadtree)
        path, closed = astar(graph, start, end)
        if (len(path) == 0):
            print("Astar says: No path found!")
        else:
            cgrid = cv2.cvtColor(ggrid,cv2.COLOR_GRAY2RGB)
            draw_path(cgrid, path)
            scale = 2
            cgrid = cv2.resize(cgrid, (cgrid.shape[1]*scale, cgrid.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('quadtree', cgrid)
            cv2.imwrite("quadtree_environment_non2power_path.png", cgrid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Quadtree says: No path found!")

    #cgrid = cv2.cvtColor(ggrid,cv2.COLOR_GRAY2RGB)
    #draw_edges(cgrid, graph)