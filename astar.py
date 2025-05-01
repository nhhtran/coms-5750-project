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


def astar(grid, obstacleIds, startPos, endPos):
    queue = Heap()
    closed = set()
    nodes = {}
    
    xMax = len(grid[0])
    yMax = len(grid)

    node = Node(startPos, None, 0, 0)
    nodes[startPos] = node
    queue.push(node)

    def hcost(posA, posB):
        dY = posA[1] - posB[1]
        dX = posA[0] - posB[0]
        return math.sqrt(dY * dY + dX * dX)

    while(queue.size > 0):
        current = queue.pop()
        closed.add(current.element)

        # if the end is None
        if (current.element == endPos):
            path = []
            path.append(current.element)
            trace = current.parent
            while (not(trace is None)):
                path.append(trace.element)
                trace = trace.parent
            path.reverse()
            return (path, closed)

        # Getting the neighbors of the current node
        neighbors = []
        for i in range(max(0, current.element[1] - 1), min(yMax - 1, current.element[1] + 1) + 1):
            for j in range(max(0, current.element[0] - 1), min(xMax - 1, current.element[0] + 1) + 1):
                if (not (grid[i][j] in obstacleIds or (j, i) in closed)):
                    neighbors.append((j, i))

        for n in neighbors:
            gcost = current.gcost + 1
            fcost = hcost(n, endPos) + gcost

            if (n in nodes):
                node = nodes[n]
                if (fcost < node.fcost):
                    node.fcost = fcost
                    node.gcost = gcost
                    node.parent = current
                    queue.up(node)
            else:
                node = Node(n, current, fcost, gcost)
                nodes[n] = node
                queue.push(node)
                
    return ([], closed)


def test1_astar():
    grid = [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    obstacles = set()
    obstacles.add(1)
    startPos = (0, 0)
    endPos = (2, 2)
    (path, closed) = astar(grid, obstacles, startPos, endPos)
    print("path", path)

    for p in path:
        grid[p[1]][p[0]] = 2

    s = ""
    for r in grid:
        for c in r:
            s += str(c)
        s += "\n"
    print(s)


def test2_astar():
    grid = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    ]

    obstacles = set()
    obstacles.add(1)
    startPos = (0, 0)
    endPos = (9, 9)
    (path, closed) = astar(grid, obstacles, startPos, endPos)
    print("path", path)

    for c in closed:
        grid[c[1]][c[0]] = 3

    s = ""
    for r in grid:
        for c in r:
            s += str(c)
        s += "\n"
    print(s)


def test1_heap():
    n = [11, 100, 12, 33, 44, 10, 88]
    heap = Heap()
    for i in n:
        heap.push(Node(i, i))
    print("! Heap after pushed [11, 100, 12, 33, 44, 10, 88]")
    print(heap)
    
    k = heap.pop()
    while(k is not None):
        print("! Heap popped returns")
        print(k)
        print("! New heap")
        print(heap)
        k = heap.pop()

    print("! Heap after pop all")
    print(heap)


def test2_heap():
    k = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14)]
    n = [11, 100, 12, 33, 44, 10, 88]
    heap = Heap()
    for i in range(len(k)):
        heap.push(Node(k[i], n[i]))
    print("! Heap after pushed [11, 100, 12, 33, 44, 10, 88]")
    print(heap)

    node = Node((3, 4), 1)
    heap.up(node)
    print("! Heap after update node")
    print(heap)

    

if __name__ == "__main__":
    print("Running astar.py")
    test2_astar()