import matplotlib.pyplot as plt
import math

def efficiency():
    astar_raw = ""
    with open("astar_efficiency_benchmark.txt", "r") as file:
        astar_raw = file.read()

    astar_quadtree_raw = ""
    with open("quadtree_astar_efficiency_benchmark.txt", "r") as file:
        astar_quadtree_raw = file.read()

    astar_efficiency = astar_raw.split("\n")
    qastar_efficiency = astar_quadtree_raw.split("\n")
    grid_size = []
    astar_time = []
    qastar_time = []

    for i in range(len(astar_efficiency)):
        s = astar_efficiency[i].strip()
        a = s.split(" ")
        for j in range(1, len(a)):
            grid_size.append(int(a[0]))
            astar_time.append(math.log2(int(a[j])))

    for i in range(len(qastar_efficiency)):
        s = qastar_efficiency[i].strip()
        a = s.split(" ")
        for j in range(1, len(a) - 1):
            qastar_time.append(math.log2(int(a[j])))

    plt.xlabel("Square Grid Length")
    plt.ylabel("log2(Nodes Explored)")
    plt.title("A* vs Quadtree A* Pathfinding Efficiency")

    plt.scatter(grid_size, astar_time , label="A*")
    plt.scatter(grid_size, qastar_time, label="Quadtree A*")
    plt.legend()
    plt.savefig("astar_quadtree_efficiency_benchmark_log2.png")
    plt.show()


def time():
    astar_benchmark = ""
    with open("astar_benchmark2.txt", "r") as file:
        astar_benchmark = file.read().split("\n")

    quadtree_astar_efficiency_benchmark_raw = ""
    with open("quadtree_astar_efficiency_benchmark.txt", "r") as file:
        quadtree_astar_efficiency_benchmark_raw = file.read().split("\n")

    quadtree_build_graph_speed_raw = ""
    with open("quadtree_build_graph_speed.txt", "r") as file:
        quadtree_build_graph_speed_raw = file.read().split("\n")

    quadtree_speed_raw = ""
    with open("quadtree_speed.txt", "r") as file:
        quadtree_speed_raw = file.read().split("\n")


    grid_size = []
    astar_time = []
    qastar_time = []
    
    build_time = []
    qsearch_time = []
    quadtree = []


    for i in range(len(astar_benchmark)):
        s = astar_benchmark[i].strip()
        a = s.split(" ")
        grid_size.append(int(a[0]))
        astar_time.append(float(a[1]))

    for i in range(len(astar_benchmark)):
        s = quadtree_astar_efficiency_benchmark_raw[i].strip()
        a = s.split(" ")
        qst = float(a[11])
        qsearch_time.append(float(a[11]))

        s = quadtree_speed_raw[i].strip()
        a = s.split(" ")
        qst += float(a[1])
        quadtree.append(float(a[1]))

        s = quadtree_build_graph_speed_raw[i].strip()
        a = s.split(" ")
        qst += float(a[1])
        build_time.append(float(a[1]))

        qastar_time.append(qst)

    plt.xlabel("Square Grid Length")
    plt.ylabel("Time in Seconds")
    plt.title("A* vs Quadtree A* Time")

    #plt.plot(grid_size, quadtree , label="Build Quadtree")
    #plt.plot(grid_size, build_time , label="Build Graph")
    #plt.plot(grid_size, qsearch_time, label="A* Search")
    plt.plot(grid_size, astar_time, label="A*")
    plt.plot(grid_size, qastar_time, label="Quadtree A*")
    plt.legend()
    plt.savefig("astar_quadtree_time_benchmark2.png")
    plt.show()

if __name__ == "__main__":
    time()