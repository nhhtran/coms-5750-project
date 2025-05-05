import matplotlib.pyplot as plt

if __name__ == "__main__":
    astar_raw = ""
    with open("astar_node_explored.txt", "r") as file:
        astar_raw = file.read()

    astar_time_pair = astar_raw.split("\n")
    print(astar_time_pair)
    grid_size = []
    time = []

    plt.xlabel("Square Grid Length")
    plt.ylabel("Nodes Explored")
    plt.title("A Star Pathfinding Efficiency")

    for pair_raw in astar_time_pair:
        pair = pair_raw.strip()
        pair = pair.split(" ") 
        for i in range(1, len(pair)):
            grid_size.append(int(pair[0]))
            time.append(int(pair[i]))
            
    print(time)
    plt.scatter(grid_size, time)
    plt.savefig("astar_efficiency_benchmark.png")
    plt.show()