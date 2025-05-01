import matplotlib.pyplot as plt

if __name__ == "__main__":
    astar_raw = ""
    with open("astar_benchmark.txt", "r") as file:
        astar_raw = file.read()

    astar_time_pair = astar_raw.split("\n")

    grid_size = []
    time = []

    plt.xlabel("Square Grid Length")
    plt.ylabel("Time in Seconds")
    plt.title("A Star Pathfinding Speed")

    for pair_raw in astar_time_pair:
        pair = pair_raw.split(" ") 
        grid_size.append(int(pair[0]))
        time.append(float(pair[1]))

    plt.plot(grid_size, time)
    plt.show()