
import matplotlib.pyplot as plt

import numpy as np

def solve_maze(maze):
    import sys
    sys.setrecursionlimit(9000000)

    ret = maze.copy()

    def move(path, cur):
        if cur in path:
            return
        path.append(cur)

        i, j = cur[0], cur[1]

        if cur == [149, 150]:
            print("Arrived!")
            for ele in path:
                ret[ele[0]][ele[1]] = 1
            return

        up, down, left, right = ret[i - 1][j], ret[i + 1][j], ret[i][j - 1], ret[i][j + 1]
        tem1 = path.copy()
        tem2 = path.copy()
        tem3 = path.copy()
        tem4 = path.copy()

        if up == 0:
            move(tem1, [i - 1, j])
        if down == 0:
            move(tem2, [i + 1, j])
        if left == 0:
            move(tem3, [i, j - 1])
        if right == 0:
            move(tem4, [i, j + 1])

        return

    move([[1, 0]], [1, 1])

    return ret


if __name__ == '__main__':
    data = np.load('maze_data.npy')
    data2 = np.load('maze_solve.npy')
    idx = np.random.randint(1)
    for i in range(0, 100):
        ret = solve_maze(data[i])
        print(np.array_equal(ret, data2[i]))

