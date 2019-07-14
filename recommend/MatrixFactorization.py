from scipy.spatial import distance
import numpy as np


def main():
    user1 = np.array([4, 5, 0, 1, 0])
    user2 = np.array([3, 0, 1, 5, 0])
    user3 = np.array([4, 0, 0, 3, 5])
    print(distance.cosine(user1, user2))


if __name__ == '__main__':
    main()
