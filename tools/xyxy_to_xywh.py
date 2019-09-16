import numpy as np

if __name__ == '__main__':
    for i in range(6):
        filename = 'b{}.txt'.format(i + 1)
        results = np.loadtxt(filename, delimiter=',')
        data = ''
        for line in results:
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}\n'.format(int(line[0]), int(line[1]),
                                                                        line[2], line[3], line[4] - line[2],
                                                                        line[5] - line[3])
        file = open(filename, 'w+')
        file.write(data)
        file.close()
