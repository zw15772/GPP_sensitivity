# coding=gbk

from analysis import *



def test1():
    growseason = np.array(growseason) - 1
    print growseason - 1
    new_growing_season = []
    for i in growseason:
        # print i
        if i < 1:
            new_i = 12 + i
            new_growing_season.append(new_i)
        else:
            new_growing_season.append(i)
    new_growing_season = np.array(new_growing_season)




    pass

class TEST1:
    def __init__(self):

        pass

    def run(self):

        pass


def main():

    test1()


if __name__ == '__main__':
    main()