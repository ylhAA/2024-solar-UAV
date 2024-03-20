import numpy as np


def modify_pop(pop_reverse, population, max_rate):
    max_rate = max_rate / 2
    for i in range(population):
        for j in range(30):
            if pop_reverse[i][j] > max_rate:
                pop_reverse[i][j] = max_rate
            elif pop_reverse[i][j] < -max_rate:
                pop_reverse[i][j] = -max_rate
            else:
                pass
    return pop_reverse
