from random import random


def variable_input(args):
    if len(args) > 0:
        val = {}
        for arg in args:
            val[arg] = random()
        return val
    else:
        raise IOError('Variable required')


if __name__ == "__main__":
    ans = variable_input(['hola',])
    print(ans)
    ans = variable_input(['hola', 'y', 'algo', 'más'])
    print(ans)
    # ans = variable_input()
    # print(ans)
    elements = ("H_O", "Fe_Xe", "Fe_H")
    ans = variable_input(elements)
    print(ans)