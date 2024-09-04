
def sS_action(state, sS):
    if state[0] + state[5] < sS[0]:
        order = sS[1] - state[0] - state[5]
    else:
        order = 0
    return order