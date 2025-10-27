import random

state_set = {"A", "B", "C", "T"}
transition_map = {
    "A" : ["A", "B"],
    "B" : ["A", "C"],
    "C" : ["B", "T"],
}


total_steps = 0
for i in range(1000000):
    steps = 0

    state = "A"

    while(state != "T"):
        state = random.choice(transition_map[state])
        steps += 1
    
    total_steps += steps


print(total_steps/1000000)
        