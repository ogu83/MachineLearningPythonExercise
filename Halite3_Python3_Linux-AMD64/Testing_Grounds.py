def surroundings():
    size = 3
    
    surroundings = []

    for y in range(-1 * size, size+1):
        row=[]
        for x in range(-1 * size, size+1):
            row.append([x,y])

        surroundings.append(row)

    for r in surroundings:
        print(r)

surroundings()