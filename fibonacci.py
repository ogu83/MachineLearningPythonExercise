from datetime import datetime

# very bad algo
def fib(n):    
    if n <= 2:
        f = 1
    else:
        f = fib(n-1) + fib(n-2)

    return f

#memoized dynamic programing algo
memo = {}
def fib2(n):
    if n in memo:
        return memo[n]
    if n <= 2:
        f = 1
    else:
        f = fib(n-1) + fib(n-2)
    
    memo[n] = f
    return f

#bottom up DP algo
fs = {}
def fib3(n):
    for k in range(n+1):
        if k <= 2:
            f = 1
        else:
            f = fs[k-1] + fs[k-2]

        fs[k] = f

    return fs[n]

#shortest paths (guessing)



if __name__ == "__main__":
    # start=datetime.now()
    # print(fib(30), " : ", datetime.now()-start)
    start=datetime.now()
    print(fib3(999), " : ", datetime.now()-start)