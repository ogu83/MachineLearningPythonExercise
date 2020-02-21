# Peak finder algoritms according to https://www.youtube.com/watch?v=HtSuA80QTyo 1. Algorithmic Thinking, Peak Finding from MIT OpenCourseWare

import numpy as np
import datetime

def get_an_random_arr(max_number, test_limit):
    # arr = np.random.randint(1, max_number, test_limit)
    test_limit = test_limit // 2
    arr = np.linspace(0, max_number, num=test_limit)    
    dv = np.random.randint(1, max_number, 1)[0]    
    arr1 = arr[:dv]
    arr2 = arr[dv:]
    arr = np.append(arr2,arr1)
    return arr

def full_scan(max_number, test_limit):    
    arr = get_an_random_arr(max_number, test_limit)
    
    now = datetime.datetime.now() 
    first_peak = 0
    cycle = 0
    for n in range(1, len(arr)):
        cycle += 1
        if (arr[n] >= arr[n-1] and arr[n] >= arr[n+1]):
            first_peak = arr[n-1], arr[n], arr[n+1]
            return first_peak, (datetime.datetime.now() - now).microseconds, cycle
    
    return "No Peak"
    
def divide_conquer_strategy(max_number, test_limit):    
    arr = get_an_random_arr(max_number, test_limit)
    # print(arr)
    
    now = datetime.datetime.now()
    first_peak = 0
    cycle = 0
    while(True):
        cycle += 1
        n = len(arr)
        
        if n == 3:
            # if (arr[n] >= arr[n-1] and arr[n] >= arr[n+1]):
            first_peak = arr[n//2-1], arr[n//2], arr[n//2+1]
            return first_peak, (datetime.datetime.now() - now).microseconds, cycle
            # else
        if n == 2:
            return -1,-1,-1
        if n == 1:
            return -1,-1,-1
        
        # print(arr)
        if arr[n//2] < arr[n//2-1]:
            arr = arr[:(n//2+1)]
        elif arr[n//2] < arr[n//2+1]:
            arr = arr[(n//2-1):]
        else:
            first_peak = arr[n//2-1], arr[n//2], arr[n//2+1]
            return first_peak, (datetime.datetime.now() - now).microseconds, cycle

if __name__ == "__main__":    
    test_count = 5
    test_limit = 10**7
    max_number = 10**7
    
    # FULL SCAN TEST
    average_f = 0
    sum = 0
    for i in range(test_count):    
        peak, ms, cycle = full_scan(max_number, test_limit)
        sum += cycle
        print("First Peak:", peak, "Micro Seconds:", ms, "Cycle:", cycle)
        
    average_f = sum / test_count
    print("full_scan Time Average", average_f)
    print("")
    
    # Divide and Conquer Strategy
    average_dc = 0
    sum = 0
    for i in range(test_count):    
        peak, ms, cycle = divide_conquer_strategy(max_number, test_limit)
        sum += cycle
        print("First Peak:", peak, "Micro Seconds:", ms, "Cycle:", cycle)
        
    average_dc = sum / test_count
    print("divide_conquer_strategy Time Average", average_dc)
    print("")
    
    if (average_f < average_dc):
        print("FULL SCAN is better, Perc: %", (average_dc - average_f) / average_f)
    else:
        print("divide_conquer_strategy is better, Perc: %", (average_f - average_dc) / average_dc)
    