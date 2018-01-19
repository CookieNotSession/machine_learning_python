hashing = []
for hashindex in range(1000001):
    hashing.append(0)

def uva100(n):
    count = 1
    cal = []
    cal.append(n)
    fixedvalue = n
    while n != 1:
        if n<len(hashing) and hashing[n] >0 :
            count = count + hashing[n] -1
            cal.append(n)
            break
        else :
            if n % 2 == 1:
                n = 3 * n + 1
            else:
                n = n // 2
            cal.append(n)
            count += 1
    countmp = count
    for j in cal[::-1]:
        countmp = countmp-1
        if j < len(hashing):
            if hashing[j] == 0:
               hashing[j] = count - countmp
            else:
                break
    hashing[fixedvalue] = count
    return count

if __name__ == "__main__":

    while True:
        try:
            i, j = map(int, input().split())
            minnum = min(i,j)
            maxnum = max (i,j)
            #for ans in range(minnum,maxnum+1):
                #max(uva100(ans))
            maxcycle = max(uva100(ans) for ans in range(minnum,maxnum+1))
            print(i, j, maxcycle)
        except EOFError:
            break
