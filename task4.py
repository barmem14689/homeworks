a=int(input())
l=[]
sum = 0
for i in range(a):
    n = int(input())
    if n<30:
        sum+=1
    l.append(n)
print(max(l)-min(l))
print(sum)