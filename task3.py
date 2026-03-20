a = int(input())
l=[]

for i in range(a):
    n = int(input())

    if n%10==4:
        l.append(n)

print(min(l))