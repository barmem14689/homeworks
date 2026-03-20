a = int(input())
l=[]
for i in range(a):
    n = int(input())
    if n%6==0 and n%10==4:
        l.append(n)
print(sum(l))
