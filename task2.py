a = int(input())
l=[]
while True:
    if a==0:
        break
    if a%4==0 or a%9==0:
        l.append(a)
    a = int(input())
print(sum(l))
