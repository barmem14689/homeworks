a = int(input())
sum = 0

while True:
    if a ==0:
        break
    if a%4==0 and a>99 and a<100:
        sum+=1

    a=int(input())

print(sum)