class repeat:
    def spotrepeat(a):
        inputfromuser = input("enter a string: \n")
        count = {}

for x in inputfromuser:
        if x in count:
            count[x] += 1
        else:
            count[x] = 1

for y in count:
        if count[y]>1:
            print(y,"occurs",count[y],"times")

s=repeat()
s.spotrepeat()