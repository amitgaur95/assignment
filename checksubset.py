# I also tried the 2nd question using Python

def validsub():
    arr=[] 
    for i in range (7):
        x=int(input("enter arr of 7 length\n")) 
        arr.insert(i,x)
        i+=1

    subseq=[] 
    for i in range (3):
        x=int(input("enter subseq of 3 length \n")) 
        subseq.insert(i,x)
        i+=1
    
    arrayindex=0
    seqindex=0
    while arrayindex<len(arr) and seqindex<len(subseq):
        if  arr[arrayindex]==subseq[seqindex]:
           seqindex +=1
        arrayindex +=1
        
        
    return seqindex== len(subseq)
a=validsub()
print(a)