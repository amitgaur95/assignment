def purplecount(b, positionX, positionY, m, n):
    purple = 0
    for p in range(positionX-1, positionX+2):
        for q in range(positionY-1, positionY+2):
            if(p < 0 or q < 0 or p >= m or q >=n):
                continue

            if(p == positionX and q == positionY):
                continue

            if(b[p][q] == 1):
                purple += 1
    return purple

    def main():
        temp = input()
        (sizeofX, sizeofY) = (int(temp.split(',')[0]), int(temp.split(',')[1]))

        board = []
        s = []

        for i in range(sizeofX):
            temp = []
            t = []
            inp = input()
            for j in range(sizeofY):
                temp.append(int(inp[j]))
                t.append
                s.append(t)
                board.append(temp)

        fin = input()
        (targetX, targetY, turns) = (int(fin.split(',')[0]), int(fin.split(',')[1]), int(fin.split(',')[2]))

        genpurple = 0

        for generation in range(turns + 1):
            if(board[targetX][targetY] == 1):
                genpurple += 1

        for x in range(sizeofX):
            for y in range(sizeofY):

                purpleneighbour = purplecount(board, x, y, sizeofX, sizeofY)

                if(board[x][y] == 0):
                    if(purpleneighbour == 3 or purpleneighbour == 6):
                        s[x][y] = 1
                else:
                    if(purpleneighbour in [0,1,4,5,7,8]):
                        s[x][y] = 0

            else:
                if(purpleneighbour in [0,1,4,5,7,8]):
                    s[x][y] = 0
                else:
                    s[x][y] = 1

        board = [x[:] for x in s]

        print(genpurple)

    if _name_ == "_main_":
        main()