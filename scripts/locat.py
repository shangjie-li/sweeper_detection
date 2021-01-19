# -*- coding: UTF-8 -*- 

class Locat():
    def __init__(self):
        print("Locat created.")
        
    def loadlocat(self, l1, l2, l3, l4, l5):
        self.l1 = l1 * 1.0
        self.l2 = l2 * 1.0
        self.l3 = l3 * 1.0
        self.l4 = l4 * 1.0
        self.l5 = l5 * 1.0
        
    def findregion(self, x, z):
        if z >= self.l1 and z < self.l1 + self.l2:
            if x >= - self.l5 - self.l4 / 2 and x < - self.l4 / 2:
                return 1
            elif x >= - self.l4 / 2 and x < 0:
                return 2
            elif x >= 0 and x < self.l4 / 2:
                return 3
            elif x >= self.l4 / 2 and x < self.l4 / 2 + self.l5:
                return 4
            else:
                return 0
        elif z >= self.l1 + self.l2 and z < self.l1 + self.l2 + self.l3:
            if x >= - self.l5 - self.l4 / 2 and x < - self.l4 / 2:
                return 5
            elif x >= - self.l4 / 2 and x < 0:
                return 6
            elif x >= 0 and x < self.l4 / 2:
                return 7
            elif x >= self.l4 / 2 and x < self.l4 / 2 + self.l5:
                return 8
            else:
                return 0
        else :
            return 0
            
if __name__ == '__main__':
    locat = Locat()
    locat.loadlocat(7, 3, 3, 3, 3)
    print("input x:")
    x = input()
    x = float(x)
    print("input z:")
    z = input()
    z = float(z)
    region = locat.findregion(x, z)
    print("the number of area is", region)
    
