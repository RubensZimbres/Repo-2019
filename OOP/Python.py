class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width
    
rectangle = Rectangle(4,2)
rectangle.area()

class Square:
    def __init__(self, length):
        self.length = length

    def area(self):
        return self.length * self.length

    def perimeter(self):
        return 4 * self.length
    
    
class Cube(Square):
    def surface_area(self):
        face_area = super().area()
        return face_area * 6
    def volume(self):
        face_area=super().area()
        return face_area*self.length

cube=Cube(3)

cube.surface_area()

cube.volume()        


class Point:
    def reset(self):
        self.x=0
        self.y=0
    
p=Point()
p.reset()

p.x


import math
class Point:
    def move(self,x,y):
        self.x=x
        self.y=y
        
    def reset(self):
        self.move(0,0)
        
    def distance(self,other):
        return math.sqrt((self.x-other.x)**2-(self.y-other.y)**2)
    
p1=Point()
p2=Point()

p1.reset()

p2.move(5,0)

p1.distance(p2)


import math
class Point:
    def __init__(self,x,y):
        '''Initialize the position'''
        self.move(x,y)
    
    def move(self,x,y):
        '''Move to new location'''
        self.x=x
        self.y=y
        
    def reset(self):
        self.move(0,0)
        
    def distance(self,other):
        return math.sqrt((self.x-other.x)**2-(self.y-other.y)**2)

p=Point(3,1)

p.distance(p2)
