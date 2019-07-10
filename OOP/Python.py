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
