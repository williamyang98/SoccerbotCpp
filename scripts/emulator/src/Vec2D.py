class Vec2D:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    @staticmethod
    def from_tuple(t):
        return Vec2D(t[0], t[1])
    
    def copy(self):
        return Vec2D(self.x, self.y)
    
    def __add__(self, o):
        return Vec2D(self.x+o.x, self.y+o.y)
    
    def __sub__(self, o):
        return Vec2D(self.x-o.x, self.y-o.y)
    
    def __mul__(self, k):
        return Vec2D(self.x*k, self.y*k)

    def __rmul__(self, k):
        return Vec2D(self.x*k, self.y*k)
    
    def __truediv__(self, k):
        return Vec2D(self.x/k, self.y/k)

    def __floordiv__(self, k):
        return Vec2D(self.x//k, self.y//k)
    
    def __eq__(self, o):
        return (self.x == o.x) and (self.y == o.y)
    
    def set(self, x, y):
        self.x = x
        self.y = y
    
    def dot(self, o):
        return self.x*o.x + self.y*o.y

    def length(self):
        return (self.x*self.x + self.y*self.y)**0.5

    def norm(self):
        l = self.length()
        if l < 1:
            return self
        return self/l

    def to_tuple(self):
        return (self.x, self.y)

    def cast_tuple(self, t):
        return (t(self.x), t(self.y))

    def __str__(self):
        return f"P({self.x:.2f},{self.y:.2f})"
    
    def __repr__(self):
        return str(self)