from typing import Union

class Angular:
    def __init__(self, angular_momentum:int):
        if not isinstance(angular_momentum, int):
            raise TypeError("Angular momentum must be an integer")

        self.angular_momentum = angular_momentum
    
    def __str__(self):
        return f"J={self.angular_momentum}"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        return self.angular_momentum == other.angular_momentum

    def value(self):
        return self.angular_momentum / 2

    def index(self):
        return self.angular_momentum

    def projections(self):
        """
        Returns the possible projections of the angular momentum
        """
        return [Angular(i) for i in range(-self.index(), self.index() + 1, 2)]    
    
    def __add__(self, other):
        return Angular(self.angular_momentum + other.angular_momentum)
    
    def __sub__(self, other):
        return Angular(self.angular_momentum - other.angular_momentum)
    
    def couple(self, other):
        """
        Couple two angular momenta
        """
        minimum = abs(self.angular_momentum - other.angular_momentum)
        maximum = self.angular_momentum + other.angular_momentum
        return [Angular(i) for i in range(minimum, maximum + 1, 2)]

class QN:
    def __init__(self, angular_momentum:Union[int, Angular], parity: int) -> None:
        if isinstance(angular_momentum, int):
            self.angular = Angular(angular_momentum)
        else:
            self.angular = angular_momentum
        if not isinstance(parity, int):
            raise TypeError("Parity must be an integer")
        if not parity in [-1, 1]:
            raise ValueError("Parity must be either -1 or 1")
        self.parity = parity     

    def __str__(self):
        return f"{self.angular}^{self.parity}"

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.angular == other.angular and self.parity == other.parity

    def __add__(self, other):
        return QN(self.angular + other.angular, self.parity * other.parity)
    
    def __sub__(self, other):
        return QN(self.angular - other.angular, self.parity * other.parity)
    
    def couple(self, other):
        """
        Couple two quantum numbers
        """
        return [QN(j, p) for j in self.angular.couple(other.angular) for p in [self.parity * other.parity]]

