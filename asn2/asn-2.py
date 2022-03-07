import numpy as np
import random

from numpy.linalg import inv

# Step 1 - Problem 1
a = np.matrix('1 2 3; 4 5 6; 7 8 9')
b = np.matrix('1 4 7; 2 5 8; 3 6 9')

# Part a
matrixMult = np.matmul(a, b)

#Part b
matrixElemMult = np.multiply(a, b)

#Part c
transposeA = np.matrix.transpose(a)
inverseA = inv(a)
prod = np.dot(transposeA, b)
finalProd = np.dot(prod, inverseA)

#Part d
a = np.reshape(a, 9)
b = np.reshape(b, 9)
cv = np.concatenate((a, b))

l2norm = np.linalg.norm(cv, axis=0)

# Number 2
i = np.random.randint(0, 256, (256, 256))

m = np.random.randint(0, 2, (256, 256))

for row in range(len(i)):
    for col in range(len(i[0])):
        if (i[row][col] == m[row][col]):
            i[row][col] = 0

# Step 2

# Number 2
def sqMatrixOps(a, b):

    if (len(a) == len(b)):
        if(len(a[0]) == len(b[0])):
            transposeA = np.matrix.transpose(a)
            inverseA = inv(a)
            prod = np.dot(transposeA, b)
            finalProd = np.dot(prod, inverseA)
            return finalProd
        else:
            raise Exception("Matrices need to be of same size!")
    else:
        raise Exception("Matrices need to be of same size!")

# Number 3
def probabilityFunc():
    num = random.random()
    if num < 0.5: 
        return -1
    else:
        return 1


#Number 5
list = []
for i in range(0, 10):
    list.append(random.randint(0, 10))

tempList = []
for i in range(11):
    tempList.append(i)

list = [0 if (x>=5 and x <=7) else 1 for x in tempList]


# Number 6
list2 = []
for i in range(0, 10):
    list2.append(random.randint(0, 10))

list2 = ["Even" if x == x % 2 else "Odd" for x in tempList]

# Number 7
randArr = np.random.randint(0, 255, (256, 256))
randArrS = np.random.randint(0, 255, (256, 256, 3))

for row in range(len(randArrS)):
    for col in range(len(randArrS[0])):
        randArrS[row][col][2] = randArr[row][col]
