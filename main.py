import random


# Hard-coded values for p_a and p_b
a = None
p_a = []

b = None
p_b = []

training_set_file = open('data/mnist_test.csv', 'r')
for line in training_set_file:
    number = line.split(',')
    if a is None:
        a = [ int(x) for x in number ]
    else:    
        if number[0] != a[0] and b is None:
            b = [ int(x) for x in number ]
            break

print('Digito:',a[0])
print('Digito:', b[0])





# Constant GAMMA
GAMMA = 0.0001

# Constant for TIME
T = 10

# 
epsilon = 0.5


classifiers = []


t = 1
w_a = []
w_b = []

w_a[t] = 1
w_b[t] = 1

p_a[t] = 0.5
p_b[t] = 0.5

for t in range(1,T):
    sum_w = w_a[t+1] + w_b[t+1]
    p_a[t+1] = w_a[t+1] / sum_w
    p_b[t+1] = w_b[t+1,1] / sum_w
    
print("FIM!")
training_set_file.close()