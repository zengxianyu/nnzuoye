import matplotlib.pyplot as plt

with open('cmake-build-debug/train_loss2.txt') as f:
    val = f.readlines();
val = [float(v.strip('\n')) for v in val]
plt.plot(val[100:])
plt.xlabel('iterations')
plt.ylabel('loss')
plt.savefig('figure.png')
