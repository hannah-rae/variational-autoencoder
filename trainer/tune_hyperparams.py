#import util
import os
from random import randint

MAX_ITERATIONS = 60

N_Z_MIN = 1700
N_Z_MAX = 2700

BATCH_MIN = 2
BATCH_MAX = 50

LAYER1_MIN = 128
LAYER1_MAX = 256

LAYER2_MIN = 16
LAYER2_MAX = 128

KERNEL_MIN = 3
KERNEL_MAX = 7


for i in range(0, MAX_ITERATIONS):
    # Generate random values for hyperparameters
    n_z = randint(N_Z_MIN, N_Z_MAX)
    batch_size = randint(BATCH_MIN, BATCH_MAX)
    layer1 = randint(LAYER1_MIN, LAYER1_MAX)
    layer2 = randint(LAYER2_MIN, LAYER2_MAX)
    kernel_size = randint(KERNEL_MIN, KERNEL_MAX)
    log_dir = '/home/hannah/src/MastcamVAE/log/%d_%d_%d_%d_%d' % (n_z, batch_size, layer1, layer2, kernel_size)

    os.system("mkdir %s" % log_dir)

    os.system("python util.py --n_z %d --batch_size %d --C1 %d --C2 %d --kernel_size %d --log_dir %s" % (n_z, batch_size, layer1, layer2, kernel_size, log_dir))




