import os

# n_zs = [1805, 2279, 1831, 2306, 2077, 1844, 1848, 2561, 1726, 1856, 2023]
# batch_sizes = [4, 13, 26, 23, 29, 16, 34, 3, 17, 10, 15]
# layer1s = [213, 167, 247, 225, 140, 142, 140, 136, 132, 242]
# layer2s = [29, 16, 19, 20, 27, 42, 21, 18, 38, 18]
# kernel_sizes = [4, 5, 6, 7, 6, 7, 4, 5, 5, 3]

# n_zs = [1805, 1726]
# batch_sizes = [4, 17]
# layer1s = [213, 132]
# layer2s = [29, 38]
# c1_kernel_sizes = [5, 5]
# c2_kernel_sizes = [3, 3]

hyperparams = [(1726, 17, 132, 38, 3, 2)]
for hp in hyperparams:
    log_dir = '/home/hannah/src/MastcamVAE/log/oct27_grey_%d_%d_%d_%d_%d_%d' % hp
    os.system("mkdir %s" % log_dir)
    log_dir = (log_dir,)
    os.system("python util.py --n_z %d --input_filters 1 --batch_size %d --C1 %d --C2 %d --kernel_size_c1 %d --kernel_size_c2 %d --log_dir %s" % (hp + log_dir))