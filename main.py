import os
import time
from datetime import timedelta

from config import get_arguments
from train import distributed_device_train
from evaluate import single_device_evaluate

if __name__ == '__main__':
    args = get_arguments()

    tictoc = time.time()
    if args.train_flag:
        distributed_device_train(args)
    else:
        single_device_evaluate(args)
    print('%s: Process is Done During %s'%(time.ctime(), str(timedelta(seconds=(time.time() - tictoc)))))
