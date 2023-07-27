


use_dummy = False

if use_dummy:
    from multiprocessing.dummy import Process, Queue, Semaphore
else:
    from multiprocessing import Process, Queue, Semaphore
