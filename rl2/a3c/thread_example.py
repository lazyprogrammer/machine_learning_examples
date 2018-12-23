import itertools
import threading
import time
import multiprocessing
import numpy as np


class Worker:
  def __init__(self, id_, global_counter):
    self.id = id_
    self.global_counter = global_counter
    self.local_counter = itertools.count()

  def run(self): 
    while True:
      time.sleep(np.random.rand()*2)
      global_step = next(self.global_counter)
      local_step = next(self.local_counter)
      print("Worker({}): {}".format(self.id, local_step))
      if global_step >= 20:
        break

global_counter = itertools.count()
NUM_WORKERS = multiprocessing.cpu_count()

# create the workers
workers = []
for worker_id in range(NUM_WORKERS):
  worker = Worker(worker_id, global_counter)
  workers.append(worker)

# start the threads
worker_threads = []
for worker in workers:
  worker_fn = lambda: worker.run()
  t = threading.Thread(target=worker_fn)
  t.start()
  worker_threads.append(t)


# join the threads
for t in worker_threads:
  t.join()

print("DONE!")