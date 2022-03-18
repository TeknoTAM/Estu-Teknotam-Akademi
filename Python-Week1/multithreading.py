import threading
import time
import random
from loguru import logger

def worker():
    t = threading.currentThread()
    r = random.randint(1,10)
    logger.info(f'sleeping {r}')
    
    time.sleep(r)
    logger.info(f'ending {t.getName()}')

class WorkerThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self)
        self.thread_name = self.getName()

    def run(self):
        r = random.randint(1,10)
        logger.info(f' Thread name: {self.thread_name} Sleeping time: {r}')
        time.sleep(r)
        logger.info(f'ending {self.thread_name}')


if __name__ == "__main__":

    """Multi-threading with threading module"""

    for i in range(3):
        t = threading.Thread(target=worker) # daemon = True
        t.start()

    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        logger.info(f'wait for joining {t.getName()}')
        t.join()

    """Multi-threading with class that inherit from threading.Thread class"""
    for i in range(3):
        t = WorkerThread() # daemon = True
        t.start()



    main_thread = threading.currentThread()
    for t in threading.enumerate():
        if t is main_thread:
            continue
        logger.info(f'wait for joining {t.getName()}')
        t.join()
        logger.info(f'thread is finished {t.getName()}')