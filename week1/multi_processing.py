from multiprocessing import Process
import time
from loguru import logger


class MyProcess(Process):

    active_processes = []

    def __init__(self,sleeping_time):
        Process.__init__(self)
        self.sleeping_time = sleeping_time
        self.process_name = self.name
        MyProcess.active_processes.append(self)

    def run(self):
        logger.info(f'Process name: {self.process_name} sleeping time: {self.sleeping_time}')
        time.sleep(self.sleeping_time)
        logger.info(f'ending {self.process_name}')



if __name__ == "__main__":
    for i in range(3):
        p = MyProcess(i+1 * 5)
        p.start()

    for p in MyProcess.active_processes:
        logger.info(f"Wait for joining for {p.name}")
        p.join()
        logger.info(f"{p.name} is joined.")

    

