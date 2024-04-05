import argparse
import multiprocessing
import time
from queue import Empty
import cv2
import numpy as np
import logging

logging.basicConfig(filename='4.log', level=logging.ERROR)

def clear(q):
    try:
        while True:
            q.get_nowait()
    except Empty:
        pass

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    '''Sensor X'''
    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, cam, res):
        if cam == 'default':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(int(cam))
        if not self.cap.isOpened():
            logging.error(f"Failed to open camera with name: {cam}")
            raise ValueError(f"Failed to open camera with name: {cam}")

        self.cap.set(3, res[0])
        self.cap.set(4, res[1])

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error(f"Failed to read frame from camera")
            raise RuntimeError(f"Failed to read frame from camera")
        return frame

    def __del__(self):
        self.cap.release()


class WindowImage:
    def __init__(self, freq):
        self.freq = freq
        cv2.namedWindow("window")

    def show(self, img, s1, s2, s3):
        try:
            x = int(img.shape[1]/5)
            y = int(img.shape[0]/8)
            text = f"Sensor 1: {s1} Sensor 2: {s2} Sensor 3: {s3}"
            cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imshow("window", img)
            time.sleep(1/self.freq)
        except Exception as e:
            logging.error(f"Error occurred while displaying frame: {e}")

    def __del__(self):
        try:
            cv2.destroyWindow("window")
        except Exception as e:
            logging.error(f"Error occurred while destroying window: {e}")


def process(que, sensor):
    while True:
        if que.qsize() == 3:
            clear(que)
        new_val = sensor.get()
        while que.full():
            que.get()
        que.put(new_val)


def main(args):
    try:
        picsize = (int(args.res.split('*')[0]), int(args.res.split('*')[1]))
        sensor1 = SensorX(1)
        sensor2 = SensorX(0.1)
        sensor3 = SensorX(0.01)
        window = WindowImage(args.freq)
        camera = SensorCam(args.cam, picsize)

        queue1 = multiprocessing.Queue()
        queue2 = multiprocessing.Queue()
        queue3 = multiprocessing.Queue()

        read1 = multiprocessing.Process(target=process, args=(queue1, sensor1))
        read2 = multiprocessing.Process(target=process, args=(queue2, sensor2))
        read3 = multiprocessing.Process(target=process, args=(queue3, sensor3))

        read1.start()
        read2.start()
        read3.start()

        val1 = val2 = val3 = 0
        while True:
            if not queue1.empty():
                val1 = queue1.get()
            if not queue2.empty():
                val2 = queue2.get()
            if not queue3.empty():
                val3 = queue3.get()
            valim = camera.get()

            window.show(valim, val1, val2, val3)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        read1.terminate()
        read2.terminate()
        read3.terminate()

        read1.join()
        read2.join()
        read3.join()
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        raise SystemExit("Program terminated due to error: ", str(e))

if __name__ == '__main__':
    cv2.destroyAllWindows()
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam', type=str, default='default', help='Camera name')
    parser.add_argument('--res', type=str, default='900*900', help='Camera resolution')
    parser.add_argument('--freq', type=int, default=40, help='Window frequency')
    args = parser.parse_args()
    main(args)
