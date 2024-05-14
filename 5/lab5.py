import argparse
import threading
import queue
import time
from ultralytics import YOLO
import cv2


def reading(path_video, frame_queue, event_stop):
    cap = cv2.VideoCapture(path_video)
    ind = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Reading error")
            break
        frame_queue.put((frame, ind))
        ind += 1
        time.sleep(0.0001)
    event_stop.set()


def writing(length, fps, queue, path):
    t = threading.current_thread()
    frames = [] * length
    while getattr(t, "do_run", True):
        try:
            frame, ind = queue.get(timeout=1)
            frames[ind] = frame
        except queue.Empty:
            pass

    height, width  = frames[0].shape
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()


def pose(frame_queue, out_queue, event_stop):
    local_model = YOLO(model="yolov8s-pose.pt", verbose=False)
    while True:
        try:
            frame, ind = frame_queue.get(timeout=1)
            results = local_model.predict(source=frame, device='cpu')[0].plot()
            out_queue.put((results, ind))
        except queue.Empty:
            if event_stop.is_set():
                break


def main(arg):
    frame_queue = queue.Queue(1000)
    out_queue = queue.Queue()
    event_stop = threading.Event()
    cap = cv2.VideoCapture(arg.input)

    thread_read = threading.Thread(target=reading, args=(arg.input, frame_queue, event_stop,))
    thread_read.start()

    thread_write = threading.Thread(target=writing, args=(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS), out_queue, arg.output,))
    thread_write.start()

    cap.release()
    start = time.monotonic()
    threads = []
    for _ in range(arg.thread):
        threads.append(threading.Thread(target=pose, args=(frame_queue, out_queue, event_stop,)))

    for thr in threads:
        thr.start()

    for thr in threads:
        thr.join()

    thread_read.join()
    thread_write.do_run = False
    thread_write.join()

    print(f'Time: {time.monotonic() - start}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input.mp4', help='Input video')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video')
    parser.add_argument('--thread', type=int, default=1, help='Number of threads')
    args = parser.parse_args()
    main(args)
