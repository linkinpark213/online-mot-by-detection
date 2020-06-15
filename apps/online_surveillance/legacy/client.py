import cv2
import json
import time
import socket
import base64
import numpy as np

if __name__ == '__main__':
    while True:
        # time.sleep(1)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        sock.settimeout(5)
        sock.connect(('163.221.68.100', 44213))

        message = json.dumps([{'timestamp': time.localtime(time.time())}])
        print('Sending length: {}'.format(len(message)))

        sock.sendall(str.encode(message, encoding='utf-8'))

        message = b''
        while True:
            response = sock.recv(1280000)
            if not response:
                break
            message += response
        print('Response length: {}'.format(len(message)))

        message = json.loads(message)
        print(message[0].keys())
        frame = message[0]['frame']
        frame = base64.b64decode(frame)
        frame = np.frombuffer(frame, dtype=np.uint8)
        frame = np.reshape(frame, (480, 640, 3))
        print(frame.shape)

        cv2.imshow('Remote camera', frame)
        cv2.waitKey(1)

        sock.close()
