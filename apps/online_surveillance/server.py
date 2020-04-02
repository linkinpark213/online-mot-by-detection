import json
import base64
import argparse
import mot.utils
import threading
import socketserver

from mot.tracker import build_tracker
from mot.utils import cfg_from_file


class ThreadedTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        message = self.request.recv(1024)
        message = json.loads(message)
        print('Received message %r' % message)

        current_thread = threading.current_thread()
        ret, frame = self.server.capture.read()
        if not ret:
            response = [{'status': 'Error fetching stream!'}]
            return

        tracker.tick(frame)

        frame = base64.b64encode(frame)

        response = [{'thread': current_thread.name, 'frame': str(frame)[2:-1]}]
        response = json.dumps(response)

        print('Sending length: {}'.format(response.__len__()))

        self.request.sendall(bytearray(response, encoding='utf-8'))


class ThreadedTCPServer(socketserver.TCPServer, socketserver.ThreadingMixIn):
    def __init__(self, address, handler, tracker, capture):
        self.tracker = tracker
        self.capture = capture
        super(ThreadedTCPServer, self).__init__(address, handler)


def run_demo(tracker, args, **kwargs):
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        tracker.tick(frame)

    tracker.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tracker_config', default='configs/deepsort.py')
    parser.add_argument('--port', type=int, default=44213, help='Server port')
    args = parser.parse_args()

    cfg = cfg_from_file(args.tracker_config)
    kwargs = cfg.to_dict()

    tracker = build_tracker(cfg.tracker, **kwargs)
    capture = mot.utils.get_capture('0')

    socketserver.TCPServer.allow_reuse_address = True
    server = ThreadedTCPServer(('', args.port), ThreadedTCPHandler, tracker, capture)
    ip, port = server.server_address

    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    print('Listening at {}:{}'.format(ip, port))

    server.serve_forever()
