import zmq

import mot.utils

from .format import *


class SCTOutputWriter:
    def __init__(self, args, identifier, fps):
        self.args = args
        self.identifier = identifier
        self.fps = fps
        self.video_writer = None
        self.raw_video_writer = None
        self.result_writer = mot.utils.get_result_writer(args.save_result)
        self.result_writer.write('#{}'.format(args.start_time))

        # Streaming port
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.bind('tcp://*:' + str(args.footage_port))
        # Tracker state port
        self.tracker_socket = context.socket(zmq.PUB)
        self.tracker_socket.bind('tcp://*:' + str(args.tracker_port))

    def write(self, tracker, raw_frame, annotated_image):
        # Save to video if necessary. Video size may change because of extra contents visualized.
        if tracker.frame_num == 1:
            self.video_writer = mot.utils.get_video_writer(self.args.save_video, annotated_image.shape[1],
                                                           annotated_image.shape[0], self.fps)
            self.video_writer = mot.utils.RealTimeVideoWriterWrapper(self.video_writer, self.fps)

            self.raw_video_writer = mot.utils.get_video_writer(str(self.args.save_video)[:-4] + '_raw.mp4',
                                                               raw_frame.shape[1],
                                                               raw_frame.shape[0],
                                                               self.fps)
            self.raw_video_writer = mot.utils.RealTimeVideoWriterWrapper(self.raw_video_writer, self.fps)

        self.video_writer.write(annotated_image)
        self.raw_video_writer.write(raw_frame)

        # Save to result file if necessary.
        self.result_writer.write(snapshot_to_mot(tracker))

        # Send to streaming port.
        self.footage_socket.send(image_to_base64(annotated_image))

        # Send tracker state to tracker port.
        self.tracker_socket.send(snapshot_to_base64(tracker, self.identifier))

    def close(self):
        self.video_writer.release()
        self.raw_video_writer.release()
        self.result_writer.close()
