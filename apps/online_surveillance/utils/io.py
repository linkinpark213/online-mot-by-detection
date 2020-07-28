import os
import cv2
import zmq
import shutil

import mot.utils

from .format import *


class SCTOutputWriter:
    def __init__(self, args, identifier, fps):
        self.args = args
        self.identifier = identifier
        self.fps = fps

        # Both tracked video writer and raw video writer
        self.video_writer = None
        self.raw_video_writer = None

        # Single-cam tracking result writer
        self.result_writer = mot.utils.get_result_writer(args.save_result)
        self.result_writer.write('#{}\n'.format(args.start_time))

        patches_path = os.path.abspath('patches')
        if os.path.isdir('patches'):
            print('Image patch directory `{}` exists. Deleting...'.format(patches_path))
            shutil.rmtree('patches')

        print('Creating image patch directory `{}`'.format(patches_path))
        os.mkdir('patches')

        # Streaming port
        context = zmq.Context()
        self.footage_socket = context.socket(zmq.PUB)
        self.footage_socket.bind('tcp://*:' + str(args.footage_port))
        # Tracker state port
        self.tracker_socket = context.socket(zmq.PUB)
        self.tracker_socket.bind('tcp://*:' + str(args.tracker_port))

    def write(self, tracker, current_time, raw_frame, annotated_image):
        # Save to video if necessary. Video size may change because of extra contents visualized.
        if self.video_writer is None or self.raw_video_writer is None:
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
        self.result_writer.write(snapshot_to_mot(tracker, current_time))

        # Save image samples of currently active identities
        for tracklet in tracker.tracklets_active:
            if tracklet.is_detected():
                image = tracklet.feature['patch']
                cv2.imwrite('patches/g{}_l{}.jpg'.format(tracklet.globalID, tracklet.id), image)

        # Send to streaming port.
        self.footage_socket.send(image_to_base64(annotated_image))

        # Send tracker state to tracker port.
        self.tracker_socket.send(snapshot_to_base64(tracker, self.identifier))

    def close(self):
        self.video_writer.release()
        self.raw_video_writer.release()
        self.result_writer.close()
