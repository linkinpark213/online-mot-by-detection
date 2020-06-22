import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-path', type=str, required=True, help='Path to the input frames')
    parser.add_argument('--out-path', type=str, required=True, help='Path to the output frames')
    parser.add_argument('--in-fps', type=int, required=True, help='Original frame rate')
    parser.add_argument('--out-fps', type=int, required=True, help='Output frame rate')
    parser.add_argument('--start-frame', type=int, required=False, default=0, help='Starting frame number')
    parser.add_argument('--end-frame', type=int, required=False, default=-1, help='Ending frame number')
    parser.add_argument('--ext', type=str, required=False, default='jpg', help='Frame file type (extension)')
    args = parser.parse_args()

    if not os.path.isdir(args.out_path):
        os.makedirs(args.out_path)

    assert args.in_fps >= args.out_fps, 'Input FPS should be higher than output FPS'

    fps_ratio = args.in_fps / args.out_fps
    img_filenames = os.listdir(args.in_path)
    frame_nums = sorted(list(map(lambda x: int(x.split('.')[0]), img_filenames)))

    assert args.start_frame in frame_nums, 'Start frame should be in the input path'
    assert args.end_frame == -1 or args.end_frame in frame_nums, 'End frame should be in the input path'

    i = args.start_frame
    end_frame = frame_nums[-1] if args.end_frame == -1 else args.end_frame
    while i < end_frame:
        img_filename = '{:06d}.'.format(int(i)) + args.ext
        ori_path, dst_path = os.path.join(args.in_path, img_filename), os.path.join(args.out_path, img_filename)
        shutil.copy(ori_path, dst_path)
        print('Copying {} to {}'.format(ori_path, dst_path))
        i += fps_ratio
