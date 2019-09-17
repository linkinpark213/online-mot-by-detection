from pascal_voc_writer import Writer
import os


def main(frame_path, txt_path, save_path, threshold=0.5):
    f = open(txt_path)

    temp = 1
    writer = Writer(frame_path + 'frame{:05d}.jpg'.format(1), 2560, 1440)
    while True:
        lines = f.readline()
        if lines:
            lines = lines.strip()
            lines_list = lines.split(', ')
            if float(lines_list[5]) > threshold:
                if int(lines_list[0])==temp:
                    img_name = frame_path + 'frame{:05d}.jpg'.format(int(lines[0]))



                    writer.addObject('1', float(lines_list[1])-float(lines_list[3])/2, float(lines_list[2])-float(lines_list[4])/2,
                                     float(lines_list[1]) + float(lines_list[3])/2,
                                     float(lines_list[2]) + float(lines_list[4])/2)

                else:
                    writer.save(save_path + 'frame{:05d}.xml'.format(int(lines_list[0])-1))

                    print('frame{:05d}.xml  saved'.format(int(lines_list[0])-1))
                    img_name = frame_path + 'frame{:05d}.jpg'.format(int(lines[0]))

                    writer = Writer(img_name, 2560, 1440)

                    writer.addObject('1', float(lines_list[1])-float(lines_list[3])/2, float(lines_list[2])-float(lines_list[4])/2,
                                     float(lines_list[1]) + float(lines_list[3])/2,
                                     float(lines_list[2]) + float(lines_list[4])/2)
                    temp=int(lines_list[0])


        else:
            break


if __name__ == '__main__':
    main('/home/rvlab/Documents/level2_video/b2/', '/home/rvlab/Documents/level2_video/b2.txt',
         '/home/rvlab/Documents/level2_video/b2/')
