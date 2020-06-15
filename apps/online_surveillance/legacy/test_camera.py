import cv2

if __name__ == '__main__':
    cap = cv2.VideoCapture(1)
    out = cv2.VideoWriter('cam_test.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    try:
        while True:
            ret, frame = cap.read()
            print(frame.shape if frame is not None else 'No frame')
            if not ret:
                break
            out.write(frame)
            if cv2.waitKey(1) == 27:
                break
    finally:
        out.release()
        cap.release()
