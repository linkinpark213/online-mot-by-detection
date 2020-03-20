def snapshot_to_mot(tracker, time_lived_threshold: int = 1, ttl_threshold: int = 3, detected_only: bool = True) -> str:
    data = ''
    for tracklet in tracker.tracklets_active:
        if tracklet.time_lived >= time_lived_threshold and tracklet.ttl >= ttl_threshold and (
                tracklet.is_detected() or not detected_only):
            box = tracklet.last_detection.box
            data += '{:d}, {:d}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, -1, -1, -1, -1\n'.format(tracker.frame_num,
                                                                                          tracklet.id,
                                                                                          box[0],
                                                                                          box[1],
                                                                                          box[2] - box[0],
                                                                                          box[3] - box[1])
    return data
