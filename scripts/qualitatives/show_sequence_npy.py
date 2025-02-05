import os
import cv2
import argparse

import numpy as np
from tqdm import tqdm


def show_sequence(args):
    video_writer = None
    ann_file = os.path.join(args.mot_anns_path, f"{args.sequence}.npy")
    anns = np.load(ann_file)

    # write video with bbox and distance
    prev_frame_anns = None
    for frame_idx in tqdm(range(400)):
        if "mot17" in args.benchmark and frame_idx == 0:
            continue
        # read frame annotations and filter by valid if ground truth is available
        frame_anns = anns[anns[:, 0] == frame_idx]
        if args.use_gt:
            frame_anns = frame_anns[frame_anns[:, 5] == 1]

        if args.benchmark == "motsynth":
            frame_path = (
                f"{args.mot_frames_path}/frames/{args.sequence}/rgb/{frame_idx:04d}.jpg"
            )
        else:
            frame_path = f"{args.mot_frames_path}/{args.split}/{args.sequence}/img1/{frame_idx:06d}.jpg"

        # read frame
        frame = cv2.imread(frame_path)
        assert frame is not None, f"Frame {frame_path} not found"
        w, h = frame.shape[:2][::-1]

        # draw bbox and distance
        for ann in frame_anns:
            bbox = ann[1:5]

            # draw previous bbox if ground truth is available
            if args.use_gt and prev_frame_anns is not None:
                track_id = ann[6]
                prev_frame_ann = prev_frame_anns[prev_frame_anns[:, 6] == track_id]
                if len(prev_frame_ann) > 0:
                    prev_frame_ann = prev_frame_ann[0]
                    prev_bbox = prev_frame_ann[1:5]

                    cv2.line(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(prev_bbox[0]), int(prev_bbox[1])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.rectangle(
                        frame,
                        (int(prev_bbox[0]), int(prev_bbox[1])),
                        (int(prev_bbox[2]), int(prev_bbox[3])),
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{int(track_id)}",
                        (int(prev_bbox[0]), int(prev_bbox[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{int(track_id)}",
                        (int(prev_bbox[0]), int(prev_bbox[1] - 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            # draw bbox
            score = ann[5]
            cv2.rectangle(
                frame,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0, 0, 255),
                2,
            )

            cv2.putText(
                frame,
                f"{score:.1f}",
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"{score:.1f}",
                (int(bbox[0]), int(bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                1,
            )

            if args.show_dist:
                dist = ann[6]
                cv2.putText(
                    frame,
                    f"{dist:.1f}",
                    (int(bbox[0]), int(bbox[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{dist:.1f}",
                    (int(bbox[0]), int(bbox[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    1,
                )
                var = ann[7]
                cv2.putText(
                    frame,
                    f"{var:.2f}",
                    (int(bbox[0]), int(bbox[1] + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{var:.2f}",
                    (int(bbox[0]), int(bbox[1] + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    1,
                )

            assert bbox[0] < bbox[2] and bbox[1] < bbox[3], f"Invalid bbox {bbox}"
            if "mot17" not in args.benchmark:
                assert bbox[0] >= 0 and bbox[1] >= 0, f"Invalid bbox {bbox}"
                assert bbox[2] <= w and bbox[3] <= h, f"Invalid bbox {bbox}"

        cv2.putText(
            frame,
            f"{frame_idx:04d}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2,
        )
        cv2.putText(
            frame,
            f"{frame_idx:04d}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
        )

        prev_frame_anns = frame_anns
        if video_writer is None:
            # create video writer
            video_writer = cv2.VideoWriter(
                os.path.join(args.save_path, f"npy_{args.sequence}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,
                (w, h),
            )
        # write frame
        video_writer.write(frame)

    # release video writer
    video_writer.release()

    saved_video_path = os.path.join(args.save_path, f"npy_{args.sequence}.mp4")
    print(f"Saved video to {saved_video_path}")


def get_args():
    parser = argparse.ArgumentParser(description="Show sequence")
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Benchmark name",
        required=True,
        choices=["mot17", "mot20", "motsynth"],
    )
    parser.add_argument(
        "--split", type=str, help="Split name", choices=["train", "test"]
    )
    parser.add_argument("--sequence", type=str, help="Sequence name", required=True)
    parser.add_argument(
        "--mot_frames_path", type=str, help="MOT path with frames", required=True
    )
    parser.add_argument(
        "--mot_anns_path", type=str, help="MOT path with annotations", required=True
    )
    parser.add_argument(
        "--save_path", type=str, help="Path to save the video", required=True
    )
    parser.add_argument(
        "--use_gt", action="store_true", help="Use ground truth annotations"
    )
    parser.add_argument(
        "--show_dist", action="store_true", help="Show distance and variance"
    )
    return parser.parse_args()


if __name__ == "__main__":
    show_sequence(get_args())
