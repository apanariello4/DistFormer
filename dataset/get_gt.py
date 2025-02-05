import dataset.kitti.kitti_utils as utils
from dataset.kitti.calibration import Calibration
import matplotlib.pyplot as plt
import numpy as np


def parse_line(line):
    label = line.strip().split(" ")
    cat = label[0]
    h, w, l = [label[8]], [label[9]], [label[10]]
    loc = label[11:14]
    heading = [label[14]]
    boxes = loc + l + h + w + heading
    return np.array(boxes, dtype=np.float32), cat


def load_3d_boxes(path):
    with open(path, "r") as f:
        lines = f.readlines()

    boxes = [parse_line(line)[0] for line in lines]

    # print(f'{len(boxes)} boxes in the point clouds')
    assert len(boxes) != 0

    return np.array(boxes)


def points_in_img_mask(points, H, W):
    return (
        (points[:, 0] >= 0)
        & (points[:, 0] < W)
        & (points[:, 1] >= 0)
        & (points[:, 1] < H)
    )


def gt_distances(dataset, ds_path, frame):
    points_path = f"{ds_path}/{dataset}/raw/training/velodyne/{frame}.bin"
    label_path = f"{ds_path}/{dataset}/raw/training/label_2/{frame}.txt"
    calib_path = f"{ds_path}/{dataset}/raw/training/calib/{frame}.txt"

    calib = Calibration(calib_path)
    points = utils.load_point_clouds(points_path)
    bboxes = load_3d_boxes(label_path)
    bboxes = calib.bbox_rect_to_lidar(bboxes)

    corners3d = utils.boxes_to_corners_3d(bboxes)
    points_flag = utils.is_within_3d_box(points, corners3d)

    distances = []
    for i in range(len(bboxes)):
        if np.sum(points_flag[i]) == 0:
            distances.append(-1)
            continue

        # points in box sorted by z coordinate (closest to the camera)
        points_in_box = points[points_flag[i]]
        points_in_box = points_in_box[points_in_box[:, 2].argsort()]

        # add a column of ones to the points
        points_in_box = np.concatenate(
            [points_in_box, np.ones((points_in_box.shape[0], 1))], axis=1
        )

        # get the 10% closest points to the camera
        keypoint_idx = int(0.1 * len(points_in_box))

        # transform points to camera coordinates
        pts3d_cam = calib.R0.dot(calib.V2C.dot(points_in_box.transpose()))

        # option for set distance considering only points in the image

        # H, W = plt.imread(img_path).shape[:2]
        # pts2d_cam = calib.P2.dot(np.vstack((pts3d_cam, np.ones((1, pts3d_cam.shape[1]))))) / pts3d_cam[2, :]
        # pts2d_cam = np.concatenate([pts2d_cam[:2, :].transpose(), np.ones((pts2d_cam.shape[1], 1)) * i], axis=1)
        # all_cam_points = np.concatenate([all_cam_points, pts2d_cam], axis=0)

        # pts_in_img_mask = points_in_img_mask(pts2d_cam, H, W)
        # pts2d_cam = pts2d_cam[pts_in_img_mask]
        # pts3d_cam = pts3d_cam[:, pts_in_img_mask]v
        # keypoint_idx_2d = int(0.1 * len(pts2d_cam))

        # print(f'distance all pts: {pts3d_cam[2, keypoint_idx]:.4f} - '
        #     f'distance in img: {pts3d_cam[2, keypoint_idx_2d]:.4f}')
        # all_cam_points = np.array(all_cam_points)

        distances.append(pts3d_cam[2, keypoint_idx])

    return distances


def vis_pcl(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c="r", marker="o")
    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


def vis_img(img_path, points):
    img = plt.imread(img_path)
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 2], marker="o", s=5)
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    gt_distances("000936")
