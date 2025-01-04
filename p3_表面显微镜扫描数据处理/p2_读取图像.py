from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from phy_base.scripts.base import PhyCommand

from p2_点云数据处理.config import base_dir, specimen_name


class P2_读取图像(PhyCommand):
    @property
    def raw_image(self) -> np.ndarray:
        path = base_dir / specimen_name / 'optical-image-raw.png'
        with Image.open(path) as img:
            return np.array(img.convert('RGB'))

    @property
    def cropped(self) -> np.ndarray:
        steps_dir = base_dir / specimen_name / 'optical-image-steps'
        steps_dir.mkdir(parents=True, exist_ok=True)

        image = self.raw_image.copy()
        orig = image.copy()

        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        self.save_step(steps_dir, 1, '1-hls.png', hls)

        L_channel = hls[:, :, 1]
        self.save_step(steps_dir, 2, '2-lightness.png', L_channel)

        _, fixed_thresh = cv2.threshold(L_channel, 15, 255, cv2.THRESH_BINARY)
        self.save_step(steps_dir, 3, '3-fixed_threshold.png', fixed_thresh)

        lines_img = orig.copy()
        edges = cv2.Canny(fixed_thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
        if lines is None:
            raise ValueError("未检测到任何线条。")
        horizontal = []
        vertical = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if abs(angle) < 10:
                horizontal.append(line[0])
            elif abs(angle) > 80:
                vertical.append(line[0])
        if len(horizontal) < 2 or len(vertical) < 2:
            raise ValueError("未检测到足够的水平或垂直线条。")
        horizontal = sorted(horizontal, key=lambda x: x[1])
        vertical = sorted(vertical, key=lambda x: x[0])
        top_line = horizontal[-2]
        bottom_line = horizontal[-1]
        left_line = vertical[0]
        right_line = vertical[-1]
        cv2.line(lines_img, (top_line[0], top_line[1]), (top_line[2], top_line[3]), (255, 0, 0), 2)
        cv2.line(lines_img, (bottom_line[0], bottom_line[1]), (bottom_line[2], bottom_line[3]), (255, 0, 0), 2)
        cv2.line(lines_img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 2)
        cv2.line(lines_img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (255, 0, 0), 2)
        self.save_step(steps_dir, 4, '4-detected_lines.png', lines_img)

        def compute_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom == 0:
                return None
            Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            return [Px, Py]

        intersections = []
        intersections.append(compute_intersection(top_line, left_line))
        intersections.append(compute_intersection(top_line, right_line))
        intersections.append(compute_intersection(bottom_line, right_line))
        intersections.append(compute_intersection(bottom_line, left_line))
        if None in intersections:
            raise ValueError("无法计算所有交点。")
        intersections = np.array(intersections, dtype="float32")
        ordered_points = self.order_points(intersections)
        ordered_img = orig.copy()
        for point in ordered_points:
            cv2.circle(ordered_img, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        cv2.polylines(ordered_img, [ordered_points.astype(int)], True, (0, 0, 255), 2)
        self.save_step(steps_dir, 5, '5-ordered_points.png', ordered_img)

        (tl, tr, br, bl) = ordered_points
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = max(int(heightA), int(heightB))

        max_dim = max(maxWidth, maxHeight)
        dst = np.array([
            [0, 0],
            [max_dim - 1, 0],
            [max_dim - 1, max_dim - 1],
            [0, max_dim - 1]
        ], dtype="float32")
        self.save_step(steps_dir, 6, '6-destination_square.png', self.draw_destination_square(dst, max_dim))

        M = cv2.getPerspectiveTransform(ordered_points, dst)
        warped = cv2.warpPerspective(orig, M, (max_dim, max_dim))
        self.save_step(steps_dir, 7, '7-warped.png', warped)

        return warped

    def handle(self, **options):
        cropped_image = self.cropped
        if cropped_image.dtype != np.uint8:
            cropped_image = cropped_image.astype(np.uint8)
        path = base_dir / specimen_name / 'optical-image.png'
        Image.fromarray(cropped_image).save(path)

    def save_step(self, steps_dir: Path, step_number: int, filename: str, image: np.ndarray):
        step_path = steps_dir / filename
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_HLS2RGB)
        elif len(image.shape) == 2:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_to_save = image
        Image.fromarray(image_to_save).save(step_path)

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def draw_destination_square(self, dst: np.ndarray, max_dim: int) -> np.ndarray:
        square_image = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        for point in dst:
            cv2.circle(square_image, tuple(point.astype(int)), 5, (0, 255, 0), -1)
        return square_image


if __name__ == '__main__':
    P2_读取图像.main()
