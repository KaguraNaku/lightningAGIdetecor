"""Native whole-face hull recipes required by ProDet self blending."""

import cv2
import numpy as np


class FaceHullMask:
    """Own the four native hull recipes without importing another detector."""

    @staticmethod
    def create(landmarks, image, kind):
        points = landmarks.astype("int32")
        mask = np.zeros(image.shape[:2] + (1,), dtype=np.float32)
        if kind == 0:
            groups = [
                (points[0:17], points[48:68], points[0:1], points[8:9], points[16:17]),
                (points[27:31], points[33:34]),
                (
                    points[17:27],
                    points[0:1],
                    points[27:28],
                    points[16:17],
                    points[33:34],
                ),
            ]
        elif kind in (1, 2):
            if kind == 1:
                points = points.copy()
                left = (points[36] + points[0]) // 2
                right = (points[16] + points[45]) // 2
                quarter_left = (points[36] + left) // 2
                quarter_right = (points[45] + right) // 2
                bottom_left = np.array(
                    (quarter_left, points[36], points[37], points[38], points[39])
                )
                bottom_right = np.array(
                    (points[42], points[43], points[44], points[45], quarter_right)
                )
                top_left, top_right = points[17:22], points[22:27]
                points[17:22] = top_left + ((top_left - bottom_left) // 2)
                points[22:27] = top_right + ((top_right - bottom_right) // 2)
            groups = [
                (points[0:9], points[17:18]),
                (points[8:17], points[26:27]),
                (points[17:20], points[8:9]),
                (points[24:27], points[8:9]),
                (points[19:25], points[8:9]),
                (points[17:22], points[27:28], points[31:36], points[8:9]),
                (points[22:27], points[27:28], points[31:36], points[8:9]),
                (points[27:31], points[31:36]),
            ]
        elif kind == 3:
            cv2.fillConvexPoly(
                mask,
                cv2.convexHull(np.array(points).reshape((-1, 2))),
                255.0,
                lineType=cv2.LINE_AA,
            )
            return mask[:, :, 0] / 255
        else:
            raise ValueError(
                "Only the four configured whole-face SBI hulls are connected"
            )
        for group in groups:
            cv2.fillConvexPoly(mask, cv2.convexHull(np.concatenate(group)), 255.0)
        return mask[:, :, 0] / 255
