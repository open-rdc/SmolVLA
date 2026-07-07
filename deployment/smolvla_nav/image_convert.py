"""sensor_msgs/Image -> BGR numpy(HWC, uint8) 変換（cv_bridge非依存）。

navigation.py と place_prompt_node.py の両方で使う共通処理。
実機カメラは rgb8/bgr8 とは限らず、USBカメラの生フォーマット(YUV422系)や
mono系を吐くことがあるため、channels を encoding から先に決めてから reshape する
（3チャンネル固定でreshapeすると、encodingが2バイト/画素系の場合に
 `cannot reshape array of size ... into shape (h,w,3)` でクラッシュする）。
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# ROSでよく見るYUV422系encoding名 -> cv2の変換コード
_YUV422_TO_BGR = {
    "yuv422_yuy2": cv2.COLOR_YUV2BGR_YUY2,
    "yuyv": cv2.COLOR_YUV2BGR_YUYV,
    "yuv422": cv2.COLOR_YUV2BGR_UYVY,  # ROSの"yuv422"はUYVYパックが一般的
    "uyvy": cv2.COLOR_YUV2BGR_UYVY,
}


def image_msg_to_bgr(msg) -> Optional[np.ndarray]:
    """対応encoding: rgb8/bgr8/mono8/mono16/yuv422系(yuyv/uyvy)。非対応はNoneを返す。"""
    encoding = msg.encoding.lower()
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)

    if encoding in ("bgr8", "rgb8"):
        frame = np.ascontiguousarray(raw[:, : width * 3]).reshape(height, width, 3)
        return frame if encoding == "bgr8" else cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if encoding == "mono8":
        frame = np.ascontiguousarray(raw[:, :width]).reshape(height, width)
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if encoding == "mono16":
        frame16 = np.ascontiguousarray(raw[:, : width * 2]).reshape(height, width, 2).view(np.uint16).reshape(height, width)
        frame8 = (frame16 >> 8).astype(np.uint8)
        return cv2.cvtColor(frame8, cv2.COLOR_GRAY2BGR)

    if encoding in _YUV422_TO_BGR:
        frame = np.ascontiguousarray(raw[:, : width * 2]).reshape(height, width, 2)
        return cv2.cvtColor(frame, _YUV422_TO_BGR[encoding])

    return None
