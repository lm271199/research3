# -*- coding: utf-8 -*-
"""
实验三：基于 SIFT 特征的图像拼接与融合（image_fusion.py）
流程：
  1. 读入待拼接图像 a.jpg（左）与 b.jpg（右），转灰度图
  2. SIFT 提取关键点与描述子
  3. FLANN（KD 树）做 KNN 匹配，Lowe 比率测试筛选优质匹配
  4. RANSAC 估计单应矩阵 H（b -> a）
  5. 计算拼接画布尺寸并做平移校正
  6. 透视变换 b，与 a 叠加；重叠区域采用加权渐入渐出融合
  7. 裁剪黑边，保存并显示结果

运行：python3 image_fusion.py
依赖：pip install opencv-python numpy
"""

import cv2
import numpy as np

# 1. 读入图像
img_a = cv2.imread("a.jpg")   # 左图（基准图）
img_b = cv2.imread("b.jpg")   # 右图（将被变换到 a 的坐标系）
if img_a is None or img_b is None:
    raise FileNotFoundError("未找到 a.jpg 或 b.jpg，请确认图片与脚本在同一目录。")

gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 2. SIFT 特征提取
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)
kp_b, des_b = sift.detectAndCompute(gray_b, None)
print(f"SIFT 关键点数量:  a={len(kp_a)}, b={len(kp_b)}")

# 3. FLANN 匹配 + Lowe 比率测试
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)          # 检查次数越多越准，但越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des_a, des_b, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:    # 比率阈值 0.7~0.8
        good_matches.append(m)
print(f"KNN 原始匹配: {len(matches)} 对，比率测试后优质匹配: {len(good_matches)} 对")

matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches, None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

if len(good_matches) < 4:
    raise RuntimeError("优质匹配点不足 4 对，无法估计单应矩阵。")

# 4. RANSAC 估计单应矩阵
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(f"RANSAC 内点数: {int(mask.sum())} / {len(good_matches)}")
print("单应矩阵 H =\n", H)

# ---------- 5. 计算画布尺寸 ----------
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

corners_b = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
corners_b_t = cv2.perspectiveTransform(corners_b, H)
corners_a = np.float32([[0, 0], [0, h_a], [w_a, h_a], [w_a, 0]]).reshape(-1, 1, 2)
all_corners = np.concatenate([corners_b_t, corners_a], axis=0)

[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

translation_matrix = np.array([[1, 0, -x_min],
                               [0, 1, -y_min],
                               [0, 0, 1]], dtype=np.float64)
T_H = translation_matrix @ H

# 6. 透视变换 + 加权融合
canvas = cv2.warpPerspective(img_b, T_H, (x_max - x_min, y_max - y_min))

# a 在画布中的放置位置
x0, y0 = -x_min, -y_min
stitched_direct = canvas.copy()                      
stitched_direct[y0:y0 + h_a, x0:x0 + w_a] = img_a

# 加权渐入渐出融合版
stitched_fused = canvas.copy()
roi = stitched_fused[y0:y0 + h_a, x0:x0 + w_a]
# 重叠区域掩码：b 已有像素处为重叠区
b_mask = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
overlap = b_mask[y0:y0 + h_a, x0:x0 + w_a] > 0
# 权重沿 x 方向线性过渡：a 的权重从重叠区左缘 1 渐变到右缘 0
xs = np.tile(np.linspace(0, 1, w_a), (h_a, 1))
alpha = np.where(overlap, 1.0 - xs, 1.0)[..., None]   # a 的权重
roi[:] = (img_a * alpha + roi * (1 - alpha)).astype(np.uint8)
stitched_fused[y0:y0 + h_a, x0:x0 + w_a] = roi

# 7. 裁剪黑边
def crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

stitched_direct = crop_black(stitched_direct)
stitched_fused = crop_black(stitched_fused)

# 8. 保存结果
cv2.imwrite("result_matches.jpg", matched_keypoints_img)
cv2.imwrite("result_direct.jpg", stitched_direct)
cv2.imwrite("result_fused.jpg", stitched_fused)
print("已保存: result_matches.jpg / result_direct.jpg / result_fused.jpg")

# 9. 显示
import sys
if "--show" in sys.argv:
    cv2.imshow("Matched SIFT Keypoints", cv2.resize(matched_keypoints_img, None, fx=0.4, fy=0.4))
    cv2.imshow("Stitched (direct)", cv2.resize(stitched_direct, None, fx=0.4, fy=0.4))
    cv2.imshow("Stitched (fused)", cv2.resize(stitched_fused, None, fx=0.4, fy=0.4))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
