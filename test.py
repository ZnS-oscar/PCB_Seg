# import numpy as np

# # 假设 x 的形状为 (b, n, c)
# x = np.random.rand(2, 3, 6)  # 示例数据
# b,n,c=x.shape
# # 假设 nm 和 conf_threshold 的值
# nm = 1
# conf_threshold = 0.5

# # 计算条件
# max_values = np.amax(x[..., 4:-nm], axis=-1)  # 形状为 (b, n)
# condition = max_values > conf_threshold  # 形状为 (b, n)

# # 扩展布尔数组的形状
# condition_expanded = np.expand_dims(condition, axis=-1)  # 形状为 (b, n, 1)
# condition_expanded = np.repeat(condition_expanded, c, axis=-1)  # 形状为 (b, n, c)

# # 应用布尔索引
# x_filtered = x[condition_expanded]

# print(x_filtered.shape)
import cv2
# image=cv2.imread("/root/autodl-tmp/sahi/build/simple_test_result.jpeg")
image=cv2.imread("/root/autodl-tmp/sahi/build/the mask.png",0)
image2=0
print("i am here")