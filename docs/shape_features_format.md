# 形状特征数据格式说明

## 文件位置
- `features/shape_features/train.npy`
- `features/shape_features/val.npy`
- `features/shape_features/test.npy`

## 数据结构
```python
{
    "image_name.png": {
        "shape_features": {
            # 整体形状特征
            "flat_ratio": float,              # 扁平率 (σy/σx)
            "span_ratio": float,              # 跨度比 (y_span/x_span)

            # 躯干与脚踝距离
            "torso_ankle_distance": float,    # 欧氏距离
            "torso_ankle_distance_x": float,  # x方向距离
            "torso_ankle_distance_y": float,  # y方向距离

            # 躯干与头部距离
            "torso_head_distance_x": float,   # x方向距离
            "torso_head_distance_y": float,   # y方向距离

            # 角度特征
            "trunk_angle": float,             # 躯干与y轴夹角 (0°~90°)
            "left_thigh_angle": float,        # 左大腿与y轴夹角
            "right_thigh_angle": float,       # 右大腿与y轴夹角
            "left_shin_angle": float,         # 左小腿与y轴夹角
            "right_shin_angle": float         # 右小腿与y轴夹角
        },
        "meta": [
            {
                "cls": 0,  # 0:未摔倒, 1:摔倒
                "x_center": x_center,  # 边界框中心x坐标（归一化）
                "y_center": y_center,  # 边界框中心y坐标（归一化）
                "width": width,  # 边界框宽度（归一化）
                "height": height  # 边界框高度（归一化）
            }
        ]
    }
}
```

## 特征说明

### 形状特征
| 特征名称 | 类型 | 说明 | 典型值范围 |
|---------|------|------|----------|
| flat_ratio | float/None | 扁平率，σy/σx | 站立>1, 躺倒<1 |
| span_ratio | float/None | 跨度比，y_span/x_span | 站立>2, 躺倒<0.5 |

### 距离特征
| 特征名称 | 类型 | 说明 | 物理意义 |
|---------|------|------|----------|
| torso_ankle_distance | float/None | 躯干到脚踝的欧氏距离 | 身体伸展程度 |
| torso_ankle_distance_x | float/None | 水平分离度 | 站立时小，躺倒时可能大 |
| torso_ankle_distance_y | float/None | 垂直分离度 | 站立时大，躺倒时小 |
| torso_head_distance_x | float/None | 头部水平偏移 | 身体倾斜指标 |
| torso_head_distance_y | float/None | 头部垂直偏移 | 站立时大，躺倒时小 |

### 角度特征
| 特征名称 | 类型 | 说明 | 典型值范围 |
|---------|------|------|----------|
| trunk_angle | float/None | 躯干与y轴夹角 | 0°(站立)~90°(躺倒) |
| left_thigh_angle | float/None | 左大腿与y轴夹角 | 0°(站立)~90°(坐/躺) |
| right_thigh_angle | float/None | 右大腿与y轴夹角 | 0°(站立)~90°(坐/躺) |
| left_shin_angle | float/None | 左小腿与y轴夹角 | 0°(站立)~90°(躺倒) |
| right_shin_angle | float/None | 右小腿与y轴夹角 | 0°(站立)~90°(躺倒) |

## 注意事项
- 当关键点缺失或无法计算时，对应特征值为None
- 角度单位为度数（degree），不是弧度
- 距离特征的单位为像素（未归一化）
- meta字段保留原始标注信息，格式与pose_features相同

## 特征应用
这些形状特征主要用于：
1. **姿态分类**: 区分站立、坐着、躺倒等基本姿态
2. **摔倒检测**: 通过角度和距离的突变识别摔倒
3. **动作识别**: 通过左右腿角度差异识别步态
4. **异常检测**: 识别不正常的身体姿态