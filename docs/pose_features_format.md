# 特征数据格式说明

## 文件位置
- `features/pose_features/train.npy`
- `features/pose_features/val.npy`
- `features/pose_features/test.npy`

## 数据结构
```python
{
    "split1_001.png": {
        "keypoints": {
            "nose": (x, y),
            "left_eye": (x, y),
            "right_eye": (x, y),
            "left_ear": (x, y),
            "right_ear": (x, y),
            "left_shoulder": (x, y),
            "right_shoulder": (x, y),
            "left_elbow": (x, y),
            "right_elbow": (x, y),
            "left_wrist": (x, y),
            "right_wrist": (x, y),
            "left_hip": (x, y),
            "right_hip": (x, y),
            "left_knee": (x, y),
            "right_knee": (x, y),
            "left_ankle": (x, y),
            "right_ankle": (x, y)
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

## 字段说明
| 字段      | 类型      | 说明                               |
| --------- | --------- | ---------------------------------- |
| 文件名    | str       | 图片文件名，如 "split1_001.png"    |
| keypoints | dict/None | 17个关键点坐标，未检测到人时为None |
| meta      | list      | 所有人的标注信息列表               |
| cls       | int       | 类别标签：0(未摔倒), 1(摔倒)       |
| x_center  | float     | 边界框中心x坐标（归一化0-1）       |
| y_center  | float     | 边界框中心y坐标（归一化0-1）       |
| width     | float     | 边界框宽度（归一化0-1）            |
| height    | float     | 边界框高度（归一化0-1）            |

## 注意事项
- keypoints只包含第一个检测到的人
- meta包含图片中所有人的标注
- 空文件时坐标为(None, None)