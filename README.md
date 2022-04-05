# DeepSORT_YOLOv4
A implementation of Multiple Object Tracking with YOLOv4

## 環境
python 3.7

tensorflow 2.3.0

## 使用流程
1. 先使用https://github.com/EthanYang0213/YOLOv4_tensorflow2 訓練自己的YOLOv4權重
2. 訓練時要將模型儲存成完整的h5檔案 (model.save('yolo_weight.h5'), 在model.load_weights後使用)
3. h5權重和類別文件要一致(相同的類別)
4. 修改yolo.py
    - self.model_path = 'model_data/Yolo_model.h5'
    - self.anchors_path = 'model_data/yolo_anchors.txt'
    - self.classes_path = 'model_data/coco_classes.txt'
    - predicted_class 設定要偵測的類別
5. 修改demo.py
    - model_filename = 'model_data/mars-small128.pb' (可選擇使用的DeepSORT權重)
    - file_path = 'test.mp4'
    - fourcc = cv2.VideoWriter_fourcc(*'XVID') # 解析格式要跟影片格式配對
    - out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
    - 其他影片相關格式設定和BBox格式皆可修改
    - 以counter來計數
