# sweeper_detection

## 常用参数
 - 在`detection.py`中切换调试设备
   ```Shell
   seucar_switch = False
   ```
 - 在`param.yaml`中定义一些可配置参数
   ```Shell
   display_mode: True
   record_mode: True
   region_l1: 7
   region_l2: 3
   region_l3: 3
   region_l4: 3
   region_l5: 2
   ```

## 启动
 - 启动相机节点
   ```Shell
   roslaunch my_image_publisher img_publisher.launch
   ```
 - 加载参数文件
   ```Shell
   rosparam load param.yaml
   ```
 - 启动目标检测节点
   ```Shell
   python3 detection.py
   ```

## 动态控制
 - Display detection result:
   ```Shell
   rosparam set /display_mode True
   ```
 - Close display window:
   ```Shell
   rosparam set /display_mode False
   ```
 - Start recording:
   ```Shell
   rosparam set /record_mode True
   ```
 - Save video:
   ```Shell
   rosparam set /record_mode False
   ```


