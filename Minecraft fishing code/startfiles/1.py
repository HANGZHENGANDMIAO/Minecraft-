import cv2
import torch
import numpy as np
import pyautogui
import mss
import time
from PIL import Image
import tkinter as tk
from tkinter import Label

# ============== 配置区域 ==============
SCREEN_REGION = {'top': 350, 'left': 250, 'width': 1200, 'height': 500}
YOLO_MODEL_PATH = 'best.pt'  # 替换为你的YOLO模型路径
LOCAL_YOLOV5_PATH = r"C:\Users\HZ\yolov5"  # 替换为你本地YOLOv5仓库的路径
MOVE_THRESHOLD = 15  # 触发收杆的浮动幅度阈值
# 移除置信度阈值的配置
# CONFIDENCE_THRESHOLD = 0.7
# =====================================

def load_yolov5_model():
    """加载YOLOv5模型"""
    try:
        model = torch.hub.load(LOCAL_YOLOV5_PATH, 'custom', path=YOLO_MODEL_PATH, source='local')
        # 移除设置置信度阈值的代码
        # model.conf = CONFIDENCE_THRESHOLD
        return model
    except Exception as e:
        print(f"加载YOLOv5模型时出错: {e}")
        return None

def grab_screen(sct):
    """捕获屏幕画面"""
    try:
        img = sct.grab(SCREEN_REGION)
        return Image.frombytes('RGB', img.size, img.bgra, 'raw', 'BGRX')
    except Exception as e:
        print(f"屏幕捕获出错: {e}")
        return None

def detect_buoy(model, img):
    """YOLO检测浮标位置"""
    if img is None:
        return None
    try:
        results = model(img)  # 推理
        df = results.pandas().xyxy[0]  # 获取检测结果DataFrame

        if not df.empty and 'buoy' in df['name'].values:  # 假设类别名为'buoy'
            # 取置信度最高的检测结果（这里其实不考虑置信度了，只是按原逻辑取一个结果）
            best = df[df['confidence'] == df['confidence'].max()].iloc[0]
            x1, y1, x2, y2 = int(best['xmin']), int(best['ymin']), int(best['xmax']), int(best['ymax'])
            return ((x1 + x2) // 2, (y1 + y2) // 2)  # 返回中心坐标
        return None
    except Exception as e:
        print(f"浮标检测出错: {e}")
        return None

def cast_rod():
    """抛竿动作"""
    pyautogui.rightClick()
    time.sleep(2)  # 等待浮标出现
    return True

def reel_in():
    """收竿动作"""
    pyautogui.rightClick()
    return False

def main():
    # 创建 Tkinter 窗口
    root = tk.Tk()
    root.title("浮标检测信息")
    root.geometry("300x200")  # 调整窗口高度以容纳新标签
    root.attributes("-topmost", True)  # 窗口置顶

    # 创建标签用于显示浮标信息
    buoy_info_label = Label(root, text="未检测到浮标")
    buoy_info_label.pack(pady=10)

    # 创建标签用于显示浮标竖直方向移动数值
    movement_label = Label(root, text="竖直移动幅度: 0")
    movement_label.pack(pady=10)

    # 创建标签用于显示收杆时的竖直移动距离
    reel_in_movement_label = Label(root, text="收杆竖直移动距离: 0")
    reel_in_movement_label.pack(pady=10)

    # 加载YOLOv5模型
    model = load_yolov5_model()
    if model is None:
        return

    # 初始化屏幕捕获
    sct = mss.mss()
    prev_y = None
    casting = False
    buoy_detected_prev = False  # 记录上一帧是否检测到浮标

    # 初始抛竿
    casting = cast_rod()

    def update_ui():
        nonlocal prev_y, casting, buoy_detected_prev
        try:
            # 捕获屏幕
            img = grab_screen(sct)
            if img is None:
                buoy_info_label.config(text="屏幕捕获出错")
                movement_label.config(text="竖直移动幅度: 0")
                reel_in_movement_label.config(text="收杆竖直移动距离: 0")
                root.after(100, update_ui)
                return

            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # YOLO检测浮标
            buoy_pos = detect_buoy(model, img)
            buoy_detected = bool(buoy_pos)  # 当前帧是否检测到浮标

            move = 0
            if buoy_detected:
                current_y = buoy_pos[1]
                if prev_y is not None:
                    # 计算纵向移动幅度
                    move = abs(current_y - prev_y)
                    if move > MOVE_THRESHOLD:
                        print(f"检测到下沉！幅度: {move}，收杆中...")
                        casting = reel_in()
                        # 显示收杆时的竖直移动距离
                        reel_in_movement_label.config(text=f"收杆竖直移动距离: {move}")
                        # 收杆后等待 0.5 秒再抛竿
                        time.sleep(0.5)
                        # 收杆后立即抛竿
                        casting = cast_rod()
                        # 重置 prev_y 以重新计算竖直距离差值
                        prev_y = None
                        buoy_info_label.config(text=f"检测到下沉！幅度: {move}，收杆并重新抛竿")
                        movement_label.config(text=f"竖直移动幅度: {move}")
                    else:
                        buoy_info_label.config(text=f"检测到浮标，位置: {buoy_pos}，移动幅度: {move}")
                        movement_label.config(text=f"竖直移动幅度: {move}")
                else:
                    buoy_info_label.config(text=f"检测到浮标，位置: {buoy_pos}")
                    movement_label.config(text="竖直移动幅度: 0")
                prev_y = current_y
            elif not buoy_detected and buoy_detected_prev and casting:
                # 浮标消失，执行收杆操作
                print("检测到浮标消失，收杆中...")
                casting = reel_in()
                # 收杆后等待 0.5 秒再抛竿
                time.sleep(0.5)
                # 收杆后立即抛竿
                casting = cast_rod()
                # 重置 prev_y 以重新计算竖直距离差值
                prev_y = None
                buoy_info_label.config(text="检测到浮标消失，收杆并重新抛竿")
                movement_label.config(text="竖直移动幅度: 0")
                reel_in_movement_label.config(text="收杆竖直移动距离: 0")
            else:
                buoy_info_label.config(text="未检测到浮标")
                movement_label.config(text="竖直移动幅度: 0")
                reel_in_movement_label.config(text="收杆竖直移动距离: 0")

            if not buoy_detected:
                prev_y = None

            buoy_detected_prev = buoy_detected  # 更新上一帧的检测结果

            # 显示检测结果（调试用）
            if buoy_pos:
                cv2.circle(frame, buoy_pos, 5, (0, 255, 0), -1)
            cv2.imshow('Auto Fishing', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                root.destroy()
                return

        except Exception as e:
            print(f"循环中出现错误: {e}")
            buoy_info_label.config(text=f"出现错误: {e}")
            movement_label.config(text="竖直移动幅度: 0")
            reel_in_movement_label.config(text="收杆竖直移动距离: 0")

        root.after(100, update_ui)

    # 开始更新 UI
    update_ui()

    # 运行 Tkinter 主循环
    root.mainloop()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# 注意：如果出现 torch.cuda.amp.autocast 弃用警告，需要手动修改 YOLOv5 代码
# 找到 C:\Users\HZ\yolov5\models\common.py 文件（根据实际路径修改）
# 将 with amp.autocast(autocast): 替换为 with torch.amp.autocast(device_type='cuda'):