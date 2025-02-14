import requests
import cv2
import numpy as np
import time

# APIのエンドポイントURL
url = "http://localhost:5000/pose"

# ウィンドウサイズ
window_width, window_height = 640, 480

# OpenCVウィンドウの設定
cv2.namedWindow("Pose Landmarks", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose Landmarks", window_width, window_height)

# ランドマークデータを保存する辞書
landmarks_dict = {}

while True:
    try:
        # APIからデータを取得
        response = requests.get(url)

        if response.status_code == 200:
            landmarks = response.json()  # JSONをパース
            landmarks_dict.clear()  # 前のデータをクリア

            # 黒い背景画像を作成
            background = np.zeros((window_height, window_width, 3), dtype=np.uint8)

            # 各ランドマークの座標情報を整理
            for landmark in landmarks:
                landmark_id = landmark['landmark_id']
                x = int(landmark['x'] * window_width)  # 画面サイズに合わせてスケール
                y = int(landmark['y'] * window_height)

                # ランドマークのデータを保存
                landmarks_dict[landmark_id] = {'x': x, 'y': y, 'z': landmark['z']}

                # ランドマークIDと座標を表示
                text = f"ID: {landmark_id} ({x}, {y})"
                cv2.putText(background, text, (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 画面に表示
            cv2.imshow("Pose Landmarks", background)

        else:
            print(f"Error: Unable to fetch data. Status code: {response.status_code}")

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 0.1秒待機（適宜調整可能）
        time.sleep(0.1)

    except Exception as e:
        print(f"An error occurred: {e}")
        break

# ウィンドウを閉じる
cv2.destroyAllWindows()