import cv2
import mediapipe as mp
from flask import Flask, jsonify
import threading
import time

# Flaskアプリの初期化
app = Flask(__name__)

# MediaPipeのPoseモジュールをセットアップ
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1,  # 精度重視のモデル (0: 簡易版, 1: 高精度)
    smooth_landmarks=True,  # ランドマークのスムージングを有効にする
    enable_segmentation=False,  # セグメンテーション機能は無効
    smooth_segmentation=True,  # セグメンテーションのスムージング
    min_detection_confidence=0.5,  # 最小検出信頼度 (0.0〜1.0)
    min_tracking_confidence=0.5   # 最小追跡信頼度 (0.0〜1.0)
)

# MediaPipeの描画ユーティリティ
mp_drawing = mp.solutions.drawing_utils

# グローバルな変数でカメラ映像を保持
frame = None
pose_landmarks = None

# カメラの初期化
cap = cv2.VideoCapture(0)

def capture_frame():
    global frame, pose_landmarks
    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        # OpenCVのBGRをRGBに変換（MediaPipeはRGBを使用）
        frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)

        # ポーズ検出を実行
        results = pose.process(frame_rgb)

        # ランドマークの座標をグローバル変数に保存
        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark

        # 最新のフレームを保持
        frame = current_frame

        # 骨格検出結果を描画
        if results.pose_landmarks:
            # ポーズランドマークを描画（関節を線で結ぶ）
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # ランドマークの座標を表示
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                # 座標 (x, y) を画面に表示 (xは幅、yは高さの割合)
                h, w, c = frame.shape  # 画像の高さ、幅、チャンネル数
                x, y = int(landmark.x * w), int(landmark.y * h)  # 座標をピクセルに変換
                # 座標をコンソールに出力
                print(f"Landmark {idx}: ({x}, {y})")
                # 座標をフレーム上に表示
                cv2.putText(frame, f"{idx}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


        

        # 出力ウィンドウに表示
        cv2.imshow('Pose Detection', frame)

        # 'q'キーでウィンドウを閉じる
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # 少し待つ（無限ループ防止）
        time.sleep(0.05)

# 座標情報をAPIで返すエンドポイント
@app.route('/pose', methods=['GET'])
def get_pose():
    global pose_landmarks
    if pose_landmarks:
        landmarks_data = []
        for idx, landmark in enumerate(pose_landmarks):
            landmarks_data.append({
                'landmark_id': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        return jsonify(landmarks_data)
    else:
        return jsonify({"error": "No pose landmarks detected"}), 400

# Flaskを別スレッドで起動
if __name__ == '__main__':
    # 別スレッドでカメラのフレームキャプチャを開始
    capture_thread = threading.Thread(target=capture_frame, daemon=True)
    capture_thread.start()
    
    # Flaskアプリを起動
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
    
    # カメラをリリースしてウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
