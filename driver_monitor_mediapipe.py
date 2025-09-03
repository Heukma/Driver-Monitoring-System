import cv2
import mediapipe as mp
import numpy as np
import math

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Media pipe 눈 랜드마크 인덱스
LEFT_EYE_VERTICAL_IDX_1 = [160, 144]
LEFT_EYE_VERTICAL_IDX_2 = [158, 153]
LEFT_EYE_HORIZONTAL_IDX = [33, 133]

RIGHT_EYE_VERTICAL_IDX_1 = [385, 380]
RIGHT_EYE_VERTICAL_IDX_2 = [387, 373]

RIGHT_EYE_HORIZONTAL_IDX = [362, 263]

# EAR Area를 위한 랜드마크
LEFT_EYE_CONTOUR_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# 졸음 판단을 위한 Threshold
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
frame_counter = 0
# Debug 
Debug = True

# Webcam 실행
cap = cv2.VideoCapture(0)

def polygon_area(coords):
    """신발끈 공식을 사용하여 다각형 넓이 계산"""
    return 0.5 * np.abs(np.dot(coords[:, 0], np.roll(coords[:, 1], 1)) - np.dot(coords[:, 1], np.roll(coords[:, 0], 1)))

def calculate_ear_area(eye_contour_idx, landmarks_positions):
    """눈 윤곽선 전체를 사용하여 EAR-Area(면적 비율)를 계산"""
    eye_contour_coords = landmarks_positions[eye_contour_idx]
    eye_area = polygon_area(eye_contour_coords)
    p1 = landmarks_positions[eye_contour_idx[0]]
    p9 = landmarks_positions[eye_contour_idx[8]]
    hor_dist = np.linalg.norm(p1 - p9)
    return eye_area / (hor_dist ** 2) if hor_dist != 0 else 0.0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("웹캠을 찾을 수 없습니다.")
        break

    # Mediapipe 성능 향상을 위한 전처리
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image, 1)

    # MediaPipe 처리
    results = face_mesh.process(image)

    # 다시 쓰기 가능 상태로 변경 및 색상 원복
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, _ = image.shape
    face_found = False

    if results.multi_face_landmarks:
        face_found = True
        for face_landmarks in results.multi_face_landmarks:
            landmarks_positions = np.array(
                [(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark],
                dtype=np.int32
            )

            # Debug: EAR 계산에 사용되는 눈 랜드마크 시각화
            if Debug:
                ear_landmarks_indices = (
                    LEFT_EYE_VERTICAL_IDX_1 + LEFT_EYE_VERTICAL_IDX_2 + LEFT_EYE_HORIZONTAL_IDX +
                    RIGHT_EYE_VERTICAL_IDX_1 + RIGHT_EYE_VERTICAL_IDX_2 + RIGHT_EYE_HORIZONTAL_IDX
                )
                for idx in ear_landmarks_indices:
                    pt = landmarks_positions[idx]
                    cv2.circle(image, tuple(pt), 2, (0, 255, 255), -1) # 노란색 원 그리기

            # EAR 계산
            left_v_dist = (np.linalg.norm(landmarks_positions[LEFT_EYE_VERTICAL_IDX_1[0]] - landmarks_positions[LEFT_EYE_VERTICAL_IDX_1[1]]) + np.linalg.norm(landmarks_positions[LEFT_EYE_VERTICAL_IDX_2[0]] - landmarks_positions[LEFT_EYE_VERTICAL_IDX_2[1]])) / 2
            left_h_dist = np.linalg.norm(landmarks_positions[LEFT_EYE_HORIZONTAL_IDX[0]] - landmarks_positions[LEFT_EYE_HORIZONTAL_IDX[1]])
            
            right_v_dist = (np.linalg.norm(landmarks_positions[RIGHT_EYE_VERTICAL_IDX_1[0]] - landmarks_positions[RIGHT_EYE_VERTICAL_IDX_1[1]]) + np.linalg.norm(landmarks_positions[RIGHT_EYE_VERTICAL_IDX_2[0]] - landmarks_positions[RIGHT_EYE_VERTICAL_IDX_2[1]])) / 2
            right_h_dist = np.linalg.norm(landmarks_positions[RIGHT_EYE_HORIZONTAL_IDX[0]] - landmarks_positions[RIGHT_EYE_HORIZONTAL_IDX[1]])
            
            left_ear = left_v_dist / left_h_dist if left_h_dist != 0 else 0
            right_ear = right_v_dist / right_h_dist if right_h_dist != 0 else 0
            ear = (left_ear + right_ear) / 2

            cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 졸음 탐지
            if ear < EAR_THRESHOLD:
                frame_counter += 1
                if frame_counter >= CONSECUTIVE_FRAMES:
                    cv2.putText(image, "DROWSINESS ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                frame_counter = 0

            # 얼굴 각도 추정 (Head Pose Estimation)
            model_points_3d = np.array([
                (0.0, 0.0, 0.0),             # 1번: 코 끝 (Nose tip)
                (0.0, -330.0, -65.0),        # 199번: 턱 (Chin)
                (-225.0, 170.0, -135.0),     # 33번: 왼쪽 눈의 왼쪽 끝 (Left eye left corner)
                (225.0, 170.0, -135.0),      # 263번: 오른쪽 눈의 오른쪽 끝 (Right eye right corner)
                (-150.0, -150.0, -125.0),    # 61번: 왼쪽 입가 (Left mouth corner)
                (150.0, -150.0, -125.0)      # 291번: 오른쪽 입가 (Right mouth corner)
            ])
            
            # 2. 3D 모델에 대응하는 2D 랜드마크 좌표 추출
            face_2d = []
            # key_landmarks_idx 순서는 위 model_points_3d 순서와 일치
            key_landmarks_idx = [1, 199, 33, 263, 61, 291]
            for idx in key_landmarks_idx:
                lm = face_landmarks.landmark[idx]
                x, y = int(lm.x * img_w), int(lm.y * img_h)
                face_2d.append([x, y])

            face_2d = np.array(face_2d, dtype=np.float64)

            # 3. 카메라 정보 설정 및 solvePnP 실행
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(model_points_3d, face_2d, cam_matrix, dist_matrix)
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                left_ear_area = calculate_ear_area(LEFT_EYE_CONTOUR_IDX, landmarks_positions)
                right_ear_area = calculate_ear_area(RIGHT_EYE_CONTOUR_IDX, landmarks_positions)
                ear_area = (left_ear_area + right_ear_area) / 2
                cv2.putText(image, f"EAR-Area: {ear_area:.3f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # Debug: Pitch, Yaw, Roll 3D 축 시각화
                if Debug:
                    # 3D 축의 끝점을 정의 (축의 길이는 100)
                    axis_3d = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, -100]])
                    # 3D 축을 2D 이미지 평면에 투영
                    axis_2d, _ = cv2.projectPoints(axis_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)
                    
                    # EAR-Area: 눈 전체 윤곽선 (하늘색)
                    cv2.polylines(image, [landmarks_positions[LEFT_EYE_CONTOUR_IDX]], True, (255, 255, 0), 1)
                    cv2.polylines(image, [landmarks_positions[RIGHT_EYE_CONTOUR_IDX]], True, (255, 255, 0), 1)
                    # 코 끝점 (축의 시작점)
                    nose_tip_2d = tuple(landmarks_positions[1])
                    
                    # 각 축을 다른 색상으로 그리기
                    cv2.line(image, nose_tip_2d, tuple(np.int32(axis_2d[0].ravel())), (255, 0, 0), 3) # X축 (빨강)
                    cv2.line(image, nose_tip_2d, tuple(np.int32(axis_2d[1].ravel())), (0, 255, 0), 3) # Y축 (초록)
                    cv2.line(image, nose_tip_2d, tuple(np.int32(axis_2d[2].ravel())), (0, 0, 255), 3) # Z축 (파랑)


                if pitch > 15 or abs(yaw) > 20:
                    cv2.putText(image, "ATTENTION ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    if not face_found:
        cv2.putText(image, "FACE NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Driver Monitor', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()