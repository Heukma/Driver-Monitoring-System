import cv2
import mediapipe as mp
# mediapipe openvino - adas dlib
import numpy as np

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Media pipe 눈 랜드마크
LEFT_EYE_VERTICAL_IDX = [159, 145]
LEFT_EYE_HORIZONTAL_IDX = [33, 133]
RIGHT_EYE_VERTICAL_IDX = [386, 374]
RIGHT_EYE_HORIZONTAL_IDX = [362, 263]

# 졸음 판단을 위한 Threshold
# 실험을 통해 수정필요
EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15
frame_counter = 0

# Webcam 실행
cap = cv2.VideoCapture(0)

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

    # Readable만 한 image를 수정
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    img_h, img_w, _ = image.shape
    face_found = False

    if results.multi_face_landmarks:
        # Landmark를 추출하였으면, face가 detected 되었다고 파악
        face_found = True
        for face_landmarks in results.multi_face_landmarks:
            landmarks_positions = np.array(
                [(lm.x * img_w, lm.y * img_h) for lm in face_landmarks.landmark],
                dtype=np.int32
            )

            # EAR 계산
            left_v_dist = np.linalg.norm(landmarks_positions[LEFT_EYE_VERTICAL_IDX[0]] - landmarks_positions[LEFT_EYE_VERTICAL_IDX[1]])
            left_h_dist = np.linalg.norm(landmarks_positions[LEFT_EYE_HORIZONTAL_IDX[0]] - landmarks_positions[LEFT_EYE_HORIZONTAL_IDX[1]])
            
            right_v_dist = np.linalg.norm(landmarks_positions[RIGHT_EYE_VERTICAL_IDX[0]] - landmarks_positions[RIGHT_EYE_VERTICAL_IDX[1]])
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
            face_3d = []
            face_2d = []
            key_landmarks_idx = [33, 263, 1, 61, 291, 199] 

            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in key_landmarks_idx:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z * 3000])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                pitch, yaw, roll = angles[0], angles[1], angles[2]

                cv2.putText(image, f"Pitch: {pitch:.2f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Yaw: {yaw:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.putText(image, f"Roll: {roll:.2f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if pitch > 15 or abs(yaw) > 20:
                    cv2.putText(image, "ATTENTION ALERT!", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # 4. 얼굴 유무 판단
    if not face_found:
        cv2.putText(image, "FACE NOT FOUND", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Driver Monitor', image)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()