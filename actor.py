import cv2
import numpy as np

# ========== 1. ЗАГРУЗКА ИЗОБРАЖЕНИЙ ==========
img_path = 'C:\\Users\\User\\Desktop\\pythonproject\\stathamactor.png'
glasses_path = 'C:\\Users\\User\\Desktop\\pythonproject\\glasses.png'

img = cv2.imread(img_path)
glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"Ошибка: не удалось загрузить {img_path}")
    exit()

if glasses_img is None:
    print(f"Ошибка: не удалось загрузить {glasses_path}")
    exit()

# ========== 2. ПОИСК ЛИЦА ==========
img_processed = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')

if face_cascade.empty() or eye_cascade.empty():
    print("Ошибка: не удалось загрузить каскады Хаара")
    exit()

faces = face_cascade.detectMultiScale(gray, 1.1, 4)
if len(faces) == 0:
    print("Лица не обнаружены")
    exit()

(x, y, w, h) = faces[0]

# ========== 3. РИСУЕМ ОВАЛ ВОКРУГ ЛИЦА ==========
center = (x + w // 2, y + h // 2)
axes = (w // 2, int(h * 0.6))
cv2.ellipse(img_processed, center, axes, 0, 0, 360, (0, 255, 0), 2)

# ========== 4. ПОИСК ГЛАЗ ДЛЯ ЗАЩИТНОЙ МАСКИ ==========
face_roi_gray = gray[y:y + h, x:x + w]
eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.05, minNeighbors=1, minSize=(20, 20))

glasses_width = glasses_height = 0
eye_left_x = eyes_center_y = 0
eye_rects = []

# Определяем положение очков
if len(eyes) >= 1:
    print(f"Найдено глаз: {len(eyes)}")
    eyes = sorted(eyes, key=lambda e: e[0])
    
    if len(eyes) == 1:
        ex, ey, ew, eh = eyes[0]
        eye_rects.append((x + ex, y + ey, ew, eh))
        second_eye_x = x + w - ex - ew
        eye_rects.append((second_eye_x, y + ey, ew, eh))
        print("Создан симметричный второй глаз")
    else:
        for (ex, ey, ew, eh) in eyes[:2]:
            eye_rects.append((x + ex, y + ey, ew, eh))
    
    left_eye, right_eye = eye_rects[0], eye_rects[-1]
    eye_left_x = left_eye[0]
    eye_right_x = right_eye[0] + right_eye[2]
    eyes_center_y = (left_eye[1] + right_eye[1]) // 2
    
    glasses_width = eye_right_x - eye_left_x + 60
    glasses_height = int(glasses_width * 0.35)
    
else:
    print("Глаза не найдены, используем приблизительную область")
    eye_left_x = x + w // 8
    glasses_width = w - w // 4
    glasses_height = int(glasses_width * 0.4)
    eyes_center_y = y + h // 3
    
    left_eye_w = glasses_width // 4
    right_eye_w = left_eye_w
    left_eye_x = eye_left_x + glasses_width // 6
    right_eye_x = left_eye_x + glasses_width // 2
    eye_y = eyes_center_y
    
    eye_rects.append((left_eye_x, eye_y, left_eye_w, left_eye_w // 2))
    eye_rects.append((right_eye_x, eye_y, right_eye_w, right_eye_w // 2))

# ========== 5. НАКЛАДЫВАЕМ ОЧКИ (ЕЩЕ ЛЕВЕЕ) ==========
glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))

# ИСПРАВЛЕНИЕ 1: Очки еще левее
glasses_x = eye_left_x - 28  # Было -25, теперь -28
glasses_y = eyes_center_y - glasses_height // 6

# Проверка границ
if glasses_x < 0: glasses_x = 0
if glasses_y < 0: glasses_y = 0
if glasses_x + glasses_width > img.shape[1]: glasses_width = img.shape[1] - glasses_x
if glasses_y + glasses_height > img.shape[0]: glasses_height = img.shape[0] - glasses_y

if glasses_width > 0 and glasses_height > 0:
    glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
    
    if glasses_resized.shape[2] == 4:
        alpha = glasses_resized[:, :, 3] / 255.0
        for c in range(3):
            img_processed[glasses_y:glasses_y + glasses_height, 
                         glasses_x:glasses_x + glasses_width, c] = \
                img_processed[glasses_y:glasses_y + glasses_height,
                             glasses_x:glasses_x + glasses_width, c] * (1 - alpha) + \
                glasses_resized[:, :, c] * alpha
    else:
        roi = img_processed[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width]
        glasses_gray = cv2.cvtColor(glasses_resized, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(glasses_gray, 250, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        
        roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
        roi_fg = cv2.bitwise_and(glasses_resized, glasses_resized, mask=mask)
        img_processed[glasses_y:glasses_y + glasses_height, glasses_x:glasses_x + glasses_width] = cv2.add(roi_bg, roi_fg)

# ========== 6. СОЗДАЕМ МАСКИ ДЛЯ ЗАЩИТЫ ГЛАЗ (ЗОНА УМЕНЬШЕНА ВДВОЕ) ==========
eyes_mask = np.zeros(img.shape[:2], dtype=np.uint8)

for i, (ex, ey, ew, eh) in enumerate(eye_rects):
    eye_center = (ex + ew // 2, ey + eh // 2)
    # ИСПРАВЛЕНИЕ 2: Уменьшаем зону защиты в два раза
    # Было: (ew // 2 + 8, eh // 2 + 6)
    # Стало: (ew // 4 + 4, eh // 4 + 3) - примерно в два раза меньше
    eye_axes = (max(ew // 4 + 4, 5), max(eh // 4 + 3, 4))  # Минимальные размеры для защиты
    
    cv2.ellipse(eyes_mask, eye_center, eye_axes, 0, 0, 360, 255, -1)
    
    print(f"Защита глаза {i+1}: центр={eye_center}, размер={eye_axes} (уменьшено вдвое)")

# Вычисляем расстояние между центрами защитных областей
if len(eye_rects) >= 2:
    center1 = (eye_rects[0][0] + eye_rects[0][2] // 2, eye_rects[0][1] + eye_rects[0][3] // 2)
    center2 = (eye_rects[1][0] + eye_rects[1][2] // 2, eye_rects[1][1] + eye_rects[1][3] // 2)
    distance = np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)
    print(f"Расстояние между центрами защитных областей: {distance:.1f} пикселей")

### раскомментировать, чтобы увидеть Стэтхема в очках.
###''' 
# ========== 7. РАЗМЫТИЕ ВСЕГО ЛИЦА, КРОМЕ УЗКИХ ОБЛАСТЕЙ ГЛАЗ ==========
face_mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.ellipse(face_mask, center, (axes[0], int(axes[1] * 1.1)), 0, 0, 360, 255, -1)

# Вычитаем уменьшенные защитные области глаз
blur_mask = cv2.subtract(face_mask, eyes_mask)

# Применяем размытие
blurred_face = cv2.GaussianBlur(img_processed, (35, 35), 0)

# Накладываем размытую область
img_processed = np.where(blur_mask[:, :, np.newaxis] == 255, blurred_face, img_processed)

# ========== 8. ДОРАБОТКА: ДОПОЛНИТЕЛЬНОЕ РАЗМЫТИЕ ОБЛАСТИ МЕЖДУ ГЛАЗАМИ ==========
if len(eye_rects) >= 2:
    left_eye = eye_rects[0]
    right_eye = eye_rects[1]
    
    between_eyes_x = left_eye[0] + left_eye[2]
    between_eyes_width = right_eye[0] - between_eyes_x
    between_eyes_y = min(left_eye[1], right_eye[1]) - 10
    between_eyes_height = max(left_eye[3], right_eye[3]) + 20
    
    nose_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.rectangle(nose_mask, 
                 (between_eyes_x, between_eyes_y),
                 (between_eyes_x + between_eyes_width, between_eyes_y + between_eyes_height),
                 255, -1)
    
    # Вычитаем УЖЕ УМЕНЬШЕННЫЕ защитные области
    nose_mask = cv2.subtract(nose_mask, eyes_mask)
    
    extra_blurred = cv2.GaussianBlur(img_processed, (45, 45), 0)
    img_processed = np.where(nose_mask[:, :, np.newaxis] == 255, extra_blurred, img_processed)
    
    print(f"Дополнительное размытие носа: x={between_eyes_x}, y={between_eyes_y}")
###''' 
### раскомментировать, чтобы увидеть Стэтхема в очках.

# ========== 9. СОХРАНЕНИЕ РЕЗУЛЬТАТА ==========
result_path = 'C:\\Users\\User\\Desktop\\pythonproject\\result_final_perfect.png'
cv2.imwrite(result_path, img_processed)

print(f"\nОбработка завершена! Результат сохранен как: {result_path}")
print(f"Очки: x={glasses_x} (сдвиг -28)")
print(f"Защитные области глаз уменьшены вдвое")

# Показываем только финальный результат
cv2.imshow('Идеальный результат', img_processed)
cv2.waitKey(0)
cv2.destroyAllWindows()