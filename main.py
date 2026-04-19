import cv2
import numpy as np
import os
from ultralytics import YOLO

# Intel MKL / OpenMP çakışmasını önler. Windows'ta PyTorch + OpenCV birlikte kullanılınca
# "OMP: Error #15" hatası çıkabilir, bunu bastırır.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# YOLOv8'in en büyük modeli (x = extra large).Daha yavaş ama en doğru.
model = YOLO("yolov8x.pt")

# Video dosyasını açar.
kamera = cv2.VideoCapture("video.mp4")

# Ekrana yazı yazmak için kullanılacak font tipi.
font = cv2.FONT_HERSHEY_DUPLEX

# Polygon (alan)
region = np.array([(352, 86), (188, 188), (377, 240), (496, 112)])
region = region.reshape((-1, 1, 2))



# Her ID için son 10 frame'deki "içeride mi?" bilgisini tutar.
# Stabilite için: tek frame'e bakıp karar vermek yerine geçmişe bakılır.
history = {}

# AYAK ÇİZGİSİ OVERLAP
def foot_line_overlap(x1, x2, y2, region_poly, samples=20):
    inside_count = 0
    for i in range(samples):
        x = int(x1 + (x2 - x1) * (i / (samples - 1))) # Ayak çizgisi üzerinde eşit aralıklı x noktaları üretir.
        y = int(y2) # Y sabit: box'ın en alt noktası (ayak).

        if cv2.pointPolygonTest(region_poly, (x, y), False) >= 0: # Bu nokta poligon içinde mi?
            inside_count += 1 

    return inside_count / samples


while True:

    ret, frame = kamera.read() # ret: okuma başarılı mı? frame: o anki görüntü
    if not ret:
        break

    current_inside = set() # Bu frame'de bölge içinde olan ID'ler.
    results = model.track(frame, 
                          persist=True,  # ID'lerin frame'ler arası korunması (tracking)
                          verbose=False, # Konsola gereksiz çıktı basmasın
                          device=0)      # GPU (CUDA device 0) kullan


    # History temizleme: aktif olmayan ID'leri sil
    if results[0].boxes.id is not None:
        active_ids = set(int(i) for i in results[0].boxes.id)
    else:
        active_ids = set()

    for k in [k for k in history if k not in active_ids]:
        del history[k]

    # Poligonu çiz
    cv2.polylines(frame, [region], True, (255, 255, 0), 2)

    if results[0].boxes.id is not None: # Eğer hiç tracked ID yoksa (kimse yok ya da tracking başlamadıysa) atla.
        
        for i in range(len(results[0].boxes)): 

            x1, y1, x2, y2 = results[0].boxes.xyxy[i]
            cls = int(results[0].boxes.cls[i])
            ids = int(results[0].boxes.id[i])

            if cls != 0:
                continue

            # ID history (opsiyonel stabilite)
            if ids not in history:
                history[ids] = []

            # AYAK ÇİZGİSİ ORANI : Ayak çizgisinin kaçta kaçı bölge içinde?
            ratio = foot_line_overlap(x1, x2, y2, region)
            is_inside = ratio >= 0.45

            history[ids].append(is_inside) # Bu frame'in kararını geçmişe ekle.
            if len(history[ids]) > 10:
                history[ids].pop(0)

            # Son 10 frame'in en az 5'inde içerdeyse → kararlı "içeride" kabul et.
            # Bu sayede 1-2 frame'lik yanlış tespitler (titreme) görmezden gelinir.
            stable_inside = history[ids].count(True) >= 5

            if stable_inside:
                current_inside.add(ids)

                # Yeşil box çiz (içerideki kişi).
                cv2.rectangle(frame, 
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 255, 0), 2)

                # Sarı ayak çizgisini görselleştir.
                cv2.line(frame,
                         (int(x1), int(y2)),
                         (int(x2), int(y2)),
                         (0, 255, 255), 2)

                # Bbox merkezine nokta koy.
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                # Kişinin üstüne ID yaz.
                cv2.putText(frame, f"ID:{ids}",
                            (int(x1), int(y1) - 8),
                            font, 0.5, (0, 255, 0), 1)

            # Kırmızı box → dışarıdaki kişi.
            else:
                cv2.rectangle(frame,
                              (int(x1), int(y1)),
                              (int(x2), int(y2)),
                              (0, 0, 255), 1)

    # UI
    cv2.rectangle(frame, (0, 0), (230, 40), (50, 0, 150), -1) 
    cv2.putText(frame,
                f"In Zone: {len(current_inside)} people",
                (5, 25),font, 0.7,(255, 255, 255), 1)
    cv2.imshow("Zone Counter - Foot Line", frame)

    if cv2.waitKey(1) & 0xFF == 27: #esc ile çıkış
        break

kamera.release()
cv2.destroyAllWindows()