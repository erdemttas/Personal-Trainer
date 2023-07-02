import cv2
import mediapipe as mp   #Gerekli kütüphaneler import edilir.
import numpy as np
import math


#Omuz - dirseğin ön kısımı - bilek, üç nokta arasındaki açıyı hesaplayan ve çizen fonksiiyon
def FindAngle(img, p1, p2, p3, lmList, draw = True):
    x1, y1 = lmList[p1][1:]   #Gelen listenin yapısı = id, cx, cy - buradan yalnız x ve y kordinatlarını alıyoruz.
    x2, y2 = lmList[p2][1:]   #buraya gelen bilgiler bilinçli olarak omuz, dirseğin ön kısımı ve bileğin kordinatlarıdır.
    x3, y3 = lmList[p3][1:]
    
    
    #math.atan2 fonksiyonu verilen iki nokta arasındaki açıyı radyan olarak hesaplar.
    #math.degrees fonksiyonu radyan değerini dereceye dönüştürür. eğer açı negatifse +360 eklenir ve pozitif değere dönüştürülür.
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))    
    if angle < 0:
        angle += 360
    if draw:          #seçilen 3 noktanın çizgilerini kırmızı yap ve sarı yuvarlakla işaretleme yap.
        cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 3)
        cv2.line(img, (x3, y3), (x2, y2), (0,0,255), 3)
        
        cv2.circle(img, (x1,y1), 10, (0,255,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (0,255,255), cv2.FILLED)
        cv2.circle(img, (x3,y3), 10, (0,255,255), cv2.FILLED)
         
        cv2.circle(img, (x1,y1), 15, (0,255,255))
        cv2.circle(img, (x2,y2), 15, (0,255,255))
        cv2.circle(img, (x3,y3), 15, (0,255,255))
        
        #açı değerinin ekrana yazılmasını sağlar.
        cv2.putText(img, str(int(angle)), (x2 - 40, y2 + 40), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,255), 2)   
    return angle
    
    

cap = cv2.VideoCapture("VID_20230702_114845(0).mp4")   #videoyu içeriye aktarıyoruz.  içeriye (0) yazarsak kamera açılır.

mpPose = mp.solutions.pose   #Pose Estimation(poz tespiti) modülü mpPose değişkenine atanır.
pose = mpPose.Pose()   #Pose sınıfından bir nesne oluşturulur. bu nesne poz tespiti yapmak için kullanılır.
mpDraw = mp.solutions.drawing_utils   #poz tespitinde eklemlerin ve eklem bağlatılarının çizilmesini sağlar.


dir = 0   #şınav sayısını tutacak değişkenler tanımlanır.
count = 0
while True:
    succes, img = cap.read()   #cap.read() fonsiyonu = birinci paremetre olarak görüntü geldimi diye kontrol yapar, True YADA False döner. İkinci parametrede ise resim geldiyse resimi döndürür. 
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   #OpenCV renkleri BGR olarak algılar, Gerçek hayatta bu durum tam tersidir bu yüzden renkleri RGB ye dönüştürüyoruz.    
    results = pose.process(imgRGB)   #process yöntemi, verilen görüntü üzerinde pose tespiti yapar.
   
    lmList = []   #Eklem noktalarını depolamak için boş liste oluşturuyoruz.
    if results.pose_landmarks:   #bu koşul ifadesi poz tespiti sonucunda eklem noktaları tespit edildi mi edilmedi mi onu kontrol eder. (yani görüntüde bir insan var mı yok mu)
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)   #bu satır tespit başarılıysa eklem noktalarını ve eklemler arasındaki bağlantıyı çizer.
        
        for id, lm in enumerate(results.pose_landmarks.landmark):   #enumerate fonksiyonu sayesinde her bir eklem noktasının id'sini ve kordinatlarını değişkenlere aktarıyoruz.
            h, w, _ = img.shape   #oynatılacak olan video karesinin yüksekliğini genişliğini ve renk kanalını tespit ediyoruz. renk kanalı şuan önemli olmadığı için boş bırakıldı.
            cx, cy = int(lm.x*w), int(lm.y*h)   #kordinatlar gelen görüntünün boyutları ile çarpılarak piksel cinsinden hesaplanır.
            lmList.append([id, cx, cy])   #en son kordinatlar liste içerisine aktarılır.
        #   print(lmList)
        
        if len(lmList) != 0:   #liste boş değilse aşşağıdaki işlemler yapılsın.
            #şınav
            angle = FindAngle(img, 11, 13, 15, lmList)   #yukarıda oluşturduğum Açı bulma fonksiyonuna görüntü, gerekli eklem noktaları ve listeyi gönderiyoruz. Bunun sonucunda bize açıyı hesaplıyor.
            per = np.interp(angle, (200, 260), (0,100))   #np.interp fonksiyonu sayesinde kullanılan açı değeri belirli bir aralıktan(200-260) başka bir aralığa ölçeklenir(0-100).
            #print(angle)
            
            if per == 100:          #Bu kısımda gerekli değerler sağlanıyorsa sayaçların arttırımı işlemi yapılır
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                if dir == 1:
                    count += 0.5
                    dir = 0     
        
        #bu kısımda ekrana yeşil bir dikdörtgen ve çekilen şınav sayısı yazdırılır.
        cv2.rectangle(img,(180,60),(560,230), (0,255,0), cv2.FILLED)
        cv2.putText(img, str(count), (200,200), cv2.FONT_HERSHEY_PLAIN, 10, (255,0,0), 10)
        
        
    
    cv2.imshow("Goruntu", img)   #görüntü ekranda açılır.
    cv2.waitKey(1)   #görüntünün her karesi 10 milisaniye gecikerek ekrana gelir.
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    