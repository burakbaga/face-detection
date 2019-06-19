
import cv2
import imageio

# cascadeler filtre serisi bunlar yüzü ve gözü tespit etmek içi ardı ardına uygulanacak
face_cascade = cv2.CascadeClassifier('haarcascade-frontalface-default.xml')

eye_cascade = cv2.CascadeClassifier('haarcascade-eye.xml')


def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # resmi gri tonuna çevirdik

    # bu işlem yüz tespiti yapar ve tupple döndürür bu tupple içinde x y kordinatı (dikdörtgenin sol üst köşesi
    # h(yukseklik) w (genislik) değerleri vardır. 1.3 olarak verdiğimiz değer scale dir ne kadar ölçekleneceği
    # 5 değeri ise komşu sayısı en az 5 pencere olursa oraya yüz deriz.

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:  # x y w h değerlerini kaç tane yüz var ise okadar alıyoruz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        gray_face=gray[y:y+h,x:x+w] # gözü yüzün üzerinde arıyoruz yüzün seçili olduğu alanı aldık
        color_face=frame[y:y+h,x:x+w] #yüzün olsuğu alana renkli olarak aldık

        eyes=eye_cascade.detectMultiScale(gray_face,1.1,3)  # gri yüzün üzerinde göz tespiti yaptık ve tupple dondurduk
        for (ex,ey,ew,eh) in eyes: # koordinaları aldık
            cv2.rectangle(color_face,(ex,ey),(ex+w,ey+h),(0,255,0),2) # renkkli yüzde gözlere kareler çizdik

    return frame


reader=imageio.get_reader('1.mp4') #videoyu okuduk
fps=reader.get_meta_data()['fps'] # okuduğumuz videonun fps değerini aldık
writer=imageio.get_writer('output.mp4',fps=fps) # yeni video oluşturmak için videonun adı ve kaç fps olucak

for i,frame in enumerate(reader): # reader ile aldığımız videodan tek tek frame burada i sayaç kaçıncı frami aldığımızı görmek için
    frame=detect(frame)#alıp bunların üzerine detect fonksiyonunu uyguladık
    writer.append_data(frame)#detect fonk uygulanmış frameleri tektek videomuza ekliyoruz
    print(i) # kaçıncı framdeyiz

writer.close() # videoyu yazmayı kapatıyoruz
