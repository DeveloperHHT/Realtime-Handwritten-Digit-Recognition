import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import matplotlib.pyplot as plt


# Eğitim veri setini yükle
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Veri setini yeniden şekillendirme ve normalleştirme
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Modeli oluşturma
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Modeli derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
history = model.fit(train_images, train_labels, epochs=7, batch_size=64, validation_data=(test_images, test_labels))

# Eğitim ve Doğrulama Kayıp (Loss) Grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Eğitim ve Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim ve Doğrulama Doğruluk (Accuracy) Grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()

# Eğitim Kaybı Grafiği
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.title('Eğitim Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Doğrulama Kaybı Grafiği
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.title('Doğrulama Kaybı')
plt.xlabel('Epok')
plt.ylabel('Kayıp')
plt.legend()
plt.show()

# Eğitim Doğruluğu Grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.title('Eğitim Doğruluğu')
plt.xlabel('Epok')
plt.ylabel('Doğruluk')
plt.legend()
plt.show()


# Eğitilmiş modeli kaydetme
model.save('mnist_model.h5')


# Bu kısımda rakamları çizerek tahmin yapıyoruz


# Eğitilmiş modeli yükleme
model = keras.models.load_model('mnist_model.h5')


# Sayıyı tahmin etme işlemini gerçekleştiren fonksiyon
def predict_number(image):
    # Resmi yeniden boyutlandırma ve normalleştirme
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0

    # Modelin beklentisi olan şekle dönüştürme
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Sayıyı tahmin etme
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    return predicted_label

# Kullanıcı tarafından çizilen sayıyı tahmin etme
def recognize_number():
    drawing = False
    last_point = (0, 0)
    canvas = np.zeros((400, 400), dtype='uint8')

    # Çizim işlevleri için bir geri çağırma fonksiyonu
    def draw_callback(event, x, y, flags, param):
        nonlocal drawing, last_point, canvas

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            last_point = (x, y)

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                cv2.line(canvas, last_point, (x, y), (255, 255, 255), 30)
                last_point = (x, y)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False

    # Çizim alanını oluşturma
    cv2.namedWindow('Draw a Number')
    cv2.setMouseCallback('Draw a Number', draw_callback)

    while True:
        cv2.imshow('Draw a Number', canvas)
        key = cv2.waitKey(1) & 0xFF

        # Çizimi temizleme
        if key == ord('c'):
            canvas = np.zeros((400, 400), dtype='uint8')

        # Tahmin yapma
        elif key == ord('p'):
            predicted_label = predict_number(canvas)
            print("Tahmin: ", predicted_label)

        # Çıkış yapma
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

# Sayıyı tahmin etme işlemini başlatma
recognize_number()


# Bu kısımdan ,itibaren gerçek zamanlı tanımaya geçiyoruz

# Eğitilmiş modeli yükleme
model = keras.models.load_model('mnist_model.h5')

# Sayıyı tahmin etme fonksiyonu
def predict_number(image):
    # Resmi yeniden boyutlandırma ve normalleştirme
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32') / 255.0

    # Modelin beklentisi olan şekle dönüştürme
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Sayıyı tahmin etme
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction)

    return predicted_label

# Kamera yakalayıcısını başlatma
cap = cv2.VideoCapture(0)

while True:
    # Kamera görüntüsünü okuma
    ret, frame = cap.read()

    # Görüntüyü iyileştirme ve gri tonlamaya dönüştürme
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # ROI'leri bulma
    _, threshold = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    numbers = []  # Tanınan sayıları tutmak için liste

    for contour in contours:
        # Kontur alanını kontrol etme
        area = cv2.contourArea(contour)
        if 500 < area:  # Minimum ve maksimum alan değerlerini ihtiyaçlarınıza göre ayarlayabilirsiniz
            # ROI'yi işleme
            x, y, w, h = cv2.boundingRect(contour)
            roi_gray = gray[y:y + h, x:x + w]

            # Sayıyı tahmin etme
            predicted_label = predict_number(roi_gray)

            # Tanınan sayıyı listeye ekleme
            numbers.append((predicted_label, (x, y, w, h)))

    # Tanınan sayıları kare içine alma ve ekrana yazdırma
    for number, (x, y, w, h) in numbers:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(number), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Görüntüyü ekrana çizme
    cv2.imshow('Kamera', frame)

    # ESC tuşuna basılınca döngüden çıkma
    if cv2.waitKey(1) == 27:
        break

# Kamera yakalayıcısını ve pencereleri kapatma
cap.release()
