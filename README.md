# AimHero - AimBot
# Using Tensorflow Object Detection API
# Tested on AimHero v5

Inspired from Sentdex's Work on his "Python Plays GTA V" series
Thanks to, Tensorflow Object Detection API

## Instructions

1) İmages / train / içine model dosyalarınızın resimlerini at.
2) Label_Img programı ile resimleri kıraparak xml ye çevir.
3) XML_to_CSV klasorunden xml_to_csv yi çalıştırarak xml dosyasını csv ye çevir. Bu, \ object_detection \ images klasöründe bir train_labels.csv ve test_labels.csv dosyası oluşturur.
4) Ardından, generate_tfrecord.py dosyasını bir metin düzenleyicide açın. 31. satırdan başlayarak etiket haritasını kendi etiket haritanızla değiştirin, burada her nesneye bir kimlik numarası atanır. Adım 5b'deki labelmap.pbtxt dosyasını yapılandırırken aynı numara ataması kullanılacaktır.

5) Ardından, bu komutları \ object_detection klasöründen yayınlayarak TFRecord dosyalarını oluşturun:
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record

6)Bunlar \ object_detection içinde bir train.record ve bir test.record dosyası oluşturur. Bunlar yeni nesne algılama sınıflandırıcısını eğitmek için kullanılacaktır.


7) Şimdi 5. Etiket Haritası Oluşturun ve Eğitimi Yapılandırın Eğitimden önce yapılacak en son şey bir etiket haritası oluşturmak ve eğitim konfigürasyon dosyasını düzenlemek gerekir.
8) 5a. Etiket haritası Etiket haritası, eğiticiye, sınıf adlarının sınıf kimlik numaralarına eşlenmesini tanımlayarak her nesnenin ne olduğunu söyler. Yeni bir dosya oluşturmak için bir metin düzenleyici kullanın ve C: \ tensorflow1 \ models \ research \ object_detection \ training labelmap.pbtxt olarak kaydedin. (Dosya türünün .pbtxt, .txt değil olduğundan emin olun!) Metin düzenleyicide, aşağıdaki şekilde etiket haritasını kopyalayın veya yazın (aşağıdaki örnek, Pinochle Deck Card Detector'umun etiket haritasıdır):

9) C: \ tensorflow1 \ models \ research \ object_detection \ samples \ config dosyasına gidin ve faster_rcnn_inception_v2_pets.config dosyasını \ object_detection \ training dizinine kopyalayın. Ardından dosyayı bir metin düzenleyicisiyle açın. .config dosyasında, çoğunlukla sınıfların ve örneklerin sayısını değiştirerek ve eğitim yollarına dosya yollarını ekleyerek yapmak için birkaç değişiklik vardır.
NOT: Daha hızlı_rcnn_inception_v2_pets.config dosyasında aşağıdaki değişiklikleri yapın. Not: Yollar tek eğik çizgi ile girilmelidir (ters eğik çizgi), ya da TensorFlow modeli eğitmeye çalışırken bir dosya yolu hatası verecektir! Ayrıca, yollar tek tırnak işaretleri (') değil, çift tırnak işareti (") olmalıdır.

10) Satır 9. Sayısallaştırıcının sınıflandırıcının algılamasını istediğiniz farklı nesne sayısına değiştirin. Yukarıdaki basketbol, gömlek ve ayakkabı detektörü için num_classes olacaktır: 3

11) Line 110. fine_tune_checkpoint değerini değiştir.

    fine_tune_checkpoint : "C:/tensorflow1/models/research/object_detection/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt"


12) Line 126 ve 128 i değiştir. 

13) Line 132 image\test içindeki görüntü sayısı ile değiştir.

14) Line 140 ve 1421 biçimlendir.

--- NOT:
	 Şimdi bu kodu girerek  eğitimi Başlat.
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_pets.config

--- NOT:
 	 Tensorboard üzerinden eğitimi görebilirsin.
python -m tensorboard.main --logdir=training





15) Eğitim tamamlandı. Sırada dondurulmuş çıkarım grafiği oluşturmak var. (.pb dosyası)  Object_detection klasöründen, şu komutu veriniz: “model.ckpt-XXXX” içindeki “XXXX”, eğitim klasöründeki en yüksek numaralı .ckpt dosyasıyla değiştirilmelidir: XXXX En büyük step no.
Bu, \ object_detection \ inference_graph klasöründe bir frozen_inference_graph.pb dosyası oluşturur. .pb dosyası nesne algılama sınıflandırıcısını içerir.

python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_pets.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph




16) Dosyalar oluşturuldu. Şimdi bu dosyaların kullanımına geldik. Python komut dosyalarını çalıştırmadan önce, betikteki NUM_CLASSES değişkenini, tespit etmek istediğiniz sınıf sayısına eşit olacak şekilde değiştirmeniz gerekir. (Pinochle Kart Dedektörüm için, tespit etmek istediğim altı kart var, bu yüzden NUM_CLASSES = 6)

- Nesne algılayıcınızı test etmek için nesnenin veya nesnelerin bir resmini \ object_detection klasörüne taşıyın ve resmin dosya adıyla eşleşmesi için Object_detection_image.py dosyasındaki IMAGE_NAME değişkenini değiştirin. Alternatif olarak, nesnelerin bir videosunu (Object_detection_video.py kullanarak) kullanabilir veya sadece bir USB web kamerasını takabilir ve nesneye (Object_detection_webcam.py kullanarak) yönlendirebilirsiniz.



17) windowtarget-v1.py dosyasını çalıştır.
    ```
