# Deep Learning Project
---

## Project Goal

- Projedeki hedef resimler üzerinde object detection algoritması kullanılarak insanları tespit ettikten sonra, her tespit üzerinde insanı segmente edebilecek tek sınıflı bir semantic segmentasyon modeli geliştirmektir.

- Bonus olarak kullandığınız modelleri TensorRT kütüphanesi ile optimize edebilmek projeyi daha verimli hale getirecektir. 
---
## Project Steps

- Modelleri oluşturmak aşağıda linki paylaşılan datasetten faydalanabilirsiniz.
- Programa uygun olması için Deep Learning kütüphanesi olarak PyTorch kullanmanızı bekliyoruz.
- Object detection algoritması için algoritma seçiminde özgürsünüz.(darknet,faster-rcnn vb.)
- Segmentasyon modeli oluşturmak ve eğitmek için beklentimiz eğitimde yaptığımız gibi model mimarisi, dataloader ve eğitim kodunu PyTorch ile  kendiniz oluşturmanız.
- Kodlarınızı pazartesi gününe kadar bizim de görebileceğimiz bir yerde mümkünse kendi github hesabınızda bir repoda oluşturmanızı bekliyoruz.
- Kendi bilgisayarında cuda kurulumunda zorlanan arkadaşlar kullanımı kolay olan google colab'den faydalanabilir, bu durum göz önüne alınarak drive'a yükleyebileceğiniz düşük boyutlu bir dataset tercih edildi.
- Pazartesi günü yazılımın oluşturabildiğiniz kısmını beraber inceleyip, sizin de katılımınızla interaktif bir ders olmasını amaçlıyoruz.
- Pazartesi gününe projenin tamamının yetişmesinden ziyade eksik de olsa koda hakim olduğunuz bir yazılımın ortaya çıkması temel beklentimiz olacak.


Dataset: https://www.cis.upenn.edu/~jshi/ped_html/

---

## Project Tricks:
- Linkini paylaşmış olduğumuz dataset içerisinde segmente edilmiş maskeler ve custom object detection eğitimi için bbox koordinatları içeren annotation dosyaları yer almakta, bu dosyalardan faydalanabilirsiniz.
- Object detection içerisinden tespit edilen insanları kesip segmentasyon modeline input olarak vereceğimiz için, insanları ve maskeleri detection modelinin çıktısı boyutunda kesip segmentasyon modelini eğitmek başarılı olma ihtimalini arttırır. 
- Custom bir object detection modeli eğitmekte zorlanan arkadaşlar insan sınıfı içeren hazır eğitilmiş bir model kullanabilir.(örnek olarak 10. haftada darknet'de kullandığımız pretrained modelini verebiliriz.)
- Yazılımı oluşturmak için Segmentasyon ve TensorRT derslerinde gördüğümüz algoritmalardan faydalanmanız işinizi kolaylaştıracaktır. 


Sorularınız ve takıldığınız kısımlar için issue kısmını kullanabilirsiniz. Projeden keyif almanız dileğiyle, iyi çalışmalar.
