import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import matplotlib.pyplot as plt
import torchvision.datasets
import torchvision.transforms


donusum = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  #normalize etmek
])

batch_size = 64
mnist_veriseti = datasets.MNIST(root='./data',train=True,download=True,transform=donusum)
veri_yukle = DataLoader(mnist_veriseti,batch_size=64,shuffle=True) #shuffle veri setini karıştırarak overfitting'in
                                                                                #onune gecmeye calisir

class Uretici(nn.Module):
    def __init__(self,input_size,output_size):
        super(Uretici,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size,256),
            nn.ReLU(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,output_size),
            nn.Tanh()
        )

        def forward(self,x):
            return self.model(x)

class Tahminci(nn.Module):
    def __init__(self,input_size,output_size):
        super(Tahminci,self).__init__()
        self.model = nn.Sequential(
         nn.Linear(input_size,1024),
         nn.LeakyReLU(0.3),
         nn.Linear(1024,512),
         nn.LeakyReLU(0.3),
         nn.Linear(512,256),
         nn.LeakyReLU(0.3),
         nn.Linear(256,1), # en sonunda sadece 1 çıktı alacağız
         nn.Sigmoid() #cıktıyı 0 ve 1 arasında almak icin sigmoid fonksiyonuna veriyoruz
        )

    def forward(self,x):
            return self.model(x)


generator = Uretici(100,1)
discriminator = Tahminci(100,1)
ogrenme_oranı = 0.0002

hata = nn.BCELoss() # iki sınıflı problemler icin BCE , ya 0 gelecek ya 1
optimizer_g = optim.Adam(generator.parameters(),lr=ogrenme_oranı)
optimizer_d = optim.Adam(discriminator.parameters(),lr=ogrenme_oranı)

num_epochs = 10000

for epoch in range(num_epochs):

    gercek_resimler = next(iter(veri_yukle)) #gerçek verileri rastgele secme
    gercek_resimler = gercek_resimler[0].view(-1,784) #flatten yapıyoruz (duzlestirme)

    gercek_degerler = torch.ones(batch_size,1) #gercekler 1 , bu kısım hata hesaplamasında kullanılır
    sahte_degerler = torch.zeros(batch_size,1) #sahteler 0    bu kısım hata hesaplamasında kullanılır

    noise = torch.randn(batch_size,100) # 100 boyutlu rastgele görüntü

    #ayırıcı gradyanları sıfırlama
    discriminator.zero_grad()

    gercek_cıktılar = discriminator(gercek_resimler)

    sahte_resimler = Uretici(noise)
    sahte_cıktılar = discriminator(sahte_resimler.detach())

    ayırıcı_loss = hata(sahte_cıktılar,sahte_degerler) + hata(gercek_cıktılar,gercek_degerler)
    ayırıcı_loss.backward() #geriye yayılım (türev alma)
    optimizer_d.step() #ağı güncelle


    #üretici gradyanlar sıfırlama
    generator.zero_grad()
    sahte_cıktılar = discriminator(sahte_resimler)
    uretici_loss = hata(sahte_cıktılar,gercek_degerler)
    uretici_loss.backward() #geriye yayılım (türev alma),
    optimizer_g.step() #ağı guncelle

    if epoch == 1000 :
        print(f"Epoch : {epoch} ayırıcı_hata : {ayırıcı_loss.item()}  uretici hata : {uretici_loss.item()}")


with torch.no_grad():
    noise = torch.randn(64, 100).to  # 64 adet sahte görüntü için gürültü oluştur
    sahte_resimler = generator(noise).view(-1, 1, 28, 28)  # Görüntü formatına dönüştür

    # Görüntüleri çiz
    grid_img = torchvision.utils.make_grid(sahte_resimler, nrow=8, normalize=True)
    plt.imshow(grid_img.permute(1, 2, 0).cpu())  # Görüntüyü göster
    plt.axis('off')  # Eksenleri gizle
    plt.show()