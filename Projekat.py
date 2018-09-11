import math
import cv2
import numpy as np
from sklearn.datasets import fetch_mldata
from skimage.measure import regionprops
from skimage import color
from scipy import ndimage
from skimage.measure import label

brojac = -1
mnist = fetch_mldata('MNIST original', data_home='mldata')
noviMNIST = []

def sledeci():
    global brojac
    brojac += 1
    return brojac

def uDosegu(x1,x2,x3):
    red = []
    for obj in x3:
        p1,p2 = x2['center']
        p3,p4 = obj['center']
        prom = (p3 - p1, p4 - p2)
        re1, re2 = prom
        mk = math.sqrt(re1 * re1 + re2 * re2)

        if mk<x1:
            red.append(obj)

    return red

def tackaDoLinije(x1,x2,x3):
    p1,p2 = x2
    p3,p4 = x3
    vektorLinija = (p3 - p1, p4 - p2)
    p1,p2 = x2
    p3,p4 = x1
    vektorTacka = (p3 - p1, p4 - p2)
    p1,p2 = vektorLinija
    vektorDuzina = math.sqrt(p1*p1 + p2*p2)
    r1,r2 = vektorLinija
    koren = math.sqrt(r1*r1 + r2*r2)
    vektorJedinica = (r1/koren, r2/koren)
    p1,p2 = vektorTacka
    vektorTackaSkalirana = (p1*(1.0/vektorDuzina), p2*(1.0/vektorDuzina))
    l1,l2 = vektorJedinica
    v1,v2 = vektorTackaSkalirana

    rez = l1*v1 + l2*v2

    prom = 1

    if rez<0.0:
        rez = ispravna(0.0)
        prom = ispravna(-1)
    elif rez>1.0:
        rez = ispravna(1.0)
        prom = ispravna(-1)

    p1,p2 = vektorLinija
    najbliziLiniji = (p1*rez, p2*rez)
    p1,p2 = najbliziLiniji
    p3,p4 = vektorTacka
    smanji = (p3-p1, p4-p2)
    re1,re2 = smanji
    com = math.sqrt(re1*re1 + re2*re2)
    udaljenost  = com
    p1,p2 = najbliziLiniji
    p3,p4 = x2
    najbliziLiniji = (p1+p3, p2-p4)

    return (udaljenost, (int(najbliziLiniji[0]), int(najbliziLiniji[1])), prom)


def ispravna(lin):
    return lin

def nulaFunkcija():
    return 0

def Hougha(frejm, slika, parametar):
    m1 = nulaFunkcija()
    m2 = nulaFunkcija()
    m3 = nulaFunkcija()
    m4 = nulaFunkcija()

    coskovi = cv2.Canny(slika, 100, 120, apertureSize=3)
    linije = cv2.HoughLinesP(coskovi, 1, np.pi/180, 40,700, 10)

    for w1,w2,w3,w4 in linije[0]:
        m1 = w1
        m2 = w2
        m3 = w3
        m4 = w4

    for i in range(len(linije)):
        for q1,q2,q3,q4 in linije[i]:
            if q1<m1:
                m1 = q1
                m2 = q2

            if q3>m3:
                m3 = q3
                m4 = q4

    return m1,m2,m3,m4

def pronadjiLiniju(videoSnimak):
    cap = cv2.VideoCapture(videoSnimak)

    nn = nulaFunkcija()
    sivaBoja = "grayFrame"

    if nn==0:
        nn += 1
        while(cap.isOpened()):
            ret,frejm = cap.read()

            if ret:
                sivaBoja = cv2.cvtColor(frejm, cv2.COLOR_BGR2GRAY)
            if cv2.waitKey(1) & 0xff == ord('q'):
                break
            else:
                break

            cap.release()
            cv2.destroyAllWindows()
    return Hougha(frejm, sivaBoja, 2)

def prepoznavanjeRegiona(slika):
    imeSlike = label(slika)
    region = regionprops(imeSlike)
    regioni = []
    regions = nulaFunkcija()

    for reg in region:
        regions = {'reg_box':region.box,
                   'center':(round((reg.box[0] + reg.box[2])/2), round((reg.box[1] + reg.box[3])/2)),
                   'status':'r'}
        regioni.append(regions)

    aa = nulaFunkcija()
    duzina = len(regioni)

    if duzina>1:
        if duzina==2:
            if(region[0].area > region[1].area):
                return region[0].box[0], region[0].box[2], region[0].box[1], region[0].box[3]
            else:
                return region[1].box[0], region[1].box[2], region[1].box[1], region[1].box[3]
        else:
            for re in regioni:
                d = 0
                for r1 in regioni:
                    if(re['center'] != r1['center']):
                        d += pow((re['center'][0] - r1['center'][0]),2) + pow((re['center'][1] - r1['center'][1]),2)
                re['dist'] = d/(duzina - 1)
                aa += d/(duzina - 1)

            dd = 1.3*aa/duzina

            for r in regioni:
                if r['dist']>dd:
                    r['status'] = 'w'
                else:
                    r['status'] = 'r'

    m1 = ispravna(35)
    m2 = ispravna(35)
    m3 = ispravna(-1)
    m4 = ispravna(-1)

    linija = (m1 + m2 + m3 + m4)/2

    for r in regioni:
        if(r['status'] == 'r'):
            box = r['bbox']
            if box[0] < m1:
                m1 = ispravna(box[0])
            if box[1] < m2:
                m2 = ispravna(box[1])
            if box[2] > m3:
                m3 = ispravna(box[2])
            if box[3] > m4:
                m4 = ispravna(box[3])

    return m1, m3, m2, m4

def pronadjiCosak(slika, m, koeficijent):
    imeSlike = label(slika)
    region = regionprops(imeSlike)

    novaSlika = "novaSlika"

    m1 = ispravna(700)
    m2 = ispravna(700)
    m3 = ispravna(-1)
    m4 = ispravna(-1)

    prom = nulaFunkcija()

    if m==1:
        m1,m2,m3,m4 = prepoznavanjeRegiona(slika)
    else:
        for reg in region:
            bbox = reg.bbox
            if bbox[0]<m1:
                m1 = bbox[0]
            if bbox[1]<m2:
                m2 = bbox[1]
            if bbox[2]>m3:
                m3 = bbox[2]
            if bbox[3]>m4:
                m4 = bbox[3]

    ograniceno = (prom*2 + 2*28)*koeficijent
    visina = m3 - m1
    duzina = m4 - m2
    visina = visina + ograniceno

    novaSlika = np.zeros((28,28))
    novaSlika[28 - visina:28, 28 - duzina:28] = novaSlika[28 - visina:28, 28 - duzina:28] + slika[m1:m3, m2:m4]

    return novaSlika

def transformacijaMNIST(mnist, broj):
    prom = nulaFunkcija()
    while prom < broj:
        MNISTSlika = mnist.data[prom].reshape(28,28)
        sivaMNISTSlika = ((color.rgb2gray(MNISTSlika)/255.0) > 0.88).astype('uint8')
        novaMNISTSlika = pronadjiCosak(sivaMNISTSlika, 0, 0)
        noviMNIST.append(novaMNISTSlika)
        prom += 1
    return True

def eliminisiObojeneTacke(x, y, p1, p2):
    provera = False
    susedne = []

    if(x - p1 >= 0 & y - p1 >= 0):
        susedne.append((x - p1, y - p1))
        susedne.append((x, y - p1))
        susedne.append((x - p1, y))
        provera = True
    elif(x - p1 >= 0 & y + p1 < 28):
        susedne.append((x - p1, y + p1))
        provera = True
    elif(x + p1 < 28 & y - p1 >= 0):
        susedne.append((x + p1, y + p1))
        provera = True
    elif(x + p1 < 28 & y + p1 < 28):
        susedne.append((x + p1, y + p1))
        susedne.append((x, y + p1))
        susedne.append((x + p1, y))
        provera = True
    elif(x - p2 >= 0 & y - p2 >= 0):
        susedne.append((x - p2, y - p2))
        susedne.append((x, y - p2))
        susedne.append((x - p2, y))
        provera = True
    elif(x - p2 >= 0 & y + p2 < 28):
        susedne.append((x - p2, y + p2))
        provera = True
    elif(x + p2 < 28 & y - p2 >= 0):
        susedne.append((x + p2, y - p2))
        provera = True
    elif(x + p2 < 28 & y + p2 < 28):
        susedne.append((x + p2, y + p2))
        susedne.append((x, y + p2))
        susedne.append((x + p2, y))
        provera = True
    return susedne, provera

def pronadjiBrojNaSnimku(broj):
    i = nulaFunkcija()
    minimalna = 9999999
    r = 45
    while i < 70000:
        suma = nulaFunkcija()
        mnist_slika = noviMNIST[i]
        suma = np.sum(mnist_slika != broj)
        if minimalna > suma:
            minimalna = suma
        if suma < 16:
            return mnist.target[i], suma
        i += 1
    r = -1

    if(r == -1):
        i = nulaFunkcija()
        minimalna = 9999999
        pozitivna = -1
        while i < 70000:
            suma = nulaFunkcija()
            mnist_slika = noviMNIST[i]
            suma = np.sum(mnist_slika != broj)
            if minimalna > suma:
                minimalna = suma
                pozitivna = i
            i += 1

        if minimalna < 120:
            if pozitivna != -1:
                return mnist.target[pozitivna],minimalna

    return  -1,minimalna

def kojaJeBoja(slika, x, y):
    susedne, provera = eliminisiObojeneTacke(x, y, 1, 2)

    c1 = 0
    c2 = 0
    for x in range(0, len(susedne)):
        if(slika[susedne[x][0], susedne[x][1]] == 0):
            c1 += 1
        else:
            c2 += 1

    if c2 > 2:
        slika[x,y] = 1.0
        return 'BELA'
    else:
        slika[x,y] = 0.0
        return 'CRNA'

def promenaBoje(slika, p1, p2, p3, p4):
    sivaSlika = color.rgb2gray(slika)/255.0
    sivaSlika = (sivaSlika >= 0.88).astype('uint8')
    prom = False

    for x in range(0,28):
        for y in range(0,28):
            if slika[x,y,p1] > slika[x,y,p2] & slika[x,y,p3] > slika[x,y,p4]:
                ob = kojaJeBoja(sivaSlika,x,y)
                if ob == 'CRNA':
                    slika[x,y] = [0,0,0]
                    sivaSlika[x,y] = 0.0
                    prom = True
                else:
                    sivaSlika[x,y] = 1.0
                    slika[x,y] = [255,255,255]
                    prom = True

    return slika,prom

def kojiBroj(slika):
    slikaZaObradu = slika
    miss = 99999

    a = (5,5)
    b = (1,1)

    sivaSlika = (color.rgb2gray(slikaZaObradu) >= 0.88).astype('uint8')

    aa = np.ones(a, np.uint8)
    bb = np.ones(b, np.uint8)

    novaSlika = pronadjiCosak(sivaSlika, 0, 0)
    p, promasaj = pronadjiBrojNaSnimku(novaSlika)

    if p == -1:
        novaSlika = pronadjiCosak(sivaSlika, 1, 0)
        p,promasaj = pronadjiBrojNaSnimku(novaSlika)

    return p,promasaj

def obradaVideoSnimka(cap, linija, x1, x2, prome, ime, f, nazivVidea, izlazniFajl):
    daLiRadi = False
    broj = nulaFunkcija()
    jedinicni = np.ones((x1, x2), np.uint8)
    prvaBoja = [230,230,230]
    drugaBoja = [255,255,255]
    coskovi = [(prvaBoja, drugaBoja)]

    nizElemenata = []
    pomocni = nulaFunkcija()
    brojac = nulaFunkcija()
    sabirac = nulaFunkcija()

    while(1):
        ret, slika = cap.read()
        if not ret:
            break
        (donja, gornja) = coskovi[0]
        donja = np.array(donja, dtype="uint8")
        gornja = np.array(gornja, dtype="uint8")
        maska = cv2.inRange(slika, donja, gornja)

        slika1 = prome * maska

        slika1 = cv2.dilate(slika1, jedinicni)

        obelezeni, najblizi = ndimage.label(slika1)
        predmeti = ndimage.find_objects(obelezeni)

        for x in range(najblizi):
            lokalni = predmeti[x]
            (xc, yc) = ((lokalni[1].stop + lokalni[1].start)/2, (lokalni[0].stop + lokalni[0].start)/2)
            (dxc, dyc) = ((lokalni[1].stop - lokalni[1].start), (lokalni[0].stop - lokalni[0].start))

            if(dxc > 11 or dyc > 11):
                element = {'center':(xc,yc), 'size':(dxc,dyc), 't':pomocni}
                liste = uDosegu(20, element, nizElemenata)
                nn = len(liste)

                if nn == 0:
                    element['id'] = sledeci()
                    element['t'] = pomocni
                    element['pass'] = False
                    element['hist'] = [{'center':(xc, yc), 'size':(dxc, dyc), 'pomocni':pomocni}]
                    element['fut'] = []
                    nizElemenata.append(element)
                elif nn == 1:
                    liste[0]['center'] = element['center']
                    liste[0]['t'] = pomocni
                    liste[0]['hist'].append({'center':(xc,yc), 'size':(dxc, dyc), 'pomocni':pomocni})
                    liste[0]['fut'] = []

        for ele in nizElemenata:
            t2 = pomocni - ele['t']
            if(t2 < 3):
                distanca, tacka, r = tackaDoLinije(ele['center'], linija[0], linija[1])
                if r > 0:
                    c = (255,255,255)
                    if distanca < 9:
                        c = (0,255,160)
                        if ele['pass'] == False:
                            ele['pass'] = True
                            brojac += 1
                            (x,y) = ele['center']
                            (sx,sy) = ele['size']

                            cetrnaest = 14

                            x1 = x - cetrnaest
                            x2 = x + cetrnaest
                            y1 = y - cetrnaest
                            y2 = y + cetrnaest

                            br, nepoklapanja = kojiBroj(slika[y1:y2,x1:x2])

                            pr_br = br

                            f.write("Pronadjen je broj: " + str(br) + " \n")

                            if br != None:
                                if br != -1:
                                    sabirac += br

                ide = ele['id']
                for h in ele['hist']:
                    t3 = pomocni - h['pomocni']
                    if t3 < 100:
                        dsa = 1

                for f in ele['fut']:
                    t3 = f[0] - pomocni
                    if t3 < 100:
                        dsa = 2

        cv2.putText(slika, str(ime), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (90, 90, 250), 2)
        cv2.putText(slika, 'Suma: ' + str(sabirac), (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (90,90,250), 2)
        pomocni += 1

        cv2.imshow('frame', slika)
        k = cv2.waitKey(20) & 0xff

    cap.release()

    f.write("Suma Brojeva: " + str(sabirac) + " \n")
    izlazniFajl.write(nazivVidea + " " + str(sabirac) + " \n")
    print("Suma: " + str(sabirac))

def pokreni():
    sledeci = True
    videoSnimak = "video-0.avi"
    a,b,c,d = pronadjiLiniju(videoSnimak)
    linija = [(a,b),(c,d)]
    MNIST = transformacijaMNIST(mnist, 70000)

    if MNIST == True:
        f = open("out.txt", "w")
        izlazniFajl = open("res.txt", "w")
        f.write("Suma brojeva sa obradjenih video materijala: \n")
        izlazniFajl.write("RA 116-2013 Ksenija Cvejic\n")
        izlazniFajl.write("file\tsum\n")

        for sledeciVideoSnimak in range(0,10):
            if sledeci == True:
                videoSnimak = "video-" + format(sledeciVideoSnimak) + ".avi"
                nazivVideoSnimka = "video-" + format(sledeciVideoSnimak) + ".avi"
                cap = cv2.VideoCapture(videoSnimak)
                print("Video: " + format(sledeciVideoSnimak))
                f.write("Naziv video snimka: " + str(videoSnimak) + " \n")
                dalje = obradaVideoSnimka(cap, linija, 2, 2, 1.0, videoSnimak, f, nazivVideoSnimka, izlazniFajl)

        f.close()
        izlazniFajl.close()
        cv2.destroyAllWindows()

pokreni()