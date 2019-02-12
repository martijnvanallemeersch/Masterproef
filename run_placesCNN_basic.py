# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
import datetime
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
import os
import xlsxwriter
from PIL import Image
from anytree import Node, RenderTree
from collections import defaultdict
import  json


# importing csv module
import csv

# th architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

from os import listdir
from os.path import isfile, join
mypath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\BENLFR'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

hour = datetime.datetime.now().hour
minute = datetime.datetime.now().minute
second = datetime.datetime.now().second

date = datetime.datetime.now().date()

workbook = xlsxwriter.Workbook('C:\\Users\\marti\\Desktop\\ResultatenRouteYou\\stat' + str(date) +'_' + str(hour)+str(minute)+str(second)+'.xlsx')
worksheet = workbook.add_worksheet()
worksheet.write('A1','Type')
worksheet.write('B1','#')
worksheet.write('C1','Juist')
worksheet.write('D1','Fout')
worksheet.write('E1','False neg')
worksheet.write('F1','False pos')

juist = 0
nietGevonden = 0
falseNegTeller = 0
falsePosTeller = 0

class Resultaat:
  def __init__(self,type):
    self.type = type;
    self.aantal = 0
    self.juist = 0
    self.fout = 0
    self.falseNeg = 0
    self.falsePos = 0

GeneralTypeID_dict = {}
POITypes = dict()
Childs = dict()
Parent = dict()


RelatedWord = dict()

church = ["synagoge", "mausoleum","tower"]
castle = ["moat","canal","pond","palace","ruin","formal_garden"]
palace = ["moat","canal","pond","castle","ruin","formal_garden"]
bridge = ["canal","river","viaduct","aqueduct", "lock_chamber", "rope_bridge","moat","industrial_area"]
street = ["crosswalk","plaza","alley"]
crosswalk = ["street","plaza","alley"]
park = ["forest_road","picnic_area","forest_pad","field/wild", "japanese_garden","botanical","garden/yard","rainforest","path/forest_path","vegetable_garden"]
lawn = ["formal_garden","topiary_garden","botanical_garden","yard"]
building = ["crosswalk","parking_garage","synagoge","hangar","farm","manufactured_home","burrough","patio","porch","museum"]
house = ["oast_house"]
square = ["fountain"]
hotel_room = ["bedchamber","bedroom","youth_hostel"]
bedchamber = ["hotel_room","bedroom","youth_hostel"]
bedroom = ["bedchamber","hotel_room","youth_hostel"]
youth_hostel = ["bedchamber","bedroom","hotel_room"]
quest_room = ["bedchamber","bedroom","hotel_room","quest_room"]
restaurant = ["dining_room","dining_hall","banquet_hall","sushi_bar","pizzeria"]
orchard = ["field","path","garden"]

RelatedWord["castle"] = castle
RelatedWord["bridge"] = bridge
RelatedWord["street"] = street
RelatedWord["crosswalk"] = crosswalk
RelatedWord["park"] = park
RelatedWord["lawn"] = lawn
RelatedWord["building"] = building
RelatedWord["house"] = house
RelatedWord["square"] = square
RelatedWord["church"] = church
RelatedWord["hotel_room"] = hotel_room
RelatedWord["bedchamber"] = bedchamber
RelatedWord["bedroom"] = bedroom
RelatedWord["youth_hostel"] = youth_hostel
RelatedWord["quest room"] = youth_hostel
RelatedWord["restaurant"] = restaurant
RelatedWord["orchard"] = orchard
RelatedWord["palace"] = palace


class resPlaces(object):
    def __init__(self, data,percentage):
        self.data = data
        self.percentage = percentage

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []
        self.parent = []


    def add_child(self, obj):
        self.children.append(obj)

    def add_parent(self, obj):
        self.parent = obj

def addType(id, value):
    if id in POITypes:
        print('error')
    else:
        POITypes[id] = value

def addChilds(id, value):
    if id in Childs:
        print('error')
    else:
        Childs[id] = value

def addParent(id, value):
    Parent[id] = value

def GenerateTreePOI():

    f = open('poi_types.csv', encoding='UTF-8')

    data = []
    line_count = 0

    for line in f:

        data_line = line.rstrip().split(';')

        if line_count == 0:
            line_count += 1
        else:
                res = data_line[1]
                res = res.replace("\"\"", "\"")
                res = res[:-1]
                res = res[1:]

                jsonObject = json.loads(res)

                englishTranslation = ""
                for i in jsonObject:
                    if i.get('en'):
                        englishTranslation = i.get('en')

                addType(data_line[0],Node(englishTranslation.lower()))
                addChilds(data_line[0],data_line[2].lower())
                GeneralTypeID_dict[str(englishTranslation).lower()] = data_line[0]
                line_count += 1

    for childId in Childs:
        if(Childs.get(childId) != ""):
            test = json.loads(Childs.get(childId))

            for integ in test:
                POITypes.get(str(childId)).add_child(POITypes.get(str(integ)))
                POITypes.get(str(integ)).add_parent(POITypes.get(str(childId)))
    print("Tree build")
    return

def evaluate(resultPlaces):




    for i in range(0, 5):

        if(resultPlaces[i].percentage > 0.35):
            if (resultPlaces[i].data != None):
                return resultPlaces[i]

    perc = resultPlaces[0].percentage
    indexObjectNul = 0

    objectNul = resultPlaces[indexObjectNul].data
    #if(objectNul != None):
    while(perc < 0.35):
        if (objectNul != None):
            if(objectNul.data != "poi"):
                perc = resultPlaces[indexObjectNul].percentage
                for i in range(indexObjectNul+1, 5):
                    if(resultPlaces[i].data != None):
                        object = resultPlaces[i].data
                        while(object.data != "poi"):
                            if(objectNul == object):
                                perc = perc + resultPlaces[i].percentage
                                break
                            else :
                                object = object.parent
                if(perc < 0.35):
                    objectNul = objectNul.parent
            else:
                while True:
                    indexObjectNul = indexObjectNul + 1
                    objectNul = resultPlaces[indexObjectNul].data
                    perc = resultPlaces[indexObjectNul].percentage
                    if(indexObjectNul == 4):
                        return -1
                    break
        else:
            while True:
                indexObjectNul = indexObjectNul + 1
                objectNul = resultPlaces[indexObjectNul].data
                perc = resultPlaces[indexObjectNul].percentage
                if (indexObjectNul == 4):
                    return -1
                break
        # else:
        #     return -1
    return resPlaces(objectNul,perc)



resultaatList = []

GenerateTreePOI()

for file in onlyfiles:

    print('fileName -> {}', file)

    # load the test image

    # if not os.access(img_name, os.W_OK):
    #     img_url = 'http://places.csail.mit.edu/demo/' + img_name
    #     os.system('wget ' + img_url)

    if(file.find(".jpeg") != -1 or file.find(".jpg") != -1 or file.find(".png") != -1):

        try:
            img = Image.open(mypath + "\\" + file)
            input_img = V(centre_crop(img).unsqueeze(0))

            # forward pass
            logit = model.forward(input_img)
            h_x = F.softmax(logit, 1).data.squeeze()
            probs, idx = h_x.sort(0, True)

            print('{} prediction on {}'.format(arch, file))

            resultPlaces = []

            check = 0
            juistCheck = 0
            nietGevondenCheck = 0
            falseNegCheck = 0
            falsePosCheck = 0
            checkResult = 0
            klasse = []
            falseNeg = []
            falsePos = []


            result = ""

            # output the prediction
            for i in range(0, 5):
                print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))


            for i in range(0, 5):

                result = result + "_" + classes[idx[i]]

                #todo
                #Sommige classes uit de categorie_places365.txt bevatten categorien zoals church/outside
                #voorlopig neem ik gewoon het eerste deel daarvan/ in de toekomst nog wijzigen
                classes_split = classes[idx[i]].split("/")

                percentage = float('{:.3f}'.format(probs[i]))

                if(RelatedWord.get(classes_split[0]) != None):
                    for related in RelatedWord.get(classes_split[0]):
                        for index in range(0, 5):
                            if(i != index):
                                classes_split_intern = classes[idx[index]].split("/")

                                if(classes_split_intern[0] == related):
                                    percentage = percentage + float('{:.3f}'.format(probs[index]))


                # speciale gevallen
                if(classes_split[0] == "bedroom" or classes_split[0] == "hotel_room" or classes_split[0] == "youth_hostel" ):
                    classes_split[0] = "quest room"
                #############################################

                #slecht geprogrammeerd omdat guest room hier harcoded staat maar lukt gelijk niet anders :(
                if (str('quest room') == classes_split[0]):
                    resultPlaces.append(resPlaces(POITypes.get(GeneralTypeID_dict.get("guest room")),percentage))
                else:
                    resultPlaces.append(resPlaces(POITypes.get(GeneralTypeID_dict.get(classes_split[0])),percentage))

            resEvaluate = evaluate(resultPlaces)

            check = 0
            juistCheck = 0
            nietGevondenCheck = 0
            falseNegCheck = 0
            falsePosCheck = 0
            checkResult = 0
            klasse = []
            falseNeg = []
            falsePos = []

            if(resEvaluate == -1):
                #geen goede match gevonden
                print('geen goed match gevonden')


            else:
                #goede match gevonden :)
                if(resEvaluate.data != None):

                    if ((resEvaluate.data.data.find("cathedral") != -1)):
                        klasse.append("Kerk")
                        klasse.append("Kathedraal")
                        klasse.append("Kapel")
                        klasse.append("POI")
                        klasse.append("Erfgoed")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")
                        falsePos.append("Kasteel")
                        falsePos.append("Abdij")
                        falsePos.append("Belfort")



                        falsePos.append("Burcht")

                        check = 1
                    elif ((resEvaluate.data.data.find("church") != -1)):
                        klasse.append("Kerk")
                        klasse.append("Kathedraal")
                        klasse.append("Kapel")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")
                        falsePos.append("Burcht")
                        falsePos.append("Kasteel")
                        falsePos.append("Abdij")
                        falsePos.append("Belfort")



                        check = 1

                    elif ((resEvaluate.data.data.find("synagoge") != -1)):
                        klasse.append("Kerk")
                        klasse.append("Kathedraal")
                        klasse.append("Kapel")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")
                        falsePos.append("Burcht")
                        falsePos.append("Kasteel")

                        check = 1

                    elif (resEvaluate.data.data.find("bridge") != -1):
                        klasse.append("Brug")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")


                        check = 1
                    elif (resEvaluate.data.data.find("building") != -1):

                        klasse.append("Huis")
                        klasse.append("Gebouw")
                        klasse.append("Herenhuis")
                        klasse.append("Herenwoonst")
                        klasse.append("Stadhuis")
                        klasse.append("Kantoorgebouw")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Boerderij")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falseNeg.append("Kerk")
                        falseNeg.append("Kathedraal")
                        falseNeg.append("Kapel")
                        falseNeg.append("Abdij")
                        falseNeg.append("Burcht")
                        falseNeg.append("B&B")
                        falseNeg.append("Boekenwinkel")
                        falseNeg.append("Cafe")
                        falseNeg.append("Architecturale plaats")
                        falseNeg.append("Begijnhof")
                        falseNeg.append("Museum")
                        falseNeg.append("School")
                        falseNeg.append("Voetbalstadion")
                        falseNeg.append("Hotel")
                        falseNeg.append("Hostel")
                        falseNeg.append("Kasteel")
                        falseNeg.append("Brouwerij")
                        falseNeg.append("Hoeve")
                        falseNeg.append("Landgoed")
                        falseNeg.append("Klooster")
                        falseNeg.append("Belfort")
                        falseNeg.append("Hotel")


                        falsePos.append("Plein")


                    elif (resEvaluate.data.data.find("house") != -1):

                        klasse.append("Huis")
                        klasse.append("Gebouw")
                        klasse.append("Herenhuis")
                        klasse.append("Herenwoonst")
                        klasse.append("Stadhuis")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")
                        klasse.append("Hoeve")

                        falseNeg.append("Kerk")
                        falseNeg.append("Kathedraal")
                        falseNeg.append("Kapel")
                        falseNeg.append("Abdij")
                        falseNeg.append("Burcht")
                        falseNeg.append("B&B")
                        falseNeg.append("Boekenwinkel")
                        falseNeg.append("Cafe")
                        falseNeg.append("Architecturale plaats")
                        falseNeg.append("Begijnhof")
                        falseNeg.append("Museum")
                        falseNeg.append("Hotel")
                        falseNeg.append("Hostel")
                        falseNeg.append("Brouwerij")
                        falseNeg.append("Hotel")


                        falsePos.append("Kasteel")


                    elif (resEvaluate.data.data.find("stadium") != -1):

                        klasse.append("Voetbalstadion")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")

                        check = 1

                    elif (resEvaluate.data.data.find("windmill") != -1):

                        klasse.append("Windmolen")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")

                        falsePos.append("Belfort")
                        check = 1

                    elif (resEvaluate.data.data.find("castle") != -1):
                        klasse.append("Kasteel")
                        klasse.append("Burcht")
                        klasse.append("Vesting")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")
                        klasse.append("Landgoed")

                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")
                        falsePos.append("Historisch gebouw")

                        check = 1
                    elif (resEvaluate.data.data.find("palace")!= -1):
                        klasse.append("Kasteel")
                        klasse.append("Burcht")
                        klasse.append("Vesting")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")
                        klasse.append("Landgoed")

                        falsePos.append("Abdij")
                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")
                        check = 1

                    elif (resEvaluate.data.data.find("tower")!= -1):
                        klasse.append("Belfort")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falsePos.append("Huis")
                        falsePos.append("Gebouw")
                        falsePos.append("Herenhuis")

                        falseNeg.append("Kathedraal")
                        falseNeg.append("Kerk")

                        check = 1

                    elif (resEvaluate.data.data.find("canal")!= -1):
                        klasse.append("Kanaal")
                        klasse.append("Kolk")
                        klasse.append("Water")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")


                        falseNeg.append("Brug")
                        check = 1

                    elif (resEvaluate.data.data.find("moat/water") != -1):
                        klasse.append("Kanaal")
                        klasse.append("Kolk")
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")

                        falseNeg.append("Brug")
                        falseNeg.append("Burcht")
                        check = 1

                    elif (resEvaluate.data.data.find("cemetery") != -1):
                        klasse.append("Begraafplaats")
                        klasse.append("Militaire begraafplaats")
                        klasse.append("POI")
                        klasse.append("Erfgoed")

                        falsePos.append("Park")


                        check = 1

                    elif (resEvaluate.data.data.find("park") != -1):
                        klasse.append("Park")
                        klasse.append("POI")

                        falseNeg.append("Bos")
                        falseNeg.append("Begraafplaats")
                        falseNeg.append("Militaire begraafplaats")

                        check = 1

                    elif (resEvaluate.data.data.find("guest room") != -1):
                        klasse.append("Hotel")
                        klasse.append("Hostel")
                        klasse.append("POI")

                        check = 1

                    elif (resEvaluate.data.data.find("restaurant") != -1):
                        klasse.append("POI")

                        falseNeg.append("Hotel")
                        check = 1
                    elif (resEvaluate.data.data.find("structure") != -1):

                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falseNeg.append("Brug")
                        falseNeg.append("Burcht")
                        falseNeg.append("Huis")
                        falseNeg.append("Gebouw")
                        falseNeg.append("Herenhuis")
                        falseNeg.append("Historisch gebouw")
                        falseNeg.append("Kasteel")
                        falseNeg.append("Vesting")
                        falseNeg.append("Kerk")
                        falseNeg.append("Kathedraal")
                        falseNeg.append("Kasteel")
                        falseNeg.append("Brouwerij")
                        falseNeg.append("Windmolen")
                        falseNeg.append("Klooster")
                        falseNeg.append("Hotel")


                        falseNeg.append("Kapel")
                        falseNeg.append("Abdij")
                        falseNeg.append("Burcht")
                        falseNeg.append("B&B")
                        falseNeg.append("Boekenwinkel")
                        falseNeg.append("Cafe")
                        falseNeg.append("Architecturale plaats")
                        falseNeg.append("Belfort")
                        falseNeg.append("Militaire begraafplaats")
                        falseNeg.append("Boerderij")
                        falseNeg.append("Hotel")
                        falseNeg.append("Landgoed")


                        check = 1

                    elif (resEvaluate.data.data.find("courtyard") != -1):

                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Plein")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                    elif (resEvaluate.data.data.find("square") != -1):

                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")
                        klasse.append("Erfgoed")
                        klasse.append("Plein")
                        klasse.append("Historisch gebouw")
                        klasse.append("Historische plaats")
                        klasse.append("Historische dorp")
                        klasse.append("Historische stad")
                        klasse.append("Historische gebeurtenis")

                        falsePos.append("Hotel")

                    elif (resEvaluate.data.data.find("park") != -1):

                        klasse.append("Park")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Boom")
                        falseNeg.append("Bos")

                    elif (resEvaluate.data.data.find("orchard") != -1):

                        klasse.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")
                        falseNeg.append("Bos")

                    elif (resEvaluate.data.data.find("landscape element") != -1):

                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")
                        falseNeg.append("Bos")

                    elif (resEvaluate.data.data.find("forest") != -1):

                        klasse.append("Bos")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")

                    elif (resEvaluate.data.data.find("path") != -1):

                        falseNeg.append("Bos")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")

                    elif (resEvaluate.data.data.find("swamp") != -1):

                        falseNeg.append("Bos")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")

                    elif (resEvaluate.data.data.find("garden") != -1):

                        falseNeg.append("Bos")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")

                        falsePos.append("Hotel")

                    elif (resEvaluate.data.data.find("aquarium") != -1):

                        klasse.append("Aquarium")
                        falseNeg.append("Boomgaard")
                        falseNeg.append("Park")
                        falseNeg.append("Boom")

                    elif (resEvaluate.data.data.find("swimmingpool") != -1):

                        falsePos.append("Aquarium")
                        falsePos.append("Hotel")
                    else:
                        klasse.append("POI")
                        klasse.append("Foto-stopplaats")

                    for kl in klasse:
                        if (file.find(kl) != -1):
                            juist = juist + 1
                            juistCheck = 1
                            checkResult = 1
                            break;

                    for fp in falsePos:
                        if (file.find(fp) != -1):
                            falsePosTeller = falsePosTeller + 1
                            falsePosCheck = 1
                            checkResult = 1
                            break;

                    for fn in falseNeg:
                        if (file.find(fn) != -1):
                            falseNegTeller = falseNegTeller + 1
                            falseNegCheck = 1
                            checkResult = 1
                            break;

                    if (checkResult == 0):
                        nietGevonden = nietGevonden + 1
                        nietGevondenCheck = 1

                    print('Categorie ' + resEvaluate.data.data)

                    resultDir = ""

                    gesplit = file.split("_")
                    typeBestaatCheck = 0

                    for result in resultaatList:
                        if (result.type == gesplit[0]):
                            typeBestaatCheck = 1

                    if (typeBestaatCheck == 0):
                        resultaatList.append(Resultaat(gesplit[0]))

                    for result in resultaatList:
                        if (result.type == gesplit[0]):
                            if (juistCheck == 1):
                                result.juist = result.juist + 1
                            elif (falsePosCheck == 1):
                                result.falsePos = result.falsePos + 1
                            elif (falseNegCheck == 1):
                                result.falseNeg = result.falseNeg + 1
                            elif (nietGevondenCheck == 1):
                                result.fout = result.fout + 1

                            result.aantal = result.aantal + 1

                    if (falsePosCheck == 1):
                        resultDir = "falsePos"
                    if (falseNegCheck == 1):
                        resultDir = "falseNeg"
                    if (nietGevondenCheck == 1):
                        resultDir = "nietGevonden"

                        newpath = mypath + '\\' + resultDir + '\\' + resEvaluate.data.data + '\\'
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)

                        os.rename(
                            mypath + '\\' + file,
                            newpath + file)

                    if (juistCheck == 1):
                        resultDir = "juist"

                    # if (juistCheck != 1):
                    #     newpath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + resultDir + '\\'
                    #     if not os.path.exists(newpath):
                    #         os.makedirs(newpath)
                    #
                    #     os.rename(
                    #         'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + file,
                    #         newpath + file)

                    if (nietGevondenCheck != 1):
                        # newpath = mypath + '\\' + resultDir + '\\'
                        # if not os.path.exists(newpath):
                        #     os.makedirs(newpath)
                        #
                        # os.rename(
                        #     mypath + '\\' + file,
                        #     newpath + file)

                        newpath = mypath + '\\' + resultDir + '\\' + resEvaluate.data.data + '\\'
                        if not os.path.exists(newpath):
                            os.makedirs(newpath)

                        os.rename(
                            mypath + '\\' + file,
                            newpath + file)


                else:
                    print('Errorr')

        except:
            print("Error met file " + file)


        # if (gesplit[0] == "Abdij"):
        #     if(result.find("building_facade")):
        #         #false neg
        #
        # if (gesplit[0] == "Begijnhof):
        #     if (result.find("building_facade")):
        #         # false neg
        #
        # if (gesplit[0] == "Begijnhof):
        #     if (result.find("building_facade")):
        #         # false neg
        #
        # if (gesplit[0] == "Belfort):
        #     if(result.find("building_facade")):
        #
        #     else if (result.find("building_facade")):
        #         # false neg

index_excel = 2

try:
    for res in resultaatList:
        print('type : {%s} juist -> {%s} , falsePos -> {%s}, falseNeg -> {%s}, niet gevonden -> {%s}'%(res.type, res.juist,res.falsePos,res.falseNeg,res.fout))


        worksheet.write('A' + str(index_excel), res.type)
        worksheet.write('B' + str(index_excel), res.aantal)
        worksheet.write('C' + str(index_excel), res.juist)
        worksheet.write('D' + str(index_excel), res.fout)
        worksheet.write('E' + str(index_excel), res.falseNeg)
        worksheet.write('F' + str(index_excel), res.falsePos)

        index_excel = index_excel + 1

    # print('falsePos -> {}' ,falsePosTeller)
    # print('falseNeg -> {}' ,falseNegTeller)
    # print('juist -> {}' ,juist)
    # print('niet gevonden -> {}' ,nietGevonden)

    workbook.close()
except:
    print('problemen om te schrijven naar de exel file')
