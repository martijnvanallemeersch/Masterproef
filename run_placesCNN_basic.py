# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)

import torch
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
mypath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

workbook = xlsxwriter.Workbook('C:\\Users\\marti\\Desktop\\ResultatenRouteYou\\stat.xlsx')
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
    perc = resultPlaces[0].percentage
    indexObjectNul = 0

    if(perc > 0.35):
        if (resultPlaces[0].data != None):
            return resultPlaces[0]
        else:
            perc = 0

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
                    if(indexObjectNul == 2):
                        return -1
                    break
        else:
            while True:
                indexObjectNul = indexObjectNul + 1
                objectNul = resultPlaces[indexObjectNul].data
                perc = resultPlaces[indexObjectNul].percentage
                if (indexObjectNul == 2):
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
            img = Image.open('C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + file)
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

                resultPlaces.append(resPlaces(POITypes.get(GeneralTypeID_dict.get(classes_split[0])),float('{:.3f}'.format(probs[i]))))

            resEvaluate = evaluate(resultPlaces)

            if(resEvaluate == -1):
                #geen goede match gevonden
                print('geen goed match gevonden')


            else:
                #goede match gevonden :)
                if(resEvaluate.data != None):
                    print('Categorie ' + resEvaluate.data.data)

                    newpath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + resEvaluate.data.data + '\\'
                    if not os.path.exists(newpath):
                        os.makedirs(newpath)

                    os.rename(
                        'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + file,
                        newpath + file)
                else:
                    print('Errorr')



        except():
            print("Error")


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
