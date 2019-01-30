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

POITypes = dict()
Childs = dict()

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

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

                engelseVertaling = ""
                for i in jsonObject:
                    if i.get('en'):
                        engelseVertaling = i.get('en')

                addType(data_line[0],Node(engelseVertaling))
                addChilds(data_line[0],data_line[2])
                line_count += 1

    for childId in Childs:
        if(Childs.get(childId) != ""):
            test = json.loads(Childs.get(childId))

            for integ in test:
                POITypes.get(str(childId)).add_child(POITypes.get(str(integ)))

    print("Tree build")
    return


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

                if((classes[idx[i]] == ("cathedral"))):
                    klasse.append("Kerk")
                    klasse.append("Kathedraal")
                    klasse.append("Kapel")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")
                    falsePos.append("Burcht")

                    check = 1
                elif((classes[idx[i]] == ("church/indoor"))):
                    klasse.append("Kerk")
                    klasse.append("Kathedraal")
                    klasse.append("Kapel")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")
                    falsePos.append("Burcht")

                    check = 1
                elif ((classes[idx[i]] == ("church/outdoor"))):
                    klasse.append("Kerk")
                    klasse.append("Kathedraal")
                    klasse.append("Kapel")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")
                    falsePos.append("Burcht")

                    check = 1
                elif ((classes[idx[i]] == ("bridge"))):
                    klasse.append("Brug")
                    check = 1
                elif ((classes[idx[i]] == ("building_facade"))):

                    klasse.append("Huis")
                    klasse.append("Gebouw")
                    klasse.append("Herenhuis")
                    klasse.append("Stadhuis")

                    falseNeg.append("Kerk")
                    falseNeg.append("Kathedraal")
                    falseNeg.append("Kapel")
                    falseNeg.append("Abdij")
                    falseNeg.append("Burcht")

                    check = 1
                elif ((classes[idx[i]] == ("castle"))):
                    klasse.append("Kasteel")
                    klasse.append("Burcht")
                    klasse.append("Vesting")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")

                    check = 1
                elif ((classes[idx[i]] == ("palace"))):
                    klasse.append("Kasteel")
                    klasse.append("Burcht")
                    klasse.append("Vesting")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")
                    check = 1

                elif ((classes[idx[i]] == ("tower"))):
                    klasse.append("Belfort")

                    falsePos.append("Huis")
                    falsePos.append("Gebouw")
                    falsePos.append("Herenhuis")
                    falsePos.append("Historisch gebouw")

                    falseNeg.append("Kathedraal")
                    falseNeg.append("Kerk")

                    check = 1

                elif ((classes[idx[i]] == ("canal/urban"))):
                    klasse.append("Kanaal")
                    klasse.append("Kolk")

                    falseNeg.append("Brug")
                    check = 1

                elif ((classes[idx[i]] == ("canal/natural"))):
                    klasse.append("Kanaal")
                    klasse.append("Kolk")

                    falseNeg.append("Brug")
                    check = 1

                elif ((classes[idx[i]] == ("moat/water"))):
                    klasse.append("Kanaal")
                    klasse.append("Kolk")

                    falseNeg.append("Brug")
                    falseNeg.append("Burcht")
                    check = 1

                if(check == 1):
                    break



            for kl in klasse:
                if(file.find(kl) != -1):
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

            if(checkResult == 0):
                nietGevonden = nietGevonden + 1
                nietGevondenCheck = 1

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

            if(falsePosCheck == 1):
                resultDir = "falsePos"
            if (falseNegCheck == 1):
                resultDir = "falseNeg"
            if (nietGevondenCheck == 1):
                resultDir = "nietGevonden"

            if(juistCheck != 1):
                newpath = 'C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + resultDir + '\\'
                if not os.path.exists(newpath):
                    os.makedirs(newpath)

                os.rename('C:\\Users\\marti\\Documents\\Kuleuven\\Masterjaar\\Masterproef\\fotos-pieter\\fotos\\' + file, newpath + file)

        except() :
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
