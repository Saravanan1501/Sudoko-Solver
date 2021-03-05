from utlis import *
import sudukoSolver


########################################################################
pathImage = "Resources/1.jpg"
model = load_model('Resources/myModel.h5')


######img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = cv2.imread(pathImage)
img = cv2.resize(img,(450,450))
imgThreshold = preProcess(img)
imgBlank = np.zeros((450, 450, 3), np.uint8)
imgWarpColored = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rows = np.vsplit(imgWarpColored, 9)
boxes = []
for r in rows:
    cols = np.hsplit(r, 9)
    for box in cols:
        boxes.append(box)


print(len(boxes))
numbers = getPredection(boxes, model)   #Make sure the Boxes are in grayscale

numbers = np.asarray(numbers)
posArray = np.where(numbers>0 ,0,1)
print(posArray)        #Gives pos of all empty spaces



imgDetectedDigits = imgBlank.copy()
#rec = cv2.rectangle(img,(0,0),(40,40),(0,255,0),2)
for x in range(0,9):
    for y in range(0,9):
        if(numbers[x*9+y]!=0):
            imgDetectedDigits = cv2.putText(imgDetectedDigits,  str(numbers[x*9+y]), (5+50*y,35+50*x), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (50,200,50), 2, cv2.LINE_AA)

imgSolvedDigits= imgBlank.copy()

board = np.array_split(numbers,9)
print(board)
try:
    sudukoSolver.solve(board)
except:
    pass
print(board)
flatList = []
for sublist in board:
    for item in sublist:
        flatList.append(item)
solvedNumbers =flatList*posArray
#imgSolvedDigits= displayNumbers(imgSolvedDigits,solvedNumbers)

for x in range(0,9):
    for y in range(0,9):
        if(posArray[x*9+y]==1):
            imgSolvedDigits = cv2.putText(imgSolvedDigits,  str(numbers[x*9+y]), (5+50*y,35+50*x), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (200,0,200), 2, cv2.LINE_AA)


imgFinal = cv2.bitwise_or(imgSolvedDigits,imgDetectedDigits, mask = None)




imageArray = ([img,imgThreshold,imgWarpColored, imgDetectedDigits],
                  [imgSolvedDigits, imgBlank,imgFinal,imgBlank])
stackedImage = stackImages(imageArray, 1)
stackedImage = cv2.resize(stackedImage, None, fx=0.5, fy=0.5)
cv2.imshow('Stacked Images', stackedImage)



cv2.waitKey(0)
