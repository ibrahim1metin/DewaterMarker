import os
from PIL import Image,ImageDraw
import numpy as np
import random as r
xborders=892+2*570
yborders=895+661*2
def randomTextGen(length=5):
    resultArray=[]
    while len(resultArray)<length:
        resultArray.append(r.choice(["a", "b", "c", "d", "e", "f","g", "h", "i", "j", "k","l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z","1", "2", "3", "4", "5","6", "7", "8", "9","0"]))
    return "".join(resultArray)
def processSingleFile(fileName):
    img=Image.open(fr"memes\{fileName}")
    if img.size[0]>xborders or img.size[1]>yborders:return
    resized=img.resize((704,704)).convert("RGB")
    return np.array(resized,dtype=np.float32)
def watermarkImage(fileName):
    text = randomTextGen()
    img = Image.open(fr"memes\{fileName}").convert("RGBA")
    img=img.resize((704,704))
    txt = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt)
    width, height = img.size
    watermark_count = 20
    # Adjust vertical spacing based on the number of watermarks
    y_spacing = height // (watermark_count + 1)
    y_positions = [y_spacing * (i + 1) for i in range(watermark_count)]
    for i in range(watermark_count):
        x = r.randint(0, width-50)
        y = y_positions[i]
        draw.text((x, y), text, fill=(255, 255, 255, 200))
    watermarked = Image.alpha_composite(img, txt)
    return watermarked.convert("RGB")
def processDirectory(dir):
    for file in os.listdir(dir):
        img=processSingleFile(file)
        if np.all(img==None):
            continue
        watermarked=watermarkImage(file)
        watermarked=np.array(watermarked,dtype=np.float32)
        yield (watermarked/255,img/255)
if __name__=="__main__":
    wtr=watermarkImage("2ugt2n.jpg")
    wtr.show()