import os
import requests
from base64 import b64encode
from PIL import Image,JpegImagePlugin,PngImagePlugin
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
def uploadImage(img:JpegImagePlugin.JpegImageFile)->str:
    buffer=BytesIO()
    #img.save(buffer,format="jpeg")
    img.save(buffer,format="jpeg")
    encodedImage=b64encode(buffer.getvalue())
    key=os.getenv("imgbb_api_key")
    url="https://api.imgbb.com/1/upload"
    data={
        "key":key,
        "image":encodedImage
    }
    resp=requests.post(url,data)
    status=resp.status_code
    if(status==200):
        json=resp.json()
        return json["data"]["url"]
    else:
        raise Exception(f"The image could not have been uploaded. Status code:{status} ")
def main():
    img=Image.open("memes/2ugt2n.jpg")
    uploadImage(img)
if __name__=="__main__":
    main()