import tensorflow as tf
import os
import praw
from imgBB import uploadImage
from utils import isMention
from io import BytesIO
import requests
from PIL import Image
from dotenv import load_dotenv
import time
from typing import List
import numpy as np
from processData import watermarkImage
load_dotenv()
class Agent:
    def __init__(self) -> None:
        self.client=praw.Reddit(
                    client_id=os.getenv("client_id"),
                    client_secret=os.getenv("client_secret"),
                    username=os.getenv("UNAME"),
                    password=os.getenv("password"),
                    user_agent="windows:WaterMarkRemover:1.0 (by u/imetindonmez)"
                    )
        self.answeredPosts=[]
        self.model:tf.keras.Model
        self.model=tf.keras.models.load_model("saved/model")
        self.model.summary()
    def getUnreadMentions(self):
        for notification in self.client.inbox.unread(limit=None):
            if isMention("u/"+os.getenv("UNAME"),notification.body):
                yield notification
    def getMedia(self,cmt:praw.reddit.Comment):
        post:praw.reddit.Submission=cmt.submission
        if post.id in self.answeredPosts:
            post.reply("This has already been answered.")
            return
        try:
            if post.preview and 'images' in post.preview:
                        image_url = post.preview['images'][0]['source']['url']
                        response = requests.get(image_url)
                        if response.ok:
                            img = Image.open(BytesIO(response.content))
                            self.answeredPosts.append(post.id)
                            return img
                        else:
                            cmt.reply(f"Failed to download the image from {image_url}")
        except AttributeError:
            cmt.reply("The post does not have an image media")
    def processMedia(self,img):
        img=img.convert("RGB")
        size=img.size
        imgr=img.resize((704,704))
        arr=np.array(imgr,dtype=np.float32)/255
        imgconst=tf.constant([arr],shape=(1,704,704,3))
        result=self.model.predict(imgconst,batch_size=1)
        result=tf.multiply(result,255)
        resultnp=np.array(result)
        resultnp=np.squeeze(resultnp)
        newIMG=Image.fromarray(resultnp.astype(np.uint8))
        newIMG=newIMG.resize(size)
        return newIMG
    def answerCMT(self,cmt:praw.reddit.Comment):
        media=self.getMedia(cmt)
        if media==None:return
        resultImage=self.processMedia(media)
        uploadLink=uploadImage(resultImage)
        cmt.reply(f"The download link: {uploadLink}")
####################################################################################################################################

agent=Agent()
while (True):
    mentions:List[praw.reddit.Comment]=list(agent.getUnreadMentions())
    for mention in mentions:
          agent.answerCMT(mention)
          mention.mark_read()
    time.sleep(60)

"""img=watermarkImage("2ugt2n.jpg")
agent.processMedia(img).show()"""
"""img=watermarkImage("2ugt2n.jpg")
img.save("watermarked/atermarkedIMG1.jpg")"""