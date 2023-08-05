def isMention(userName:str,notification:str):
    return userName in notification.split()
