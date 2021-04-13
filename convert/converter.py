from PIL import Image
import os

#path = r'C:\Users\david.han\Desktop\COPY_images_files'

def convert(path):
    os.chdir(path)
    for file in os.listdir(path):
        if file.endswith(".png"):
            img = Image.open(file)
            file_name, file_ext = os.path.splitext(file)
            img.save('/jpg/{}.jpg'.format(file_name))

import wx

def openFilePicker():
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.DirDialog(None, 'Open', style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path