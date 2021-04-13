import eel
import converter as cv

@eel.expose
def getPath():
    path = cv.openFilePicker()
    return path

@eel.expose
def convert(path):
    cv.convert(path)

eel.init('web')
eel.start('gui.html', mode='edge')


