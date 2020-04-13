from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
from os import listdir
from os.path import isfile, join
import sys

files = [f for f in listdir('.') if isfile(f)]

for f in files:
    print(f.title())
    drawing = svg2rlg(f.title().lower())
    name = f.title().split(".")[0].lower()
    print(name)
    renderPDF.drawToFile(drawing, name+".pdf")
#drawing = svg2rlg("file.svg")
#renderPDF.drawToFile(drawing, "file.pdf")