import os
import io
import PIL.Image as Image
import PIL.ImageColor as ImageColor
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

from google.cloud import vision
from google.cloud.vision import types

def analyse(name_info):
    # Instantiates a client
    print("*************************************************************")
    print("*************************************************************")

    print("Second step: Analyse and label those 30 pictures")

    print("*************************************************************")
    print("*************************************************************")

    font = ImageFont.truetype('times.ttf',30)
    name_info_out = name_info + '_final'
    name_info_raw = name_info + '_raw'
    try:
        os.makedirs('./' + name_info_out)
    except Exception as e:
        print(e)

    client = vision.ImageAnnotatorClient()
    print("Using cloud vision api")

    list_names = os.listdir('./' + name_info_raw)

    for name in list_names:
        path = './' + name_info_raw + '/' + name
        with io.open(path, 'rb') as image_file:
            content = image_file.read()

        image = types.Image(content=content)

        # Performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations

        label_string = "Labels for this Image: \n"
        for sub_label in labels:
            label_string += sub_label.description + "\n"

        image_temp = Image.open(path)
        draw = ImageDraw.Draw(image_temp)
        draw.text((0,0), label_string, (0,0,255), font=font)

        image_temp.save('./' + name_info_out + '/' + name)

        print("Label process finished")




