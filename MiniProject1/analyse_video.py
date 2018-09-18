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

    print("Second step: Analyse and label for pictures downloaded in first step")

    print("*************************************************************")
    print("*************************************************************")

    # set the font for labels printed on pictures
    font = ImageFont.truetype('times.ttf',30)
    name_info_out = name_info + '_final'
    name_info_raw = name_info + '_raw'

    # create a directory to store the processed pictures
    try:
        os.makedirs('./' + name_info_out)
    except Exception as e:
        print(e)

    # connect to vision api
    client = vision.ImageAnnotatorClient()
    print("Using cloud vision api")

    list_names = os.listdir('./' + name_info_raw)

    count = 1
    # for each pictures in 'raw' folder
    for name in list_names:
        path = './' + name_info_raw + '/' + name
        with io.open(path, 'rb') as image_file:
            # read the info of the picture
            content = image_file.read()

        # uploading the picture info to google vision api
        image = types.Image(content=content)

        # performs label detection on the image file
        response = client.label_detection(image=image)
        labels = response.label_annotations

        # sum the labels to one string
        label_string = "Labels for this Image: \n"
        for sub_label in labels:
            label_string += sub_label.description + "\n"

        # draw labels to the specific picture
        image_temp = Image.open(path)
        draw = ImageDraw.Draw(image_temp)
        draw.text((0,0), label_string, (0,0,255), font=font)

        # save the processed picture
        image_temp.save('./' + name_info_out + '/' + name)

        print("Totally " + str(count) + " pictures are labelled")
        count += 1

    print("Labeling process finished")


