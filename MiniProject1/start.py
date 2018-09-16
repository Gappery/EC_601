import sys
import twicture
import pic2video
import analyse_video

twicture.get_pics_urls(sys.argv[1])

pic2video.convert()

analyse_video.analyse()

