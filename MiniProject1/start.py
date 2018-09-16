import sys
import twicture
import pic2video
import analyse_video

twicture.get_pics_urls(sys.argv[1])

for i in sys.argv:
    print(i)

analyse_video.analyse(sys.argv[1])

#pic2video.convert()



