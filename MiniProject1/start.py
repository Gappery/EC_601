import sys
import twicture
import pic2video
import analyse_video

# first step
twicture.get_pics_urls(sys.argv[1])

# second step
analyse_video.analyse(sys.argv[1])

# final step
pic2video.convert2mp4(sys.argv[1])



