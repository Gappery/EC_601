import tweepy
import os
from urllib import request


# add your own credentials here
consumer_key = ""
consumer_secret = ""
access_key = ""
access_secret = ""


def get_pics_urls(name_info):

    print("*************************************************************")
    print("*************************************************************")

    print("First step: dowanload 30 pictures from given account:" + name_info)

    print("*************************************************************")
    print("*************************************************************")



    #authorize twitter and initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #create the txt file to store the url of pictures
    name_info_raw = name_info + '_raw'
    try:
        os.makedirs('./' + name_info_raw)
    except Exception as e:
        print(e)
    else:
        print('Successfully create directory ' + name_info_raw)

    print("Then we would download up to 30 pictures for the given twitter account"  + name_info)

    alltweets = []

    #make initial request for most recent tweets (10 this time)
    new_tweets = api.user_timeline(screen_name = name_info, count = 10)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = new_tweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    print("connecting........................................................")
    while len(new_tweets) > 0:
        alltweets.extend(new_tweets)
        oldest = new_tweets[-1].id - 1
        new_tweets = api.user_timeline(screen_name = name_info, count = 10, max_id = oldest)

    print("First grabbing finished")

    # use a counter to name the pictures
    count = 1
    # for each tweet stored in alltweets
    for status in alltweets:
        # preset maximum pictures number:30
        if count == 31:
            break
        # if this tweet has media attribute
        if 'media' in status.entities:
            for media in status.entities['media']:
                # if the media type is photo
                if media['type'] == 'photo':
                    image_url = media['media_url']
                    # if the photo format is '.jpg'
                    if (image_url[-4:] == '.jpg'):
                        filename = 'pic_num_' + str(count)
                        filepath = './' + name_info_raw + '/' + filename + '.jpg'
                        # try to download the picture
                        try:
                            request.urlretrieve(image_url, filepath)
                        except Exception as e:
                            print(e)
                        else:
                            # successfully download, counter adds 1
                            count += 1
                            print("Downloading Process: " + str(int((count-1)*100/30)) + "%")

    print("Finishing grabbing, " + str(count-1) + " pictures totally.\nFirst step finished")
    print("*************************************************************")
    print("*************************************************************")