import tweepy
import urllib
import os
from urllib import request


#Twitter API credentials
consumer_key = "7GaWskLF46Xx7fUmKgMiKN97C"
consumer_secret = "MybOuiFrhftY9OiJCQteueTK1Nwe7g34bnGHkyU3AFFMkJjwda"
access_key = "1040700523287138304-YLlkHfmw9O4Ty4vtjzbVgw5ccvFdQt"
access_secret = "Gy5tsayny2SIf4aMDzO2Dv4vKE7BNtmqFpUuFU88TC0rF"


def get_pics_urls(name_info):

    #authorize twitter and initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #create the txt file to store the url of pictures
    os.makedirs('./' + name_info)
    f = open('./' + name_info + '/' + name_info + '.txt','w')
    alltweets = []

    #make initial request for most recent tweets (10 this time)
    new_tweets = api.user_timeline(screen_name = name_info, count = 10)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = new_tweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        alltweets.extend(new_tweets)
        oldest = new_tweets[-1].id - 1
        new_tweets = api.user_timeline(screen_name = name_info, count = 10, max_id = oldest)

    count = 0
    for status in alltweets:
        if 'media' in status.entities:
            for media in status.entities['media']:
                if media['type'] == 'photo':
                    image_url = media['media_url']
                    if (image_url[-4:] == '.jpg'):
                        count += 1
                        filename = str(count)
                        filepath = './' + name_info + '/' + filename + '.jpg'
                        try:
                            request.urlretrieve(image_url, filepath)
                        except(NameError, KeyError):
                            pass
