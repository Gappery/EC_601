import tweepy
import urllib
import json

#Twitter API credentials
consumer_key = "7GaWskLF46Xx7fUmKgMiKN97C"
consumer_secret = "MybOuiFrhftY9OiJCQteueTK1Nwe7g34bnGHkyU3AFFMkJjwda"
access_key = "1040700523287138304-YLlkHfmw9O4Ty4vtjzbVgw5ccvFdQt"
access_secret = "Gy5tsayny2SIf4aMDzO2Dv4vKE7BNtmqFpUuFU88TC0rF"


def get_pics_urls(screen_name):

    #authorize twitter and initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)







def download_pics():
    urllib.request.urlretrieve("https://pbs.twimg.com/profile_images/980287241212710912/BN6FXg4w_normal.jpg", '/picfolder/233')

