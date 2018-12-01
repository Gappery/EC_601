import datetime
import tweepy
import os
import sys


from urllib import request


def get_pics_urls(name_info, keyword, session_id):
    consumer_key = "7GaWskLF46Xx7fUmKgMiKN97C"
    consumer_secret = "MybOuiFrhftY9OiJCQteueTK1Nwe7g34bnGHkyU3AFFMkJjwda"
    access_key = "1040700523287138304-YLlkHfmw9O4Ty4vtjzbVgw5ccvFdQt"
    access_secret = "Gy5tsayny2SIf4aMDzO2Dv4vKE7BNtmqFpUuFU88TC0rF"

    keyword = keyword.lower()

    print("First step: dowanload 30 pictures from given account:" + name_info)

    #authorize twitter and initialize tweepy
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #create the txt file to store the url of pictures
    store_path = "./" + name_info + "/" + keyword + "/" + session_id;
    try:
        os.makedirs(store_path)
    except Exception as e:
        print('Create file failed: directory already exist')
    else:
        print('Successfully create directory ' + store_path)

    # print("Then we would download up to 30 pictures for the given twitter account"  + name_info)

    alltweets = []

    #make initial request for most recent tweets (10 this time)
    try:
        new_tweets = api.user_timeline(screen_name = name_info, count = 10)
    except Exception as e:
        print('Connect to Twitter API failed')

    # chech if the account has no tweets(pictures)
    if len(new_tweets) == 0:
        print("No pictures for this twitter account, please try another account")
        sys.exit(0)

    #save most recent tweets
    alltweets.extend(new_tweets)

    #save the id of the oldest tweet less one
    oldest = new_tweets[-1].id - 1

    # keep grabbing tweets until there are no tweets left to grab
    print("Connected, start grabbing all tweets...............................")
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

        if status.text.find(keyword) != -1:
            print(status.text)
            # if this tweet has media attribute
            if 'media' in status.entities:
                for media in status.entities['media']:
                    # if the media type is photo
                    if media['type'] == 'photo':
                        image_url = media['media_url']
                        # if the photo format is '.jpg'
                        if (image_url[-4:] == '.jpg'):
                            filename = 'pic_num_' + str(count)
                            filepath = store_path+ '/' + filename + '.jpg'
                            # try to download the picture
                            try:
                                request.urlretrieve(image_url, filepath)
                            except Exception as e:
                                print(e)
                            else:
                                # successfully download, counter adds 1
                                count += 1
                                print("Downloading Process: " + str(int((count - 1) * 100 / 30)) + "%")

    print("All qualified pictures have downloaded, you can find them at " + store_path)
    return count-1


def twitter_mode():
    while(True):
        print("Please fill a username/nickname")
        user_name = input()
        print("Please input the twitter id you want to search for:")
        account_name = input()
        print("Please input the keyword you want to search for filtering the tweets")
        descriptor = input()
        session_id = int(datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        images_number = get_pics_urls(account_name, descriptor, session_id)

def database_mode():
    while(True):
        print("Please fill the statistics result you want to view:\n")
        print("1. Most popular descriptor\n2. Average pictures per feed\n")
        option = int(input())
        if option == 0:
            ok = 0




if __name__ == '__main__':
    print("Please choose your purpose\n1 for twitter search, 2 for statistics view")
    if (int)(input()) == 1:
        twitter_mode()
    elif (int)(input()) == 2:
        database_mode()
    else:
        print("invalid input")
        exit(0)




