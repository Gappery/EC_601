import pymysql
import pymongo


def sql_initiate(host, user, password):
    try:
        conn = pymysql.connect(host=host, user=user, password=password, database="twitter")
    except:
        conn = pymysql.connect(host="localhost", user="Gappery", password="De94blnei")
        try:
            conn.cursor().execute("CREATE DATABASE twitter")
            conn.connect(database="twitter")
        except:
            print("Connection Error")
            return None;
        else:
            print("finished")
    return conn


def sql_create_table(conn):
    conn.cursor().execute("create table if not exists twitter(user_name TEXT,\
                                                              session_id TEXT,\
                                                              account_info TEXT,\
                                                              image_num int,\
                                                              descriptor TEXT)")


def sql_insert(user_info, session_id, account_info, image_num, descriptor, conn):
    #sql = "INSERT INTO twitter.twitter VALUES(" +\
    #    user_info + "," + session_id + "," + account_info + "," +\
    #      image_num + "," + descriptor + ")"
    sql = "INSERT INTO twitter.twitter values ('" + user_info + "', '" + str(session_id) +\
        "', '" + account_info + "', " + str(image_num) + ", '" + descriptor + "')"
    conn.cursor().execute(sql)
    conn.commit()


def sql_search_image_num(conn):
    sql = "Select * from twitter"
    current_cursor = conn.cursor()
    current_cursor.execute(sql)
    results = current_cursor.fetchall()
    total_num = 0
    total_feeds = len(results)
    for each_result in results:
        total_num += each_result[3]
    print("average img num per feed is " + str(total_num/total_feeds))


def sql_search_popular_descriptor(conn):
    sql = "Select * from twitter"
    current_cursor = conn.cursor()
    current_cursor.execute(sql)
    results = current_cursor.fetchall()
    result_dict = {}
    for result in results:
        if result[4] in result_dict:
            result_dict[result[4]] += 1
        else:
            result_dict[result[4]] = 1
    sorted(result_dict.items(), key=lambda x: x[1], reverse=True)
    for key in result_dict.keys():
        print("\n\nThe most popular descriptor is " + key + " with searching times " + str(result_dict[key]))





def mongo_initiate():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = client['twitter']
    myCollection = mydb['twitter']
    return myCollection


def mongo_insert(user_info, session_id, account_info, image_num, descriptor, collection):
    element = {
        'user_info':user_info,
        'session_id':session_id,
        'account_info':account_info,
        'image_num':image_num,
        'descriptor':descriptor
    }
    result = collection.insert_one(element)


def mongo_average_img(collection):
    results = collection.find()
    result_num = results.count()
    total_value = 0
    for result in results:
        total_value += int(result['image_num'])
    print("average images per feed:" + str(total_value/result_num))

def mongo_popular_descriptor(collection):
    result_dict = {}
    results = collection.find()
    for result in results:
        if result['descriptor'] in result_dict:
            result_dict[result['descriptor']] += 1
        else:
            result_dict[result['descriptor']] = 1
    sorted(result_dict.items(), key=lambda x:x[1], reverse=True)
    for key in result_dict.keys():
        print("The most popular descriptor is " + key + " with searching times" + str(result_dict[key]))

def mongo_delete_all():
    client = pymongo.MongoClient("mongodb://localhost:27017")
    mydb = client['twitter']
    myCollection = mydb['twitter']
    myCollection.delete_many({})


if __name__ == '__main__':
    # conn = sql_initiate("localhost", "Gappery", "De94blnei")
    # sql_create_table(conn)
    # sql_insert("a", "123", "b", "23", "sd", conn)

    myCollection = mongo_initiate()
    #mongo_insert("a", "123", "b", "23", "vv", myCollection)
    #result_dict = {}
    mongo_delete_all()
    print(myCollection.find().count())








