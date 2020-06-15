import pymongo

if __name__ == '__main__':
    client = pymongo.MongoClient('mongodb://localhost:27017')
    db = client['motreid']
    db['events'].drop()
    db['frames'].drop()
    db['detections'].drop()
    print('All MOTReID data collections dropped')
