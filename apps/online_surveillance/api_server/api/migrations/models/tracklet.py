from mongoengine import *


# Create your models here.

class Tracklet(Document):
    meta = {
        'collection': 'tracklets'
    }
    tracklet_id = SequenceField(required=True, primary_key=True)
