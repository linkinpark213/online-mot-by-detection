from mongoengine import *


# Create your models here.

class Feature(Document):
    meta = {
        'collection': 'features'
    }
    feature_id = SequenceField(required=True, primary_key=True)
    tracklet_id = IntField(required=True)
    time = FloatField(required=True)
    image = StringField(required=True)
    feature = ListField(required=True)
