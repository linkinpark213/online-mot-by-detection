from mongoengine import *


# Create your models here.

class Detection(Document):
    meta = {
        'collection': 'detections',
        'indexes': [
            'time'
        ]
    }
    detection_id = SequenceField(required=True, primary_key=True)
    time = FloatField(required=True)
    tracklet_id = IntField(required=True)
    box = DictField(required=True)

    @queryset_manager
    def fetch_latest_detections(doc_cls, queryset: QuerySet):
        last_time = queryset.order_by('-id').first()['time']
        return queryset.order_by('-detection_id').get(time=last_time)
