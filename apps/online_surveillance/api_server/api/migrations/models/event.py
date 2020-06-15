from mongoengine import *


# Create your models here.

class Event(Document):
    meta = {
        'collection': 'events',
        'indexes': [
            'time'
        ]
    }
    event_id = SequenceField(required=True, primary_key=True)
    time = FloatField(required=True)
    event = StringField(required=True)

    @queryset_manager
    def fetch_latest_event(doc_cls, queryset: QuerySet):
        results = doc_cls.objects.order_by('-time')
        return results[0] if len(results) > 0 else {}
