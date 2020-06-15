from mongoengine import *


# Create your models here.

class Frame(Document):
    meta = {
        'collection': 'frames',
        'indexes': [
            'time'
        ]
    }
    frame_id = SequenceField(required=True, primary_key=True)
    time = FloatField(required=True)
    frame = StringField(required=True)

    @queryset_manager
    def fetch_latest_frame(doc_cls, queryset: QuerySet):
        results = doc_cls.objects.order_by('-id').first()
        print(results)
        return results if results is not None else {}
