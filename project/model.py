
class Event:
    def __init__(self, node, time , parent=None, topic=None):
        self.node = node
        self.time = time
        self.parent = parent
        self.topic = topic
        self.document = None

    def set_topic(self, topic):
        self.topic = topic

    def set_doc(self, doc):
        self.doc = doc
