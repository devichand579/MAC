import pandas as pd




class Event:
    timestamp: str
    event_type: str
    details: str


class Session:
    context: str
    session_id: str
    events: list[Event]

    def __init__(self, session_id: str, events: list[Event] = None):
        self.session_id = session_id
        self.context = ""
        self.events = events if events is not None else []
        self.events = sorted(self.events, key=lambda x: x.timestamp)

    def add_event(self, timestamp: str, event_type: str, details: str):
        event = Event()
        event.timestamp = timestamp
        event.event_type = event_type
        event.details = details
        self.events.append(event)
        self.events = sorted(self.events, key=lambda x: x.timestamp)
        self.complete = self.iscomplete()
        self.gt = self.get_gt()
        self.load_pred_by_prefix()


    def load_pred_by_prefix(self):
        self.pred_by_prefix = {}
        for event in self.events:
            if event.event_type == "suggestion_provided":
                prefix = event.details.get("prefix", "")
                suggestion = event.details.get("suggestion", "")
                self.pred_by_prefix[prefix] = suggestion

    def get_gt(self) -> str | None:
        if self.complete:
            for event in self.events:
                if event.event_type == "completed":
                    return event.details["final_text"]

        

    def iscomplete(self) -> bool:
        return "completed" in [event.event_type for event in self.events]
    




class Stater:

    def refresh(self):
        self.session_logs = pd.read_csv(self.session_logs_path)
        self.context_logs = pd.read_csv(self.context_logs_path)

        # group by session_id and aggregate context
        self.session_logs = self.session_logs.groupby('session_id').agg({
            'timestamp': 'max',
            'event_type': lambda x: ', '.join(x),
            'details': lambda x: ', '.join(x)
        }).reset_index()

    def __init__(self, session_logs_path: str, context_logs_path: str):
        self.session_logs_path = session_logs_path
        self.context_logs_path = context_logs_path

