import json
import pandas as pd
SESSION_ID = "et90faag-8ddb-4809-ac4f-1fd08d935597"

class Event:
    id: str
    session_id: str
    timestamp: str
    event_type: str
    details: dict
    parent_event_id: str


    def __init__(self, id, session_id, timestamp, event_type, details, parent_event_id):
        self.id = id
        self.session_id = session_id
        self.timestamp = timestamp
        self.event_type = event_type
        self.details = details
        self.parent_event_id = parent_event_id
        self.prefix = self.get_prefix()
    
    def get_prefix(self):
        # Return the prefix based on the event type after this event is completed
        if self.event_type == 'suggestion_accepted':
            return self.details.get('suggestion', '')
        if self.event_type == 'keystroke':
            return self.details.get('prefix', '')
        if self.event_type == 'suggestion_provided':
            return self.details.get('prefix', '')
        if self.event_type == 'completed':
            return self.details.get('final_text', '')
        if self.event_type == 'suggestion_rejected':
            return self.details.get('final_text', '')
        if self.event_type == 'keystroke':
            return self.details.get('prefix', '') + self.details.get('key', '')
        return ""

    def __repr__(self):
        return f"Event(id={self.id}, session_id={self.session_id}, event_type={self.event_type}, prefix={self.prefix}, details={self.details}, parent_event_id={self.parent_event_id})"



class Session:
    def __init__(self, session_id):
        self.session_id = session_id
        self.logs = load_session_logs(session_id)
        self.events = [Event(id=row['event_id'],
                           session_id=row['session_id'],
                           timestamp=row['timestamp'],
                           event_type=row['event_type'],
                           details=json.loads(row['details']),
                           parent_event_id=row['parent_event_id']) for index, row in self.logs.iterrows()]
        self.event_by_id = {event.id: event for event in self.events}
        self.is_completed = self.is_completed_fn()
        self.last_event = self.get_last_event()
        self.rating = self.last_event.details.get('rating', None) if self.last_event else None
        self.did_start = self.did_start_fn()
        self.gt = self.last_event.prefix if self.last_event else None
        self.flow = []
        self.emulate(self.last_event)



        self.sanity_check()


        self.effort_wo_tab = self.get_total_effort(tab_is_effort=False)
        self.effort_w_tab = self.get_total_effort(tab_is_effort=True)
        self.length = len(self.gt) if self.gt else 0

        self.tes_prev = self.effort_wo_tab / self.length if self.length > 0 else 0
        self.tes_new = self.effort_w_tab / self.length if self.length > 0 else 0

    def sanity_check(self):
        # Check if the session ID is valid
        if not self.session_id:
            raise ValueError("Session ID cannot be empty")
        # Check if the logs are loaded correctly
        if not self.events:
            raise ValueError("No events found for the session")
        if not self.is_completed:
            raise ValueError("Session is not completed")
        num_sugg_rej = sum(1 for event in self.flow if event.event_type == 'suggestion_rejected')
        print(f"Number of suggestion rejections: {num_sugg_rej}")

    def is_completed_fn(self) -> bool:
        # Check if the session has a completed event
        return any(event.event_type == 'completed' for event in self.events)

    def did_start_fn(self) -> bool:
        # Check if the session has a start event
        return any(event.event_type == 'model_assigned' for event in self.events)

    def get_last_event(self):
        # Get the last event type from the session logs
        return [event for event in self.events if event.event_type == 'completed'][-1] if self.events else None

    def get_total_effort(self, tab_is_effort: bool):
        eff = 1
        for event in self.flow:
            if event.event_type == 'suggestion_rejected':
                eff += 1
            # elif event.event_type == 'keystroke':
            #     eff += 1
            elif event.event_type == 'suggestion_accepted':
                eff += 1 if not tab_is_effort else 0
        return eff

    
    def emulate(self, last_event):
        # Emulate the last event in the session
        if not last_event or "nan" in str(last_event).lower():
            # self.flow.append("Starting session")
            print("start")
            return
        self.emulate(self.event_by_id[last_event.parent_event_id])
        print(last_event.event_type, last_event.prefix)
        self.flow.append(last_event)

def load_session_logs(session_id):
    # Load the session logs from the CSV file
    df = pd.read_csv('session_logs.csv')
    
    # Filter the DataFrame for the given session_id
    session_logs = df[df['session_id'] == session_id]
    
    return session_logs

if __name__ == "__main__":
    session = Session(SESSION_ID)
    print(f"Session ID: {session.session_id}")
    print(f"Is Completed: {session.is_completed}")
    print(f"total_effort: {session.effort_wo_tab=}, {session.effort_w_tab=}")
    print(f"Length: {session.length}")
    print(f"TES (without tab): {session.tes_prev}")
    print(f"TES (with tab): {session.tes_new}")
    print(f"Rating: {session.rating}")
    # session.emulate(session.last_event)