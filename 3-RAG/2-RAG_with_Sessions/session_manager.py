# session_manager.py
import json
import os
from datetime import datetime

class SessionManager:
    def __init__(self, sessions_dir="sessions"):
        self.sessions_dir = sessions_dir
        self.current_session_id = None
        self.current_history = []
        
        # Create sessions folder if it doesn't exist
        if not os.path.exists(sessions_dir):
            os.makedirs(sessions_dir)
    
    def new_session(self):
        """Start a brand new session"""
        from uuid import uuid4
        self.current_session_id = str(uuid4())[:8]
        self.current_history = []
        self._save()
        print(f"\n✨ New session created: {self.current_session_id}")
        return self.current_session_id
    
    def load_session(self, session_id):
        """Load an existing session"""
        filepath = os.path.join(self.sessions_dir, f"{session_id}.json")
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.current_session_id = session_id
                self.current_history = data['history']
            print(f"\n📂 Loaded session: {session_id}")
            print(f"   Messages: {len(self.current_history)}")
            return True
        else:
            print(f"\n❌ Session {session_id} not found")
            return False
    
    def add_exchange(self, user_msg, bot_msg):
        """Save a question and answer to history"""
        exchange = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "user": user_msg,
            "bot": bot_msg
        }
        self.current_history.append(exchange)
        self._save()
    
    def _save(self):
        """Save current session to disk"""
        filepath = os.path.join(self.sessions_dir, f"{self.current_session_id}.json")
        data = {
            "session_id": self.current_session_id,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "history": self.current_history
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_recent_context(self, last_n=3):
        """Get last N exchanges for context"""
        if not self.current_history:
            return ""
        
        recent = self.current_history[-last_n:]
        context = "Previous conversation:\n"
        for exchange in recent:
            context += f"User: {exchange['user']}\nAssistant: {exchange['bot']}\n"
        
        return context
    
    def list_sessions(self):
        """Show all available sessions"""
        sessions = []
        for file in os.listdir(self.sessions_dir):
            if file.endswith('.json'):
                filepath = os.path.join(self.sessions_dir, file)
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    sessions.append({
                        'id': data['session_id'],
                        'created': data['created'],
                        'messages': len(data['history'])
                    })
        return sorted(sessions, key=lambda x: x['created'], reverse=True)

# Test the session manager
if __name__ == "__main__":
    sm = SessionManager()
    
    # Create a session
    sid = sm.new_session()
    sm.add_exchange("Hello", "Hi there!")
    sm.add_exchange("What's your name?", "I'm BrewBot!")
    
    # Get context
    print("\nContext for next question:")
    print(sm.get_recent_context())
    
    # List sessions
    print("\nAll sessions:")
    for s in sm.list_sessions():
        print(f"  {s['id']}: {s['messages']} messages")