"""A bounded view of completed experience for a new task trial."""
def relevant_lessons(bank,task,limit=4):
    lessons=[lesson for lesson in bank['lessons'] if lesson['task']==task]
    return sorted(lessons,key=lambda lesson:lesson.get('completed_at',0))[-limit:]
