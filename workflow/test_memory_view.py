import unittest
from memory_view import relevant_lessons
class MemoryOrderTest(unittest.TestCase):
    def test_latest_completed_lessons_win_over_run_name_or_append_order(self):
        rows=[{'task':'target','source_run':name,'completed_at':time} for name,time in [('z-old',10),('b-newest',50),('a-middle',30),('y-recent',40),('x-older',20)]]
        rows.append({'task':'other','source_run':'other','completed_at':100})
        self.assertEqual([r['source_run'] for r in relevant_lessons({'lessons':rows},'target')],['x-older','a-middle','y-recent','b-newest'])
if __name__=='__main__':unittest.main()
