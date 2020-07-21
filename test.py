from unittests.test_search_engine import TestSearchEngine
from unittests.test_vptree import TestVPTreeNodeCreation

tests = [
    TestSearchEngine(),
    TestVPTreeNodeCreation()
]

for t in tests:
    t.test()
    print(type(t).__name__, 'OK')
