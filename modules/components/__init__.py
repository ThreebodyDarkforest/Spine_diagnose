import os
PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if not os.path.exists(os.path.join(PATH, 'logs')):
    os.makedirs(os.path.join(PATH, 'logs'))