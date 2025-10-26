import json
import sys
import os
# Ensure project root is on sys.path so we can import model_client
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model_client import predict_funding

samples = [
    "A mobile app that uses AI to detect plant diseases for smallholder farmers and suggests treatments.",
    "A payment platform that enables seamless cross-border remittances for freelancers with instant settlement.",
    "An online learning marketplace connecting tutors and learners with adaptive testing and progress tracking."
]
cities = ['Nairobi', 'San Francisco', 'Bangalore']
years = [2025, 2025, 2025]

for i, s in enumerate(samples):
    print('---')
    print('Idea:', s)
    r = predict_funding(s, None, cities[i], years[i])
    print(json.dumps(r, indent=2))
