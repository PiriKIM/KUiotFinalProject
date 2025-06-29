import sqlite3
import os

conn = sqlite3.connect('../pose_landmarks.db')
csv_dir = os.path.join(os.path.dirname(__file__), '../data') 