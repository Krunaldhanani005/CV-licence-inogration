"""Face recognition package (InsightFace + FAISS)."""
from .recognizer import FaceRecognizer, FaceMatch
from .face_db import FaceDatabase
from .faiss_index import FaissIndex

__all__ = ["FaceRecognizer", "FaceMatch", "FaceDatabase", "FaissIndex"]
