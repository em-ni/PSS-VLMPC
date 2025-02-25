from src.tracker import Tracker
from src.durability import Durability

if __name__ == "__main__":
    tracker = Tracker()
    durability = Durability()
    tracker.run()
    durability.run()