"""RLM Pipeline - End-to-end execution."""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.controller import RLMController
from src.worker import RLMWorker
from src.repl import RLMREPL


def main():
    print("=" * 80)
    print("RLM PIPELINE TEST")
    print("=" * 80)
    
    print("\n[Initializing Controller on GPU 0]")
    controller = RLMController(device="cuda:0")
    
    print("\n[Initializing Worker on GPU 1]")
    worker = RLMWorker(device="cuda:1")
    
    repl = RLMREPL(controller, worker)
    
    test_doc = """
    Episode 1: The Dungeon Entrance
    
    The party approaches a dark dungeon. Bob rolls a 15 for perception.
    Alice casts Light spell (using spell slot 1). They see three goblins.
    Combat begins! Bob rolls 18 to hit, dealing 12 damage. Goblin 1 dies.
    Alice rolls 8 to hit, misses. Goblin 2 rolls 14, hits Bob for 5 damage.
    
    Episode 2: The Treasure Room
    
    The party finds a silver flask and a rusty sword. Bob takes the flask.
    Alice examines the sword - rolls Investigation 12. It's magical!
    They continue deeper. Bob rolls Stealth 7 - fails! Alarm triggers.
    Four skeletons appear. Bob rolls 20 - critical hit! Skeleton destroyed.
    Alice casts Fireball (spell slot 3), rolls damage: 8, 6, 7, 5 = 26 total.
    
    """ * 100
    
    task = "How many times did Bob roll the dice in total?"
    
    result = repl.run(task, test_doc, verbose=True)
    
    print("\n" + "=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    print(f"Success: {result['success']}")
    print(f"Answer: {result['answer']}")
    print("=" * 80)


if __name__ == "__main__":
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    main()
