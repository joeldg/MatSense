import sys
import traceback
import atexit

def cleanup():
    print("CLEANUP CALLED")

atexit.register(cleanup)

print("Starting test...")
try:
    sys.path.insert(0, 'third_party/wham')
    from configs import constants as _C
    from lib.models.smpl import SMPL
    import torch
    
    print("Creating SMPL...")
    model = SMPL(model_path=_C.BMODEL.FLDR, gender='neutral', batch_size=1, create_transl=False)
    print("SMPL created!")
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc()
except BaseException as e:
    print("CAUGHT BASE EXCEPTION:")
    traceback.print_exc()
finally:
    print("FINALLY BLOCK CALLED")
