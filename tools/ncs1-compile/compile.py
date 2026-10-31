import sys
sys.path.insert(0, "/tk"); sys.path.insert(0, "/stub")
sys.argv = ["mvNCCompile"] + sys.argv[1:]
import runpy
runpy.run_path("/tk/mvNCCompile.py", run_name="__main__")
