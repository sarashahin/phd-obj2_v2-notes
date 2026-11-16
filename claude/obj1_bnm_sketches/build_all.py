import importlib, os, sys
out = "/home/claude/canvas/project"
os.makedirs(out, exist_ok=True)
mods = sys.argv[1:]
for m in mods:
    mod = importlib.import_module(m)
    for name, fn in mod.SHEETS:
        open(os.path.join(out, name), "w").write(fn())
        print(name)
