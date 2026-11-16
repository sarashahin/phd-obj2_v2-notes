"""Local preview: turn .dc.html boards into plain pages with local fonts and screenshot them."""
import os
import re
import sys
from playwright.sync_api import sync_playwright

FONTS = "/home/claude/fonts/node_modules/@fontsource"
FACE = f"""<style>
@font-face{{font-family:'Caveat';font-weight:700;src:url('file://{FONTS}/caveat/files/caveat-latin-700-normal.woff2')}}
@font-face{{font-family:'Patrick Hand';font-weight:400;src:url('file://{FONTS}/patrick-hand/files/patrick-hand-latin-400-normal.woff2')}}
</style>"""


def plain(src):
    s = src.replace('<script src="./support.js"></script>', FACE)
    s = re.sub(r"<link href=\"https://fonts[^>]+>", "", s)
    s = s.replace("<x-dc>", "").replace("</x-dc>", "").replace("<helmet>", "").replace("</helmet>", "")
    s = re.sub(r'<script type="text/x-dc".*?</script>', "", s, flags=re.S)
    return s


def main(indir, outdir, names):
    os.makedirs(outdir, exist_ok=True)
    with sync_playwright() as p:
        br = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--allow-file-access-from-files"])
        pg = br.new_page(viewport={"width": 1400, "height": 990})
        for n in names:
            src = open(os.path.join(indir, n)).read()
            m = re.search(r'"\$preview":\{"width":(\d+),"height":(\d+)\}', src)
            if m:
                pg.set_viewport_size({"width": int(m.group(1)), "height": int(m.group(2))})
            html = plain(src)
            tmp = os.path.join(outdir, n.replace(".dc.html", ".html"))
            open(tmp, "w").write(html)
            pg.goto("file://" + tmp)
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(outdir, n.replace(".dc.html", ".png")))
            print("ok", n)
        br.close()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
