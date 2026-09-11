import xml.etree.ElementTree as ET
import pathlib

NS = "https://www.larkoffice.com/sml/2.0"
ET.register_namespace("", NS)
tree = ET.parse("current.xml")
root = tree.getroot()
out = pathlib.Path("pages_v3")
out.mkdir(exist_ok=True)
slides = root.findall(f"{{{NS}}}slide")
print("slides:", len(slides))
for i, sl in enumerate(slides, 1):
    sid = sl.get("id")
    ET.ElementTree(sl).write(out / f"p{i:02d}_{sid}.xml", encoding="utf-8", xml_declaration=False)
    print(i, sid)
