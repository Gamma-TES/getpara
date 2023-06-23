import xml.etree.ElementTree as ET

tree = ET.parse('Setting.xml')

root = tree.getroot()

print(root)
for child in root.iter('Name'):
    print(child)
    