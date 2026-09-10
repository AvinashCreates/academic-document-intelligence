import io
import zipfile
from app import read_docx_file

content = io.BytesIO()
with zipfile.ZipFile(content, mode='w') as z:
    z.writestr('[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>')
    z.writestr('_rels/.rels', '<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>')
    z.writestr('word/document.xml', '<?xml version="1.0" encoding="UTF-8"?>\n<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Hello World</w:t></w:r></w:p></w:body></w:document>')
content.seek(0)

class FakeFile:
    def __init__(self, data, name):
        self._data = data
        self.name = name
        self.type = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    def getvalue(self):
        return self._data
    def read(self):
        return self._data

fake = FakeFile(content.getvalue(), 'test.docx')
print('bytes length', len(fake.getvalue()))
print('output:', repr(read_docx_file(fake)))
