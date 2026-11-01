"""Build an IEEE-style PDF and editable Word draft from manuscript.md."""
from pathlib import Path
import re
import typst
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parent
lines = (ROOT / 'manuscript.md').read_text(encoding='utf-8').splitlines()
blocks=[]
i=0
while i < len(lines):
    s=lines[i].strip()
    if not s: i+=1; continue
    if s.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            rows.append([x.strip() for x in lines[i].strip().strip('|').split('|')]); i+=1
        blocks.append(('table',rows)); continue
    if s.startswith('!['): blocks.append(('image',re.search(r'\(([^)]+)\)',s).group(1)))
    elif s.startswith('### '): blocks.append(('sub',s[4:]))
    elif s.startswith('## '): blocks.append(('heading',s[3:]))
    elif s.startswith('# '): blocks.append(('title',s[2:]))
    elif s.startswith('*') and s.endswith('*') and not s.startswith('**'): blocks.append(('caption',s.strip('*')))
    else: blocks.append(('paragraph',s.replace('**','')))
    i+=1

def esc(s):
    return ''.join('\\'+c if c in '\\#[]$*_@<>' else c for c in s)

out=['#set page(paper: "us-letter", margin: (x: 0.68in, y: 0.65in), columns: 2, footer: context align(center, text(size: 8pt, counter(page).display())))', '#set text(font: "Times New Roman", size: 10pt)', '#set par(justify: true, leading: 0.55em, spacing: 0.65em)', '#set place(clearance: 6pt)']
title=blocks.pop(0)[1]; authors=blocks.pop(0)[1]; affiliation=blocks.pop(0)[1]
out.append('#place(top, scope: "parent", float: true, clearance: 12pt)[#align(center)[#text(size: 17pt, weight: "bold")['+esc(title)+'] #v(8pt) #text(size: 10pt)['+esc(authors)+'] #linebreak() #text(size: 9pt)['+esc(affiliation)+']]]')
skip=set()
for idx,(kind,s) in enumerate(blocks):
    if idx in skip: continue
    if kind=='heading': out.append('#v(9pt)\n#block(below: 4pt, sticky: true)[#align(center)[#text(weight: "bold")['+esc(s)+']]]')
    elif kind=='sub': out.append('#v(7pt)\n#block(below: 3pt, sticky: true)[#text(weight: "bold")['+esc(s)+']]')
    elif kind=='paragraph':
        # Balance the final reference page in the current manuscript layout.
        if s.startswith('[3] M. M. Ali et al.'): out.append('#colbreak()')
        out.append('#par['+esc(s)+']')
    elif kind=='caption': out.append('#par[#text(size: 8pt)['+esc(s)+']]')
    elif kind=='image':
        cap=blocks[idx+1][1] if idx+1<len(blocks) and blocks[idx+1][0]=='caption' else ''
        if cap: skip.add(idx+1)
        out.append('#block(breakable: false)[#image("'+s+'", width: 100%)\n#v(3pt)\n#text(size: 8pt)['+esc(cap)+']]')
    elif kind=='table':
        cells=','.join('['+esc(c)+']' for row in s for c in row)
        weights={3:'(1fr, 1.2fr, 1.45fr)',4:'(1fr, 0.7fr, 1.3fr, 1.3fr)',5:'(1.45fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr)'}
        out.append('#block(breakable: false)[#set text(size: 7.3pt)\n#table(columns: '+weights[len(s[0])]+', stroke: (y: 0.25pt + luma(190)), inset: 2pt, align: left, '+cells+')]')
(ROOT/'manuscript.typ').write_text('\n\n'.join(out),encoding='utf-8')
typst.compile(str(ROOT/'manuscript.typ'),output=str(ROOT/'Research_Paper_IEEE_Draft.pdf'),root=str(ROOT))

doc=Document(); sec=doc.sections[0]; sec.page_width=Inches(8.5); sec.page_height=Inches(11)
sec.left_margin=sec.right_margin=Inches(.68);sec.top_margin=sec.bottom_margin=Inches(.65)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10)
normal.paragraph_format.space_after=Pt(4);normal.paragraph_format.line_spacing=1.0
p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;r=p.add_run(title);r.bold=True;r.font.size=Pt(17)
for line in [authors,affiliation]:
 p=doc.add_paragraph(line);p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(3)
sec=doc.add_section(WD_SECTION.CONTINUOUS); cols=sec._sectPr.xpath('./w:cols')[0];cols.set(qn('w:num'),'2');cols.set(qn('w:space'),'360')
references=False
for kind,s in blocks:
 if kind=='heading':
  references=(s=='REFERENCES')
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(9);p.paragraph_format.space_after=Pt(4);p.add_run(s).bold=True
 elif kind=='sub':
  p=doc.add_paragraph();p.paragraph_format.keep_with_next=True;p.paragraph_format.space_before=Pt(7);p.paragraph_format.space_after=Pt(3);p.add_run(s).bold=True
 elif kind=='paragraph':
  p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.LEFT if references else WD_ALIGN_PARAGRAPH.JUSTIFY
  if s.startswith(('Abstract—','Index Terms—')):
   label,rest=s.split('—',1);p.add_run(label+'—').bold=True;p.add_run(rest)
  else: p.add_run(s)
  if references:
   p.paragraph_format.space_after=Pt(2)
   for run in p.runs: run.font.size=Pt(8.5)
 elif kind=='caption':
  p=doc.add_paragraph(s);p.runs[0].font.size=Pt(8)
  if s.startswith('TABLE '):
   p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
  else: p.alignment=WD_ALIGN_PARAGRAPH.LEFT
 elif kind=='image':
  p=doc.add_paragraph();p.paragraph_format.keep_with_next=True;p.add_run().add_picture(str(ROOT/s),width=Inches(3.35))
 elif kind=='table':
  ncols=len(s[0]); widths={3:[.95,1.05,1.35],4:[.85,.44,1.03,1.03],5:[1.15,.55,.55,.55,.55]}[ncols]
  table=doc.add_table(rows=0,cols=ncols);table.autofit=False
  props=table._tbl.tblPr
  margins=OxmlElement('w:tblCellMar')
  for edge,value in [('top',35),('bottom',35),('left',45),('right',45)]:
   node=OxmlElement('w:'+edge);node.set(qn('w:w'),str(value));node.set(qn('w:type'),'dxa');margins.append(node)
  props.append(margins)
  borders=OxmlElement('w:tblBorders')
  for edge in ['top','bottom','left','right','insideH','insideV']:
   node=OxmlElement('w:'+edge)
   node.set(qn('w:val'),'single' if edge in ('top','bottom') else 'nil')
   if edge in ('top','bottom'): node.set(qn('w:sz'),'6');node.set(qn('w:color'),'333333')
   borders.append(node)
  props.append(borders)
  for col,w in zip(table.columns,widths): col.width=Inches(w)
  for n,row in enumerate(s):
   row_obj=table.add_row(); cells=row_obj.cells
   row_obj._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
   for col_idx,(cell,value,w) in enumerate(zip(cells,row,widths)):
    cell.width=Inches(w)
    cell.text=value
    if n==0:
     tcpr=cell._tc.get_or_add_tcPr(); cellborders=OxmlElement('w:tcBorders'); edge=OxmlElement('w:bottom');edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'777777');cellborders.append(edge);tcpr.append(cellborders)
    for para in cell.paragraphs:
     para.paragraph_format.space_after=Pt(0)
     para.paragraph_format.keep_with_next=(n < len(s)-1)
     para.alignment=WD_ALIGN_PARAGRAPH.CENTER if ncols==5 and col_idx>0 else WD_ALIGN_PARAGRAPH.LEFT
     for run in para.runs: run.font.size=Pt(8);run.bold=(n==0)
doc.save(ROOT/'Research_Paper_IEEE_Editable.docx')
print('Built PDF and editable Word document from manuscript.md')
