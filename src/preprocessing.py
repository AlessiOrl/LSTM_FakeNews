import spacy 
import csv

def getline(ist, ida, ids, idt, t):
  return "\t".join([str(e) for e in [ist, ida, t.sent, idt, t.text, t.lemma, t.pos, t.dep, t.tag,  t.ent_iob, t.ent_type]]) 

def getline_(ist, ida, ids, idt, t):
  return "\t".join([str(e) for e in  [ist, ida, ids, idt, t.text, t.lemma_, t.pos_, t.dep_, t.tag_, t.ent_iob_, t.ent_type_]]) 


def runspacy(article, id_a = 0, id_s = 0, id_t = 0 ):
  docs = nlp.pipe(article)
  list_of_docs = list(docs)
  txt = ''
  
  for sent in list_of_docs[0].sents:
    for token in sent:
      if token.pos_ == 'SPACE': continue
      txt += getline_(True, id_a, id_s, id_t, token) + "\n"
      id_t +=1
    id_s += 1


  for sent in list_of_docs[1].sents:
    for token in sent:
        if token.pos_ == 'SPACE': continue
        txt += getline_(False, id_a, id_s , id_t, token) + "\n"
        id_t +=1
    id_s += 1


  return txt, id_a+1, id_s, id_t

nlp = spacy.load('en_core_web_sm')
print("model loaded")
out = './data/test.token'
f = open(out, 'a')
f.write("is_title\tid_art\tid_sentence\tid_token\tword\tlemma\tpos\tdep\ttag\tent_iob\tent_type\n")
id_a = id_t = id_s = 0
with open('./data/Fake.csv', newline='') as csvfile:
  for row in csv.reader(csvfile):
    if id_a % 4000 == 0: print('done ' + str(id_a) + ' articles')
    a = [row[0].replace(";",""), row[1].replace(";","")]
    newt, id_a, id_s, id_t = runspacy(a, id_a, id_s, id_t)
    f.write(newt)