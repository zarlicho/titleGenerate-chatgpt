keyword = "why is my cholesterol so high" # Change the keyword to whatever you want (ex. keyword = "best coffee machines")

#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

# Optiona parameters

language = 'en' # english => accepts 'en' (English), 'es' (Spanish), 'fr' (French), 'de' (German), 'it' (Italian), rest of languages model and stopwords should be checked

country = 'us' # USA => for example for United Kingdom use country = 'uk' or for Spain 'es'

optionalHeaders = ['h2']  # feel free to remove h3 o add h4

#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

 

Headings0 = []

 

print("Installing dependencies...")

!pip install requests

import requests

 

import requests

from bs4 import BeautifulSoup

import openai

openai.api_key = "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"

 

print("Scrapeando SERPs...")

 

# Object that will contain info of each scraped page

class pagina():

    def __init__(self, url):

        self.url = url

        self.posicion = 0

        self.headings = []

 

# Class that I use it to scrape the SERP

class serp():

 

    # We start with the keyword (query)

    def __init__(self, query):

        self.query = query.replace(" ", "+")

        self.ok = False

        self.incidendias = []

        self.paginas = []

        self.start()

 

    def start(self):

        URL = f"https://www.google.com/search?hl={language}&gl={country}&q={keyword}&oq={keyword}" #url to scrape with keyword, country and language

        headers =  {"user-agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.104 Safari/537.36"}

        enlacesSerp = []

        resp = requests.get(URL, headers = headers)

        if resp.status_code == 200: # if all is ok we continue

            soup = BeautifulSoup(resp.content, "html.parser")

            links = soup.find_all("div", {"class" : "g"}) # google results

            print("Procesando urls:")

            contador = 0

            for x in links: # for each result

                links0 = x.find_all("a", href=True) # we find the url

                if len(links0) == 0:continue

                link = links0[0]['href']

                # I correct the snippet url

                if "#:~:text" in link: 

                    link = link.split("#:~:text")[0]

                if link.startswith("/"):continue

                print(link)

 

                if not link in enlacesSerp: # I check if there are repetead results

                    enlacesSerp.append(link)

                    contador = contador + 1

                    try:

                      resp = requests.get(link, headers = headers, timeout=10)

 

                    except Exception as e:

                      self.incidendias.append(f"timeout error: {str(e)} url: {link}")

                      continue

 

                    soup = BeautifulSoup(resp.content, "html.parser")

 

                    if resp.status_code == 200:

                        pag = pagina(link)

                        pag.posicion = contador

                        # headings

                        j = 1

                        for heading in soup.find_all(optionalHeaders):

                            if heading.text not in pag.headings:

                                textoHeading = heading.text.strip('\n').strip()

                                pag.headings.append([heading.name, heading.text.strip('\n').strip(), contador, j])

                                j = j+1

 

                        Headings0.extend(pag.headings)

                        self.paginas.append(pag)

 

                    else:

                        self.incidendias.append(f"Status code: {resp.status_code} url: {link}")

 

            if len(self.paginas) < 5:

                self.ok = True

                self.incidendias.append("Less than 5 results scraped")

            else:

                self.ok = True

 

        else:

            self.ok = False

 

        # I report scraping incidents

        print(str(len(self.paginas)) + " pages scraped correctly")

 

        if len(self.incidendias) > 0:

            print("incidents")

            print("---------")

            for x in self.incidendias:

                print(x)

 

# here we start scraping

scrap = serp(keyword)

 

if scrap.ok == False:

    print("The scrap has not been completed, the analysis stops => Sorry...")

    quit()

 

encabezados0 = [hea[1].strip() for hea in Headings0 if hea[1].strip() != ''] # removing empty headings

encabezados0 = list(set(encabezados0)) # removing duplicated

 

print("Installing Transoformers 🤖 to check similarity")

!pip install sentence_transformers

from sentence_transformers import SentenceTransformer, util

 

modelo = 'symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli'

 

if language == 'en':

  modelo = 'sentence-transformers/all-MiniLM-L6-v2'

elif language == 'es':

  modelo = 'hiiamsid/sentence_similarity_spanish_es'

elif language == 'fr':

  modelo = "symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli"

elif language == 'de':

  modelo = 'Sahajtomar/German-semantic'

elif language == 'it':

  modelo = 'efederici/sentence-bert-base'

 

model = SentenceTransformer(modelo)

 

# me bajo stopwords

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

 

# Me creo una clase para puntuar similitud y par clusterizar

class encodings():

    def __init__(self, lista):

        self.lista  = [x.lower() for x in lista.copy()]

        self.lista = list(set(self.lista))

        self.embeddings = model.encode(self.lista, batch_size=64, show_progress_bar=True, convert_to_tensor=True) # pasar a vectores

 

    def calcularSimilitud(self, key):

      query_emb = model.encode(key) # keyword to vector

      scores = util.cos_sim(query_emb, self.embeddings)[0].cpu().tolist() # puntuo los headings // opcional dot_score 

 

      # I combine headings and scores

      doc_score_pairs = list(zip(self.lista, [round(x,2) for x in scores]))

      doc_score_pairs = [list(x) for x in doc_score_pairs] # prefiero lista

      doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True) # ordeno por puntuaciones

      self.puntuaciones = doc_score_pairs

 

    # in the end I have decided not to cluster or group, but I save the code for other projects 😉

    def crearClusters(self, minimoElementos=2, thres=0.9):

      self.clusters00 = []

      clusters = util.community_detection(self.embeddings , min_community_size=minimoElementos, threshold=thres, init_max_size=len(self.lista))

      for keywordX, cluster in enumerate(clusters):

            clusterX = []

            for sentence_id in cluster[0:]:

                clusterX.append(self.lista[sentence_id])

            self.clusters00.append(clusterX)

    def agrupar(self):

      borrar= []

      for x in self.clusters00:

        punt = [y for y in self.puntuaciones if y[0] in x]

        punt[0].append(" - ".join([xx[0] for xx in punt[1:]]))

 

        borrar.extend([xx[0] for xx in punt[1:]])

 

      self.puntuaciones = [x for x in self.puntuaciones if x[0] not in borrar]

 

 

import pandas as pd

# function to display a table

from google.colab import data_table

def pasarATabla(lista,columnas, puntuacionMinima=0):

  if puntuacionMinima !=0:

    lista = [x for x in lista if x[1] > puntuacionMinima]

  lista = pd.DataFrame (lista, columns = columnas )

  lista = data_table.DataTable(lista, include_index=True, num_rows_per_page=20)

  display(lista)

 

# scoring headings

procesando = encodings(encabezados0) 

procesando.calcularSimilitud(keyword)

 

puntuaciones = procesando.puntuaciones.copy()

textoHeadings = [x[0] for x in puntuaciones]

 

headingsTodos = [] # retrieving type (h1, h2..) and position in SERP

for pagina in scrap.paginas:

  headingsTodos.extend(pagina.headings)

 

for encabezado in headingsTodos: # I join it with scores

  encabezadoLower = str(encabezado[1]).lower().strip()

  indice = textoHeadings.index(encabezadoLower) if encabezadoLower in textoHeadings else -1

  if indice != -1:

    encabezado.append(puntuaciones[indice][1])

  else:

    encabezado.append(0)

 

headingsTodos = sorted(headingsTodos, key=lambda x: (x[2], x[3])) # I order first by position in SERP and in headers

headingsTodos = sorted(headingsTodos, key=lambda x: x[4], reverse=True) # Finally I order by score, this prevails over the previous one.

 

pasarATabla(headingsTodos, ['H','Header','Position in SERP','Position in Headings', 'Score']) # to table

 

payload = "Hello i want to make article with this title : '{datax}' and i have candidates of my h2 heading for my article. below is the list of heading:"

def generateData(data):

  prmt = "Q: {qst}\nA:".format(qst=data)

  response = openai.Completion.create(

  model="text-davinci-003",

  prompt="Hello i want to make article with this title : \"why is my cholesterol so high\" and i have candidates of my h2 heading for my article. below is the list of heading:\n\nWhy Is My Cholesterol High?\nCauses of high cholesterol\nCauses of high cholesterol\nHigh cholesterol\nWhat causes high cholesterol?\nWhat causes high cholesterol?\nAbout high cholesterol\nWhat Is High Cholesterol?\nWhy should I lower my cholesterol?\nWhat should my cholesterol levels be?\nHow can I lower my cholesterol level?\nDiagnosing high cholesterol\nHigh cholesterol symptoms\nDiagnosis of High Cholesterol\nAbout cholesterol\nTreating high cholesterol\nPreventing high cholesterol\nLiving with high cholesterol\nWhat is cholesterol?\nComplications of high cholesterol\nWhat is cholesterol?\nHigh cholesterol treatment\nCan high cholesterol be prevented or avoided?\nHow to prevent high cholesterol\nTreatment for High Cholesterol\nSymptoms of high cholesterol\nGetting your cholesterol levels checked\nHow to lower cholesterol\nWays to Prevent High Cholesterol\nHow is high cholesterol diagnosed?\nSigns and Symptoms of High Cholesterol\nGetting a cholesterol test\nRisk factors for high cholesterol\nRecent guidelines for healthy cholesterol levels\nCholesterol Resources\nWhen should my cholesterol levels be tested?\nCholesterol-lowering medication\nLDL cholesterol, or “bad cholesterol”\nHDL cholesterol, or “good cholesterol”\nFamilial hypercholesterolaemia\nFamilial hypercholesterolaemia\nTriglycerides, a different type of lipid\nOmega-3 fatty acids\nA High-Sugar Diet\nLiver Problems\nMenopause\nToo Much Alcohol\nDiet\nDiet\nCauses\nThyroid Issues\nOther factors\nKidney Problems\nType 2 Diabetes\nQuestions to ask your doctor\nSmoking\nTake a look at your lifestyle\nStress\nMedications\nFrom Mayo Clinic to your inbox\nSitting a Lot\nMore health news + info\nExercise\nPregnancy\nSymptoms\nRisk factors\nSpeed Bump\nComplications\nA Word From Verywell\nPrevention\nLifestyle\nAdvertisement\nHelp us improve NHS inform\nUnderlying conditions\nWho should be tested?\nResults\nUnfiltered Coffee\nHeredity can play a role\nRelated\nSummary\nOverview\n\n\nAbout Us\nFeedback Alert Title\nMake the changes worth making\nRelated Articles\nResources\nContents\nContact Us\nGet Involved\nTakeaway\nSupport links\nOur Sites\n\nPrompt: please help determine between 3 to 6 headings that are the best for my article and sort them as an outline so they have a good flow. don't force it if only 3 headings are suitable for my article then it is better than 6 headings but some are not suitable. please don't add \" or quote on your answer and without number just raw text.\n\nOutline: \n\nWhat Is High Cholesterol?\nCauses of High Cholesterol\nWhat Causes High Cholesterol?\nHow Can I Lower My Cholesterol Level?\nDiagnosing High Cholesterol\nTreating High Cholesterol",

  temperature=0.7,

  max_tokens=256,

  top_p=1,

  frequency_penalty=0,

  presence_penalty=0

)

  return response.choices[0].text

for x,i in enumerate(headingsTodos):

  datas = headingsTodos[x]

  payload = 'Hello i want to make article with this title : "{datax}" and i have candidates of my h2 heading for my article. below is the list of heading: '.format(datax=datas[1])

  print(payload)

  print(x,generateData(payload))
