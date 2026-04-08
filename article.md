---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true slideshow={"slide_type": ""} tags=["title"] -->
# Neural Topic Modeling Approach for Historical Inquiry 
Tracing Forty Years of Audiovisual Memory of the 1983 March for Equality on French Television and the Web (1983–2023)  

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["contributor"] -->
### Sophie Gebeil [![orcid](https://orcid.org/sites/default/files/images/orcid_16x16.png)](https://orcid.org/0000-0002-9883-733X) 
Aix-Marseille Université
<!-- #endregion -->

<!-- #region tags=["contributor"] -->
### Arthur  Lezer
Institut National de l'Audiovisuel
<!-- #endregion -->

<!-- #region tags=["copyright"] -->
[![cc-by-nc-nd](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/) 
©<AMU / INA / FUNDED BY ANR-21-CHIP-0003>. Published by De Gruyter in cooperation with the University of Luxembourg Centre for Contemporary and Digital History. This is an Open Access article distributed under the terms of the [Creative Commons Attribution License CC-BY-NC-ND](https://creativecommons.org/licenses/by-nc-nd/4.0/)

<!-- #endregion -->

```python tags=["cover", "figure-cover-*"]
from IPython.display import Image, display
import os
display(Image("./media/header.png"))
```

<!-- #region editable=true slideshow={"slide_type": ""} tags=["keywords"] -->
Digital humanities, Protests, Immigration, France
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"edt0o": [{"id": "10163145/WS6BRI54", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} tags=["abstract"] -->
This article explores how artificial intelligence can assist historical research by enabling the large-scale, diachronic analysis of audiovisual and transmedia narratives. Through a two-step pipeline combining deep learning-based tools for automatic speech recognition (WhisperX) and neural topic modeling (BERTopic, <cite id="edt0o"><a href="#zotero%7C10163145%2FWS6BRI54">(Grootendorst 2022)</a></cite>), we examine how media narratives surrounding the 1983 March for Equality and Against Racism have evolved over four decades. Drawing on a corpus of 150 hours of French television and web media archives preserved by INA, this study investigates how AI can help surface marginalized narratives and support historians in navigating complex, transmedia corpora.
<!-- #endregion -->

## Introduction

<!-- #region tags=["abstract"] -->
Conducted within the European Polyvocal Interpretation of Contested Colonial Heritage project ([PICCH](https://anr.fr/Project-ANR-21-CHIP-0003){:target="_blank"}, 2021–2024), in collaboration with the [INA Lab](https://inalelab.hypotheses.org/){:target="_blank"}, this study adopts a transmedia perspective to examine how representations of the 1983 March have evolved in relation to public debates on immigration and colonial legacies in France. It interrogates both the visibility and portrayal of the marchers, and the extent to which references to France’s colonial past—particularly the Algerian War—have shaped media narratives over time.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"houss": [{"id": "10163145/UXQWLN6C", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} tags=["abstract"] -->
Addressing these questions requires an automated approach that combines distant reading with qualitative analysis.  We applied a two-step AI pipeline—combining automatic speech recognition (WhisperX) and topic modeling (BERTopic)—to transcribe and analyze a corpus of 140 hours of French television and web media archives (102 TV broadcasts and 314 web videos) preserved by the French National Audiovisual Institute (INA). For each television and web video, speech was transcribed using the WhisperX automatic speech recognition system (<cite id="houss"><a href="#zotero%7C10163145%2FUXQWLN6C">(Bain et al. 2023)</a></cite>). Then, we compared several variants of the BERTopic pipeline for the task of dynamic topic modelling on ASR-generated text. Most notably, we assessed the impact of several variants of pre-trained BERT neural models on topic quality, both qualitatively and quantitatively, using coherence and diversity metrics. This comparative experimentation constitutes a methodological contribution to AI-assisted historical research, demonstrating how algorithmic choices influence the interpretability and granularity of historical insights. Moreover, by systematically evaluating different configurations of the BERTopic pipeline—such as embedding models, clustering parameters, and dimensionality reduction techniques—this study demonstrates how algorithmic choices shape the interpretability and granularity of historical insights. We also investigate the potential of BERTopic’s visualization capabilities to serve as a 'corpus explorer' for an audiovisual collection over time. At the hermeneutics layer, this takes the form of Python code to perform the following steps : transcribe the A/V media, fit the Bertopic model on the transcribed text, compare the performance of BERT models, generate Plotly visualisations that allow users to navigate the topics in the form of clusters of documents or time-periods, along with archive extracts and metadata. 

<!-- #endregion -->

<!-- #region tags=["abstract"] -->

These interactive visualizations generated by BERTopic serve as a powerful exploratory interface for historians by mapping topic distributions over time and across clusters of documents. They facilitate a form of distant reading of media discourse that remains anchored in archival specificity. Thus, a key contribution of this study lies in the methodological transparency and reproducibility of its AI-assisted analytical pipeline. Each step—from speech transcription to topic modeling—is thoroughly documented and guided by quantitative evaluation metrics, enabling both interpretability and replicability in the analysis of large-scale audiovisual corpora.
<!-- #endregion -->

<!-- #region tags=["abstract"] -->
Rather than focusing on interpreting the memory of the 1983 March for Equality, this article will center on the methodological challenges involved in offering a distant reading of a transmedia and audiovisual corpus from a historical perspective. It will investigate how artificial intelligence—through automatic speech recognition and neural topic modeling—can assist historians in analyzing the diachronic transformation of media discourse in audiovisual archives, and examines the methodological and epistemological implications of applying such tools to historically biased and heterogeneous corpora.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["abstract"] -->
To explore these questions, the article will unfold in three parts. Firstly, we will situate the study at the intersection of media history and memory studies, outlining the historical context of the 1983 March and the methodological framework developed in collaboration with the INA Lab. Secondly, we will  then detail the AI-assisted pipeline combining automatic speech recognition and neural topic modeling. And thirdly, we will discuss the results obtained, as well as the limitations and future prospects of this approach for historical inquiry. This final section will open a broader reflection on the epistemological implications of integrating AI into historical research. It will particularly address the challenges posed by heterogeneous audiovisual corpora, highlighting the risks of cultural and linguistic bias—such as orthographic inconsistencies and the underrepresentation of minority voices in language models. The section will also underscore the need for transparent, critically informed methodologies capable of supporting both scholarly communities and student training in an era where generative AI is becoming increasingly embedded in everyday academic routines.
<!-- #endregion -->

## Part 1 - Intersecting Frameworks: Media History, Memory Studies, and Digital Humanities in the Interpretation of the 1983 March


<!-- #region editable=true slideshow={"slide_type": ""} -->
The triumphant arrival of the March for Equality and Against Racism on December 3, 1983, in Paris is, in many respects, a significant event in contemporary France. Studying it enables a better understanding of the identity tensions that agitate present society. From a media perspective, it signifies the emergence of the second generation of post-colonial immigration, previously considered a temporary phenomenon. The press and cameras focused particularly on these children of immigrant workers born in large housing estates, who had become young adults denouncing racist crimes and, more broadly, the mechanisms of exclusion they faced. The Maghreb-focused lens led journalists to label this unprecedented anti-racist initiative as the “beurs’ march”, a designation imbued with colonial heritage and reductionism. Indeed, the March is also a post-colonial event in the sense that the violence induced by the Algerian War (1954–1962) still lingers. 
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"1ux0q": [{"id": "10163145/ZHQWCEM3", "source": "zotero"}], "x6bdn": [{"id": "10163145/ZHQWCEM3", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Furthermore, the March represents a significant moment in the immigrant and anti-racist social movement. It originated in the working-class neighborhood of Minguettes in Lyon against a backdrop of tensions between the local youth and law enforcement, whip up by a surge in racist crimes in France (<cite id="x6bdn"><a href="#zotero%7C10163145%2FZHQWCEM3">(Hajjat 2013)</a></cite>). Toumi Djaïdja, then 19 years old and president of the association Avenir Minguette, was injured by arbitrary police gunfire and hospitalized. This incident sparked the idea of crossing France to denounce racist violence and demand better treatment for immigrants and their children. Inspired by figures such as Gandhi (1930), Martin Luther King (1963), and the Larzac farmers (1978), the March picked up support from Father Christian Delorme and the CIMADE network (Inter-Movement Committee for Evacuees) from its inception. 

The first seventeen marchers gathered in Marseille on October fifteenth, 1983, welcomed by local support committees. They journeyed across France until reaching Paris, where they were received at the Élysée Palace by President François Mitterrand, who pledged to grant a 10-year residency permit for immigrant workers (<cite id="1ux0q"><a href="#zotero%7C10163145%2FZHQWCEM3">(Hajjat 2013)</a></cite>). The following year saw the establishment of the SOS Racisme association, following the lead of the Socialist Party, one of the historical anti-racism associations behind the memorable 1985 concert. The “Don’t Touch My Buddy” badge marked an entire generation, but for the marchers and anti-racist activists, SOS Racisme was criticized as a political exploitation of the March, embodying the unfulfilled promises made by the Socialist Party to the inhabitants of working-class neighborhoods.
<!-- #endregion -->

### 1.1. Studying the memorialization of the March


The March has been the subject of studies focusing on its media coverage and its memorialisation. Our inquiry falls within the study of the uses of the past and Memory Studies, approached here through the prism of media memory as produced by television and the web. 

<!-- #region citation-manager={"citations": {"6njc5": [{"id": "10163145/YPJA74PZ", "source": "zotero"}], "c5iqg": [{"id": "10163145/CSHMTCCS", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
An emblematic event of the 1980s ‘Beur’ movement, the March subsequently came to be framed within public narratives as the crucible of SOS Racisme, which later presented itself as its direct heir. A complex and divisive event, the March remained largely overlooked in the 1990s and 2000s. It was not until its thirtieth anniversary that celebrations began to emerge, including several exhibitions and, notably, the 2013 film La Marche by Nabil Ben Yedir. The 2013 commemoration of unprecedented scope made the marchers’ accounts visible and audible, buoyed by unparalleled media coverage: “In just a few years, the March shifted from an event remembered only by a minority of community figures within postcolonial immigrant circles to a historical fact reappropriated by political and media discourses on both the right and the left, to the point of reshaping its historical significance” (<cite id="6njc5"><a href="#zotero%7C10163145%2FYPJA74PZ">(Hajjat 2022)</a></cite>, p.172). Over time, through the uses of the past and the circulation of memorial narratives, two perspectives have emerged without ever fully converging. Following an initial qualitative study highlighting the ambivalences of the memory of this event, which oscillated between nostalgia and bitterness (<cite id="c5iqg"><a href="#zotero%7C10163145%2FCSHMTCCS">(Gebeil 2016)</a></cite>), we aimed to study its representations in audiovisual media and on the web within the PICCH project. 
<!-- #endregion -->

Media imaginaries constitute a privileged lens for highlighting the memorial polyvocality marked by divergent interpretations of the event, as well as the strategies deployed by various actors to occupy an asymmetrical communicative space—one in which former marchers, initially marginalised, gradually gained a degree of visibility from 2013 onwards. 



### 1.2. Television Archive and Web Archive for a transmedia study of the March

<!-- #region citation-manager={"citations": {"5ntjc": [{"id": "10163145/7IB85MLJ", "source": "zotero"}], "u9fgo": [{"id": "10163145/3DVB7YXJ", "source": "zotero"}], "ynkkh": [{"id": "10163145/ZME5VVZN", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
From this perspective rooted in Memory Studies, our analysis also intersects with media history by engaging with the history of the media coverage of the March since 1983. Set within television history, the study is extended by the more recent born‑digital historiography that relies on web sources (<cite id="ynkkh"><a href="#zotero%7C10163145%2FZME5VVZN">(Gebeil 2021)</a></cite>). In France, media history is generally conceived as a cultural history attentive to the social history of media representations (<cite id="5ntjc"><a href="#zotero%7C10163145%2F7IB85MLJ">(Ory 2007)</a></cite>; <cite id="u9fgo"><a href="#zotero%7C10163145%2F3DVB7YXJ">(Eyries 2020)</a></cite>). This field developed in tandem with the rise of the National Audiovisual Institute (INA) from the mid‑1970s, which undertook the archiving of radio and television heritage and facilitated historians’ access to it, notably through digitisation. Media historians examine the conditions of media information production, discourse, and representations in order to uncover media imaginaries, their social anchoring, and their interactions with audiences in the context of reception. Within this project, it is primarily the second dimension that is privileged, in order to explore the transformations of audiovisual imaginaries surrounding the 1983 March over a forty-year period. 

The study focuses on television and the web—both past and present—in relation to the scope of INA, which is responsible for archiving audiovisual media content, and draws on web archiving, an interdisciplinary field to which it also contributes. Web archives are essential for historians working with the web as a source to study recent phenomena: they ensure the citability of born-digital sources and they allow historians to retrieve previous versions of now-defunct websites or to study how a single site has evolved over time.  For this project, we worked with the French web archives. In France, two institutions have been responsible for archiving the national web since the introduction of legal deposit for online content in 2006. The INA (National Audiovisual Institute) archives media-related websites, including those connected to radio, television, cinema, and digital audiovisual platforms, while the BnF (National Library of France) is in charge of archiving other types of websites. 
<!-- #endregion -->

### 1.3. Exploring an audiovisual and  transmedia corpus in a historical perspective: a methodological challenge


On the methodological level, this involves identifying and analyzing audiovisual archives, online videos, and the pages that constitute their publication context, all dealing with the March over the past forty years. These materials form a transmedia corpus that brings together audiovisual sources, born-digital sources, and re-born-digital sources (archived web). Aligned with the historian’s approach, a semio‑discursive study requires analysing these materials by applying internal and external criticism.  When conducted manually, the researcher must employ sampling to reduce the number of items examined, given the time‑consuming nature of such work. Accordingly, the use of computational methods to support the analysis of transmedia materials can yield a more comprehensive view of the corpus and refine the selection of archives for close reading, thereby aligning with the principles of the digital humanities

<!-- #region editable=true slideshow={"slide_type": ""} -->
As part of the study of the media coverage of the March, AI models are used to equip the historian in his data mining for the benefit of a distance reading. The live and archived web are, by definition, born‑digital materials and, as such, provide a fertile terrain for the application of computational methods, including AI. Accordingly, this study draws on techniques well established in live‑web research, while addressing the ongoing challenge of adapting them to transmedia corpora to enable diachronic analysis. We faced three challenges: a transmedia corpus (television and web archives), a predominance of video material, and the requirement to conduct diachronic analysis from a historical perspective.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"14ak8": [{"id": "10163145/HGXGKTED", "source": "zotero"}]}} -->
This study extends an initial 2022 exploration by (<cite id="14ak8"><a href="#zotero%7C10163145%2FHGXGKTED">(Rendina et al. 2024)</a></cite>) that drew on a web‑media corpus about the March and used NLP and deep‑learning methods. Conducted before generative AI became mainstream and was often conflated with AI more broadly, it employed named‑entity recognition, topic modelling, and network analysis (Louvain) to support systematic historical work on an INA web‑archive corpus devoted to the March. Given the prohibitive number of documents (12,688 documents) in the corpus for manual tagging, we tested whether NLP methods could facilitate automatic indexing of the documents by  identifying entities and topics over time. This concerned HTML text of the pages, without including the videos.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
In parallel, we have completed with a corpus from the living web which has also been thematized with BERTopic by the MediaLab of SciencesPo (PICCH report). At the same time, a television corpus was created using materials from the French National Audiovisual Institute (INA) by Pauline Saveant (PhD Student) with the support of Sophie Gillery (INA, Marseille). Then we completed the corpus with the video from the web archive and web alive corpus. 
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
This audiovisual transmedia collection includes 416 programs broadcast on French television and web media, all preserved under the legal deposit framework. This corpus is composed of 102 TV broadcasts and 314 web videos and represents 140 hours of video archives from 1983 to 2023. **Because our corpus includes proprietary content, including transcripts, replication data is available upon demand.**
<!-- #endregion -->

<!-- #region tags=["hermeneutics"] -->
Below code loads the corpus data from `./script/data/corpus.csv` and defines a few helper functions to integrate thumbnails in a data renderer. We display an excerpt of the corpus, comprising 5 examples of archival records for TV (`"corpus"=="tv"`) and Web (`"corpus"=="web"`).  
<!-- #endregion -->

```python
import pandas as pd
doc_df = pd.read_csv('./script/data/corpus.csv')
doc_df['publication_date'] = pd.to_datetime(doc_df.publication_date)
```

```python
from PIL import Image
from io import BytesIO
from IPython.display import HTML
import base64 
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# DEFINE YOUR PROXIES HERE IF NEEDED
# proxies = {
#   "http": "",
#   "https": "",
# }


# helper function to integrate images in df
def is_image_response(response):
    if response is None or not hasattr(response, "content"):
        return False

    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("image/"):
        return True

    # Fallback: try decoding
    try:
        Image.open(BytesIO(response.content))
        return True
    except Exception:
        return False

def get_thumbnail(path, size=(256, 256)):
    if path.startswith('http') :
        r = requests.get(path, 
                         headers={"User-Agent": "Mozilla/5.0"}, 
                         proxies={"http": None, "https": None} if 'ina' in path else None,
                         timeout=10, 
                         verify=False
                         )
        if is_image_response(r) :
            i = Image.open(BytesIO(r.content))
        else :
            i = Image.new("RGB", size, color=(255, 255, 255))
    else :
        i = Image.open(path)
    i.thumbnail(size, Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


### HELPER : URL FORMATTER
def url_formatter(url):
    if not url:
        return ""
    return f'<a href="{url}" target="_blank" rel="noopener noreferrer">{url}</a>'
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["table-corpuspreview-*", "data-table"]
excerpt_df = (doc_df
.drop_duplicates(subset='id')
.set_index('id', drop=False)
.dropna(subset='thumbnail_public_url')
.groupby('corpus')
.apply(lambda g: g.sample(n=(min(len(g),5)), replace=False, random_state=83), include_groups=False)
.reset_index(level=1, drop=True)
.reset_index()
)

HTML(excerpt_df[['thumbnail_public_url','corpus','id','publication_date','title','duration','keywords','channel','tv_show_title','video_public_url']]
    .to_html(formatters= {
        'thumbnail_public_url': image_formatter,
        'video_public_url': url_formatter
        }, 
             escape=False)
)
```

```python editable=true jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} slideshow={"slide_type": ""} tags=["figure-corpusbarplot-*"]
import plotly.express as px

gp = doc_df.groupby([doc_df.publication_date.dt.year, 'corpus'])
sum_web = round(doc_df[doc_df.corpus=="web"].duration.sum() / 3600)
sum_tv = round(doc_df[doc_df.corpus=="tv"].duration.sum() / 3600)
count_web = (doc_df.corpus=="web").sum()
count_tv = (doc_df.corpus=="tv").sum()
# tp = gp.size().rename('count').sort_index().to_frame().reset_index()
tp = (gp.duration.agg('sum')/3600).rename('count').sort_index().to_frame().reset_index()
fig = px.bar(tp, x='publication_date', y='count', color='corpus', barmode='group',
             title=f"""
             Annual aggregated duration of archives in corpus (total N = {len(doc_df)})
             <br>(web N = {count_web} and {sum_web} hours ; tv N = {count_tv} and {sum_tv} hours)
             """,
             color_discrete_sequence=["#A8E9A7","darkgrey"]
             )
fig.update_layout( autosize=False, width=800, height=400)
fig.update_yaxes(gridcolor = "#EDEDED")
fig.update_layout (yaxis_title = 'Total duration of archives',  plot_bgcolor='#FFFFFF', xaxis_title = 'Year', title = {'x':0.5}, legend_title_text='Source')


# html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
# HTML(html)
fig.write_html('./figs/corpus_barplot.html')
fig.show()
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Part 2 - Designing the AI-Assisted Pipeline: Automatic Speech Recognition and Neural Topic Modeling 
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"qxn89": [{"id": "10163145/VXYHFRSB", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Drawing on this corpus, our objective is twofold: to examine the media memories of the March, and to gain a clearer understanding of the marchers themselves, whose presence is only poorly documented in audiovisual archives. Prior to topic modelling, we performed an initial exploration using another AI-driven similarity analysis tool, Snoop, used by researchers at the INA lab. The processing chain used in this software involves segmenting videos into still frames at a rate of 6 frames per second, extracting key visual features. Snoop then allows retrieval of excerpts within the corpus, that are similar to a source image. 
For example, the user can search for images showing Toumi Djaïdja, who initiated the 1983 March after being hospitalized due to a police shooting. The latter made it possible to sift through the visual media excerpts / that the automated search engine had gathered. In our use of Snoop, we observed a clear bias: facial recognition tended to work more effectively for white individuals. We thought that this was a symptom of bias in the training data used for the V3 Inception model, which was trained on large-scale, royalty-free image datasets—primarily from platforms like Flickr, as discussed in (<cite id="qxn89"><a href="#zotero%7C10163145%2FVXYHFRSB">(Fuica and Lezer 2023)</a></cite>) 

<!-- #endregion -->

Based on this observation, we deepened the textual track by deciding to extract the voices of the videos and to combine transcribed textual data with the metadata of the videos in order to build an enriched diachronic and thematic exploration.


### 2.1. Model choices, design and evaluation

<!-- #region citation-manager={"citations": {"by77c": [{"id": "10163145/UXQWLN6C", "source": "zotero"}]}} -->
We transcribe the entirety of the corpus using open-source automatic speech recognition (ASR) software WhisperX (<cite id="by77c"><a href="#zotero%7C10163145%2FUXQWLN6C">(Bain et al. 2023)</a></cite>). WhisperX delivers diarized transcripts, that is, speech is segmented between detected speakers based on voice similarity. The diarization is discarded because of its inaccuracies.
<!-- #endregion -->

<!-- #region tags=["hermeneutics"] -->
#### 2.1.1 Speech-to-text and TV archives : cultural and technical bias

WhisperX is a system relying on a deep learning model trained on very large cultural data external to our corpus of interest, mainly video sub-titles. We encountered two main limitations : (1) **diarization errors**, attributable to [Pyannote](https://github.com/pyannote/pyannote-audio){:target="_blank"}, and (2) Whisper's own **transcription errors, exhibiting culutral bias**.

Table below displays an example of ASR transcript, and the actual video excerpt (transcribed passage starts at `00:00:32`). When compared to the [manually corrected transcript](https://fresques.ina.fr/sudorama/fiche-media/00000000269/l-arrivee-de-la-marche-pour-l-egalite-et-contre-le-racisme-a-paris.html#infos){:target="_blank"} (which can be displayed by clicking on "Synchroniser le texte" in the player controls), we can see WhisperX/Pyannote exhibits under-segmentation, that is, speech turns from different speakers are attributed to the same speaker. These errors are overlapping speech, but also seem to increase as a function of the material's age, perhaps due to sound quality. Because of these limitations, and the fact that Pyannote's speech segments are often larger than the language model's context size, we willdiscard diarization entirely for the rest of the pipeline.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""} tags=["sound-whisperexample-*"]
from IPython.display import Audio
Audio('media/RAC8300168201.mp3')
```

<!-- #region tags=["dialog-whisperexample2-*"] -->
| KMRAC-MA83000171.01_SPEAKER_10 | KMRAC-MA83000171.01_SPEAKER_11 |
| ------ | ------ |
| &nbsp; |  C'est quoi ça ? |
|  Vous êtes venue nombreux de Marseille ? Oui, il y a plusieurs cas. Pourquoi vous êtes montée de Marseille jusqu'à cette manifestation ? Parce qu'on voit la marche. Qu'est-ce que vous croyez ? Qu'est-ce que vous attendez ? | &nbsp; |
| &nbsp; |  Un monde meilleur pour nous, si c'est possible. On ne demande pas la lune, on demande de vivre, c'est tout. Et vous croyez que la manifestation, la marche, va servir à quelque chose ? Oui, à l'intérêt, oui. On est là. Ça prouve au moins qu'on est maintenant capable de s'organiser entre nous. Et vous croyez que ça va durer après la marche ? Oui, il faut le suivre, il y aura un suivi. Il y aura un suivi un peu dans toutes les régions, je crois. Dans toute la région de France. |
|  C'est la victoire de l'humanité ! C'est la victoire de l'humanité ! | &nbsp; |
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["video-jt1-*"]
HTML(""" <iframe
     allow="autoplay"
     title="Vidéo INA"
     src="https://fresques.ina.fr/sudorama/export/player/00000000269"
     width="800" height="450"
     allowfullscreen start="110">
     </iframe> """)
```

<!-- #region citation-manager={"citations": {"uoa9v": [{"id": "10163145/EUF5F7AZ", "source": "zotero"}]}} -->
Whisper's average word error rate (WER) is measured at around 6.59% for the French language (<cite id="uoa9v"><a href="#zotero%7C10163145%2FEUF5F7AZ">(Srivastav et al. 2025)</a></cite>), and concern primarily removal of hesitations or short interruptions, as well as erroneous transcription of proper nouns, such as names and surnames. In the latter case, we observed, that the model exhibits error bias favoring European surnames and is more prone to error on names of Arabic descent. We did not attempt to measure this bias, but as an example, we found more than 25 erroneous transcription variants of the name Toumi Djaïdja (see table below where we keep track of those errors) and none for Christian Delorme. The provided dataset received a regex-based correction for the name Toumi Djaïdja, as well as variants of the term Minguettes (primarily to facilitate manual search), but we left an unknown number of other names untouched.
<!-- #endregion -->

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["table-whispererrors-*", "data-table"]
pd.read_csv('.script/data/transcript_errors.csv', index_col=0).join(doc_df.set_index('id')[['publication_date','channel']])
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
#### 2.1.2 : Chunking
The archive transcripts are segmented into passages based on punctuation marks generated by WhisperX. Once segmented, the corpus comprises approximately 14,500 passages.

<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

# II - Chunking, embedding
def count_tokens(segment, tokenizer):
    """Helper function to count tokens in a text segment."""
    return len(tokenizer(segment, truncation=False, add_special_tokens=True)['input_ids'])

# Define chunkers
long_texte = "\n".join(doc_df.sentences.to_list())
sentence_split_regex = r"(?<=[.?!])\s+"

def chunk_lgc_token(docs, embedding_model, chunk_size = None) :
    
    tokenizer = embedding_model.tokenizer

    def length_function(text):
        return count_tokens(text, tokenizer = tokenizer)
    
    if chunk_size is None :
        # calculer la max_length du modèle
        chunk_size = len(tokenizer(long_texte, truncation=True)['input_ids'])
    
    # Define the chunker
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[sentence_split_regex, "(?<=\. ).", "\n"],
        chunk_size = chunk_size - 2, # pour compter le CLS et END token
        chunk_overlap  = 0,
        length_function = length_function,
        is_separator_regex = True,
        )
    docs = [text_splitter.split_text(doc) for doc in docs]
    return docs

```

```python editable=true slideshow={"slide_type": ""}
get_chunks = chunk_lgc_token
# Create a directory to save models
model_name = "dangvantuan/sentence-camembert-large"
short_model_name = model_name.replace('/', '_')
result_path = f'./script/data/TM/{get_chunks.__name__}/{short_model_name}'
chunk_df_path = f'{result_path}_chunked_df.pkl'
embeddings_path = f'{result_path}_embeddings.pkl'
# os.makedirs(result_path, exist_ok=True)
```

```python editable=true slideshow={"slide_type": ""}
from sentence_transformers import SentenceTransformer
import re
import os

orig_docs = doc_df['sentences'].to_list()

# Define the embedder here
# In some contexts (not here) we use its tokenizer to control the length of chunks 
embedding_model = SentenceTransformer(model_name, device="cuda")

# Chunk the docs
chunked_docs = get_chunks(orig_docs, 
                        embedding_model = embedding_model, 
                        chunk_size = 128 # for this run, we keep the chunk_size fixed
                        )
doc_df['chunked_doc'] = chunked_docs


# Create a dataframe with the chunks
chunk_df = doc_df.copy()
chunk_df = chunk_df.explode('chunked_doc').rename({'chunked_doc':'chunk'}, axis = 'columns')

# remove hallucinations
hallus=[
'jaune, jaune',
'sous-titres',
'sous-titrage',
'oh, oh, oh',
'oh oh oh',
'oh you',
't, t, t',
't t t',
'talk talk',
'talk, talk',
]

hallu_regex = "|".join([re.escape(h) for h in hallus])
filtre_contient_hallu = chunk_df.chunk.str.contains(hallu_regex, 
                                                    case=False, 
                                                    regex=True) 

chunk_df = chunk_df[~ (filtre_contient_hallu | chunk_df.chunk.isna())] # also remove empty chunks

```

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
#### 2.1.3 Adding chunk-level metadata
At this step, we define additional metadata for each of the chunks. We didn't keep word-level timestamps, but we recompute a chunk start timestamp based on the chunk's length (number of words) relative to the length of the parent document.
<!-- #endregion -->

```python
from script.utils import sec_to_tc

chunk_df['doc_size'] = chunk_df.sentences.str.split().apply(len)
chunk_df['chunk_size'] = chunk_df.chunk.str.split().apply(len)

chunk_df["chunk_approx_duration"] = ((chunk_df.chunk_size / chunk_df.doc_size) * chunk_df['duration']).round() # taille du chunk

chunk_df["chunk_start_approx"] = (
    chunk_df.groupby("id")["chunk_approx_duration"]
          .cumsum()
          .shift(fill_value=0)
)

chunk_df['chunk_url'] = chunk_df['url'] + "/?type=stream&redirect=1" + "&additional_params=start=" + chunk_df['chunk_start_approx'].astype(int).astype(str)
chunk_df['chunk_tc'] = chunk_df.chunk_start_approx.apply(sec_to_tc)

# save the chunked data if needed
# chunk_df.to_csv('./script/data/chunked_corpus.csv', index=False)
```

<!-- #region citation-manager={"citations": {"tt5o8": [{"id": "10163145/MRWCSDZ4", "source": "zotero"}], "ymmxe": [{"id": "10163145/ZECEJYDL", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
#### 2.1.4 Generating embeddings from a pre-trained model
A language model is used to represent each passage in the form of a contextual lexical embedding (<cite id="ymmxe"><a href="#zotero%7C10163145%2FZECEJYDL">(Reimers and Gurevych 2019)</a></cite>). A lexical embedding is a vector that can be understood as a set of coordinates in a space determined by the distribution of words across large volumes of text. Passages whose embeddings are similar within this space are assumed to be semantically similar. To this end, we use a model (https://huggingface.co/dangvantuan/sentence-camembert-large) adapted from CamemBERT (<cite id="tt5o8"><a href="#zotero%7C10163145%2FMRWCSDZ4">(Martin et al. 2020)</a></cite>) and specialized for semantic similarity tasks. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import pickle

# Call its tokenizer 
tokenizer = embedding_model.tokenizer

# in the context of bertopic, we'll call "docs" the chunked docs
docs = chunk_df['chunk'].to_list()

# compute embeddings
embeddings = embedding_model.encode(docs, show_progress_bar=True)

# # save embeddings
# with open (embeddings_path, 'wb') as outfile :
#     pickle.dump(embeddings, outfile)
```

The extraction of embeddings takes approximately 2:30 minutes using a consumer-grade graphics card.

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
#### 2.1.5 Setup representation models for C-TF-IDF
We specify text models that will be used for topic representations at a later stage of the pipeline.

<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
# Import bertopic related libs
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from bertopic import BERTopic
import math
from cuml.cluster import HDBSCAN as cu_HDBSCAN
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, PartOfSpeech
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from umap import UMAP
```

```python editable=true slideshow={"slide_type": ""}
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer
from nltk.stem.snowball import SnowballStemmer
import spacy

stemmer = SnowballStemmer(language='french')
nlp = spacy.load("fr_core_news_lg")
word_tokenize = nlp.tokenizer

class LemmaTokenizer:
    def __init__(self):
        self.frl = FrenchLefffLemmatizer(load_only_pos=['adj','adv','nc','np','ver','auxAvoir','auxEtre'])
        self.stemmer=stemmer
    def __call__(self, doc):
        allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN', 'INTJ', 'NUM', 'X']
        tokens = word_tokenize(doc)
        return ([token.lemma_ if (token.pos_ in allowed_postags and '_' not in token.text) else token.text for token in tokens if not token.is_punct and not token.is_space])
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
### 2.2 Model parameterization and evaluation
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"5ji3t": [{"id": "10163145/3AE44DD7", "source": "zotero"}], "h63bk": [{"id": "10163145/ZPL7888X", "source": "zotero"}], "rkawl": [{"id": "10163145/I2H36MMV", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Topic modeling makes no assumption about the number or nature of topics. This is true for the seminal method Latent Dirichlet Allocation (<cite id="rkawl"><a href="#zotero%7C10163145%2FI2H36MMV">(Blei 2003)</a></cite>) and its successors, whose key parameter—among many that can be tuned—is the number of topics $K$ to be discovered. We aim to learn a model where the assignment of documents to $K$ clusters reflects both the diversity and coherence of latent topics (themes, lexical fields) within the corpus.
There are two main ways to evaluate the quality of a topic model (<cite id="5ji3t"><a href="#zotero%7C10163145%2F3AE44DD7">(EURECOM, Sophia Antipolis, France et al. 2021)</a></cite>) to determine the optimal value of $K$ :

1. Comparison to a ground truth, i.e., a set of documents manually annotated with topics.
2. Endogenous metrics, which is the method we employ here.

We use two of the most common endogenous metrics:

- $Cv$ Coherence (<cite id="h63bk"><a href="#zotero%7C10163145%2FZPL7888X">(Röder, Both, and Hinneburg 2015)</a></cite>), which measures the similarity of the contexts where the top-words of a topic appear within the corpus.
- Diversity, which measures how distinct the top terms are across topics, penalizing redundancy.   



<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
#### 2.2.1 Coherence metric
We use [gensim](https://radimrehurek.com/gensim/models/coherencemodel.html)'s implementation of the $Cv$ score, which is calculated by first, computing $K$ Coherence scores at the topic level, and then taking the arithmetic mean of those scores.  
Coherence is defined, for a topic with top-N words $W$, as the similarity of each of its words context, to the general topic context, like so :
$$C_v(W) = \frac{1}{N} \sum_{i=1}^{N} \text{cosim}(\vec{v}(w_i), \vec{v}(W))$$

where the context vectors $\vec{v}(w_i)$, and $\vec{v}(W)$ of size $V$ describe, respectively, the context of each of the words $w_i$, and the overall context of the topic words $W$, in the vocabulary, as follows :

$$\vec{v}(w_i) = \left[\sum_{w_k \in \{w_i\}} \text{NPMI}(w_k, w_j)^\gamma\right]_{j \in V}$$
$$\vec{v}(W) = \left[\sum_{w_k \in W} \text{NPMI}(w_k, w_j)^\gamma\right]_{j \in V}$$

( $\vec{v}(w_i)$ _can also be written as_ $\left[\text{NPMI}(w_k, w_j)^\gamma\right]_{j \in V}$ )

where $V$ is the vocabulary, $\gamma$ is a tunable parameter that amplifies or diminishes differences between associations, which we set to the default 1, and $NPMI$ (Normalized Pointwise Mututal Information) is :

$$\text{NPMI}(w_i, w_j) = \frac{\log \frac{P(w_i, w_j) + \epsilon}{P(w_i) \cdot P(w_j)}}{-\log(P(w_i, w_j) + \epsilon)}$$

The probabilities $P(w_i)$, $P(w_j)$ and $P(w_i, w_j)$ are estimated using a boolean sliding window over the corpus, where each window position creates a virtual document for counting word occurrences and co-occurrences.

Finally the cosine similarity is :

$$\text{cosim}(\vec{v}(w_i), \vec{v}(W)) = \frac{\vec{v}(w_i) \cdot \vec{v}(W)}{\|\vec{v}(w_i)\| \cdot \|\vec{v}(W)\|}$$
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

#### Define metrics
def get_coherence_score(bertopic_model, docs) :
    # Preprocess documents
    cleaned_docs = bertopic_model._preprocess_text(docs)

    # Extract vectorizer and analyzer from BERTopic
    vectorizer = bertopic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()

    # Extract features for Topic Coherence evaluation
    words = vectorizer.get_feature_names()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    # Extract words in each topic if they are non-empty and exist in the dictionary
    topic_words = []
    for topic in range(len(set(bertopic_model.topics_))-bertopic_model._outliers):
        words = list(zip(*bertopic_model.get_topic(topic)))[0]
        words = [word for word in words if word in dictionary.token2id]
        topic_words.append(words)

    topic_words = [words for words in topic_words if len(words) > 0]

    # Evaluate
    coherence_model = CoherenceModel(topics=topic_words, 
                                    texts=tokens, 
                                    corpus=corpus,
                                    dictionary=dictionary, 
                                    coherence='c_v')
    coherence = coherence_model.get_coherence()
    return(coherence)
```

<!-- #region citation-manager={"citations": {"0cian": [{"id": "10163145/Q6SU9JHQ", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
#### 2.2.2 Diversity metric
The Diversity measure can be written as :
$$\text{Diversity} = \frac{\left| \bigcup_{k=1}^{K} W_k \right|}{K \times M}$$

where :  
- $W_k$ is the set of top-M words for topic k,  
- $\bigcup_{k=1}^{K} W_k$ is the union of all these sets (unique words),  
- $K$ is the number of topics,  
- $M$ is the number of top words considered per topic, which we set at $M = 20$ 

We use the diversity implementation from the python package octis (<cite id="0cian"><a href="#zotero%7C10163145%2FQ6SU9JHQ">(Terragni et al. 2021)</a></cite>).
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
from octis.evaluation_metrics.diversity_metrics import TopicDiversity 

diversity = TopicDiversity()

def get_diversity_score(bertopic_model) :
    l=[]
    for topic in bertopic_model.get_topics().values() :
        l.append([top_word for (top_word, score) in topic])
    l = {"topics":l}
    return (diversity.score(l))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
#### 2.2.3 Evaluation of a topic model
We compute a series of Bertopic models and tune the following parameters inside a predetermined range.
- chunking strategy
- embedding model
- UMap's `n_components`
- HDBSCAN's `min_cluster_size`
- HDBSCAN's `min_samples`  

For each of those models, we measure Coherence and Diversity. In the following section, we present only an evaluation of the effect of HDBSCAN's `min_cluster_size`, which determines the minimal number of documents per cluster, and in turn, the number of topics $K$ : a higher `min_cluster_size` naturally results in fewer topics. We set another parameter of HDBSCAN, `min_samples`, which determines the proportion of documents that can remain assigned to no clusters, at a value of $0.15 \times$ `cluster_size`.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
The code below will train the BERTopic pipeline with specified hyperparameters.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import os 
# Make a directory to save the models
os.makedirs(f'./script/data/TM/{get_chunks.__name__}/{short_model_name}/', exist_ok=True)

results={'chunking':[], 
         'model':[],  
         'params':[], 
         'umap_n_components':[], 
         'umap_n_neighbors':[], 
         'hdbscan_cluster_size':[], 
         'hdbcscan_min_samples':[], 
         'bertopic_coherence':[], 
         'bertopic_diversity' : []}

# You can gridsearch these parameters or test all values as we did
p0 = 64 # UMAP n_components
p1 = 145 # UMAP n_neighbors

for p2 in (48,78) : # HDBSCAN min_cluster_size

# uncomment below to compute models for all min_cluster_size)
# for p2 in range(4, 96) :

    p3 = math.ceil(0.15*p2) # HDBSCAN min_samples

    umap_model = UMAP(n_neighbors=p1, n_components=p0, min_dist=0.0, metric='cosine', random_state=42)
    
    # Reduce dimensionality of embeddings for visualization. This step is optional but much faster to perform iteratively:
    reduced_embeddings = UMAP(n_neighbors=p0, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)


    param_string = f'umap_{p0}-{p1}_hdbscan_cuml{p2}-{p3}'

    print(f'computing bertopic model for : {get_chunks.__name__}/{short_model_name}/{param_string} ...')

    hyperparams=dict(dim_size = p0, umap_n_neighbors=p1, min_cluster_size=p2, min_samples=p3)
    hdbscan_model = cu_HDBSCAN(hyperparams['min_cluster_size'], hyperparams['min_samples'], metric='euclidean', cluster_selection_method='leaf', prediction_data=True)

    # KeyBERT
    keybert_model = KeyBERTInspired()

    # Part-of-Speech
    pos_model = PartOfSpeech("fr_core_news_lg")

    # MMR
    mmr_model = MaximalMarginalRelevance(diversity=0.3)

    # All representation models
    representation_model = {
    "KeyBERT": keybert_model,
    "MMR": mmr_model,
    "POS": pos_model
    }

    from sklearn.feature_extraction.text import CountVectorizer
    from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop

    vectorizer_model = CountVectorizer(stop_words=list(fr_stop), min_df=0.05, max_df=0.5, ngram_range=(1, 3), tokenizer=LemmaTokenizer())

    topic_model = BERTopic(

    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    # hdbscan_model=kmeans_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,

    # Hyperparameters
    calculate_probabilities=True,
    top_n_words=15,
    verbose=True
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)


    results['model'].append(short_model_name)
    results['params'].append(param_string)
    results['umap_n_components'].append(p0)
    results['umap_n_neighbors'].append(p1)
    results['hdbscan_cluster_size'].append(p2)
    results['hdbcscan_min_samples'].append(p3)
    results['chunking'].append(get_chunks.__name__)

    topic_model.save(f'./script/data/TM/{get_chunks.__name__}/{short_model_name}/{param_string}_bertopic_model.pkl')
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
We save the training metadata separately.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
pd.DataFrame(results).to_csv('./script/data/evaluation.csv')
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
The next figure illustrates how evaluation guides our choice of the "optimal" value for $K$ : the intersection between lines representing Coherence and Diversity (after min-max normalization) represent the best compromise between coherence and diversity.
<!-- #endregion -->

```python editable=true jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} slideshow={"slide_type": ""} tags=["figure-evaluation-*"]
import numpy as np
import pandas as pd

df = pd.read_csv('./script/data/evaluation.csv')

df['coherence_normalized'] = (df['bertopic_coherence'] - df['bertopic_coherence'].min()) / (df['bertopic_coherence'].max() - df['bertopic_coherence'].min())
# df['coherence_normalized'] = (df['bertopic_coherence'] - df['bertopic_coherence'].mean()) / df['bertopic_coherence'].std() # z-score
# df['coherence_normalized'] = df['bertopic_coherence'] / df['bertopic_coherence'].max() # max scaled

df['diversity_normalized'] = (df['bertopic_diversity'] - df['bertopic_diversity'].min()) / (df['bertopic_diversity'].max() - df['bertopic_diversity'].min())
# df['diversity_normalized']  = (df['bertopic_diversity']  - df['bertopic_diversity'].mean())  / df['bertopic_diversity'].std() # z-score
# df['diversity_normalized']  = df['bertopic_diversity']  / df['bertopic_diversity'].max() # max scaled

# Melt the DataFrame to have 'coherence_normalized' and 'diversity_normalized' as a single column
df_melted = df.melt(
    id_vars=[
        "hdbscan_cluster_size", 
            #  "model", 
            #  "umap_n_components", 
            #  'chunking'
               ], 
                     value_vars=["coherence_normalized", "diversity_normalized"], 
                     var_name="metric", 
                     value_name="value")

# Define color and marker mappings
color_map = {'grey'}
marker_map = {'x'}

# Plot 
fig = px.line(
    df_melted,
    x="hdbscan_cluster_size",  # Using hdbscan_cluster_size as X-axis
    y="value",
    # color="model",
    line_dash="metric",  # Different line styles for coherence vs. diversity
    markers=True,
    # symbol="chunking",  # Different markers for parameter_1 and parameter_2
    title="Model evaluation and selection of K",
    # hover_data={"chunking": True}  # Add 'chunking' to tooltip

)

fig.update_layout(
    xaxis_title="HDBSCAN Cluster Size",
    yaxis_title="Metric Value",
    legend_title="Legend",
    template="plotly_white",
    width=900,
    height=450
)
fig.update_traces(
    line=dict(color="grey"),
    marker=dict(symbol="x", color="grey")
)

# Plot a red line at intersection(s)
coh = df.set_index("hdbscan_cluster_size")["coherence_normalized"]
div = df.set_index("hdbscan_cluster_size")["diversity_normalized"]

df2 = pd.DataFrame({"coh": coh, "div": div}).sort_index()
df2["diff"] = df2["coh"] - df2["div"]

x_vals = df2.index.values
diff = df2["diff"].values

# --- Find sign changes ---
sign_changes = np.where(np.sign(diff[:-1]) * np.sign(diff[1:]) < 0)[0]

intersection_exists = len(sign_changes) > 0

if intersection_exists:
    i = sign_changes[0]      # first crossing segment

    x0, x1 = x_vals[i], x_vals[i+1]
    y0, y1 = diff[i], diff[i+1]

    # Linear interpolation of intersection x*
    x_inter = x0 - y0 * (x1 - x0) / (y1 - y0)

else:
    # No crossing → find the minimal absolute difference
    i = np.argmin(np.abs(diff))
    x_inter = float(x_vals[i])      # just use the closest point

# Add vertical line at the intersection
fig.add_vline(
    x=x_inter,
    line_width=2,
    line_color="red",
    line_dash="dot"
)

#### COPILOT  https://m365.cloud.microsoft/chat/entity1-d870f6cd-4aa5-4d42-9626-ab690c041429/eyJpZCI6IlZYTmxjbFl4ZkdoMGRIQnpPaTh2YzNWaWMzUnlZWFJsTFdsdWRDNXZabVpwWTJVdVkyOXRMM3hQU1VRNll6QmpZbUZrWkRJdFpEZGhaaTAwT0dVM0xXRXdZamt0WVRSbE5XSTJNR1UxTmpnMGZEY3lZV00zTkdGbUxXVmxNRGt0TkRJd01DMDRZelUwTFRWbE5qQTVOV0UxTlRZd1lYd3lNREkxTFRFeUxURXlWREUzT2pRek9qUTNMakF3T1RNMU5UUmEiLCJzY2VuYXJpbyI6InNoYXJlTGlua1ZpYVBvcG92ZXIiLCJwcm9wZXJ0aWVzIjp7InByb21wdFNvdXJjZSI6InVzZXIiLCJjbGlja1RpbWVzdGFtcCI6IjIwMjUtMTItMTJUMTc6NDM6NDcuODYzWiJ9LCJjaGF0VHlwZSI6IndlYiIsInZlcnNpb24iOjEuMX0
# --- Add a red label at that position without touching ticks ---
# Keep default ticks; overlay a red annotation aligned to the x-axis.
fig.add_annotation(
    x=x_inter,
    y=0,                 # position on the x-axis
    xref="x",
    yref="paper",
    text=f"{x_inter:.2f}",
    showarrow=False,
    yanchor="top",       # stick to the axis line
    yshift=-10,          # nudge below the axis (adjust if it's clipped)
    font=dict(color="red", size=12),
    align="center"
)

# (Optional) add a small red tick mark on the axis, without changing axis ticks
# Comment out if you only want the text label.
fig.add_shape(
    type="line",
    x0=x_inter, x1=x_inter,
    y0=0.0, y1=0.03,     # small stub above the axis (paper coords)
    xref="x",
       yref="paper",
    line=dict(color="red", width=2)
)

fig.show()
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
We manually examine topic models around that intersection line and find a satisfying one, which we present in the next section. In addition, we will compare it to a model with a lower coherence, but fewer topics, which offers a more synthetic view of the corpus. The two selected models are :
-  `min_cluster_size` $= 48$, close to what we measure as "optimal" value, with $K = 26$ topics which we used for corpus exploration
-  `min_cluster_size` $= 78$, with $K = 17$ topics, better for corpus summarization

In [section 3.2](#32-improve-the-study-of-the-audiovisual-mediatisation-of-the-march) we present a topic analysis drawing from both models, which we will hereby refer to as `V48` (smaller clusters, more topics) and `V78` (larger clusters, fewer topics).
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## 2.3 Topic visualizations
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
In this section, we will plot topic representations derived by extracting characteristic terms from clusters, following a series of pre-processing steps.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
- We enrich the vocabulary with frequent bi- and tri-grams (e.g. the trigram _« Marche-des-beurs »_ is treated as a single term, alongside the three individual tokens _“marche”_, _“des”_, _“beurs”_). 
- French stop-words are removed, as well as terms appearing in more than 50% or fewer than 5% of the passages, in order to eliminate uninformative or overly specific words from cluster representations. 
- N-grams are consolidated into their longest form (e.g. _"jamel"_, _"debbouze"_ are excluded since the bigram _“jamel debbouze”_ is retained in the vocabulary). 
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
After these steps, we identify the 25 most distinctive terms for each cluster, ranked using the C-TF-IDF metric proposed in BERTopic. This metric adapts the classic TF-IDF principle to a class-based setting: instead of computing term importance across individual documents, it evaluates a term’s relevance within an entire cluster compared to all other clusters. In other words, C-TF-IDF highlights words that are not only frequent in a given cluster but also discriminative across the corpus, making them strong indicators of that cluster’s topic. In the next section, we present visualizations, which we interpret in section 3.1.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
We reload both BERTopic models (`V48` and `V78`) at this step. 
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import math
p0 = 6
p1 = 145
p2 = 48
p3 = math.ceil(0.15*p2)

param_string = f'umap_{p0}-{p1}_hdbscan_cuml{p2}-{p3}'

base_path = f'./script/data/TM/{get_chunks.__name__}/{short_model_name}'
model_48 = BERTopic.load(f'{base_path}/{param_string}_bertopic_model.pkl')
```

```python
p0 = 64
p1 = 145
p2 = 78
p3 = math.ceil(0.15*p2)
param_string = f'umap_{p0}-{p1}_hdbscan_cuml{p2}-{p3}'
base_path = f'./script/data/TM/{get_chunks.__name__}/{short_model_name}'
model_78 = BERTopic.load(f'{base_path}/{param_string}_bertopic_model.pkl')
```

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
We need to re-compute the topic representations when loading a pickled bertopic model. In addition, `topic_per_class` needs to be computed before topic representations, which we use later for dynamic topic modelling.
<!-- #endregion -->

```python
# topic_per_class needs to be computed before custom representations. we will use it later (class = years)
docs = chunk_df['chunk'].to_list()
years=chunk_df.publication_date.astype(str).str[:4].astype(int)
topics_per_class_48 = model_48.topics_per_class(docs, years)
topics_per_class_78 = model_78.topics_per_class(docs, years)
```

```python
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, OpenAI, PartOfSpeech
# KeyBERT
keybert_model = KeyBERTInspired()

# Part-of-Speech
pos_model = PartOfSpeech("fr_core_news_lg")

# MMR
mmr_model = MaximalMarginalRelevance(diversity=0.3)


vectorizer_model = CountVectorizer(stop_words=list(fr_stop), min_df=0.05, max_df=0.5, ngram_range=(1, 3), tokenizer=LemmaTokenizer())


# All representation models
representation_model = {
"KeyBERT": keybert_model,
"MMR": mmr_model,
"POS": pos_model
}
docs = chunk_df['chunk'].to_list()

model_48.update_topics(docs, vectorizer_model = vectorizer_model, representation_model = representation_model, top_n_words=100)
model_78.update_topics(docs, vectorizer_model = vectorizer_model, representation_model = representation_model, top_n_words=100)
```

```python
from bertopic.representation._base import BaseRepresentation
from typing import List, Tuple, Mapping

class StrictLongestFormNgramRepresentation(BaseRepresentation):
    def __init__(self, top_n_words: int = 25):
        self.top_n_words = top_n_words

    def extract_topics(
        self,
        topic_model,
        documents,
        c_tf_idf,
        topics: Mapping[int, List[Tuple[str, float]]]
    ) -> Mapping[int, List[Tuple[str, float]]]:
        new_topics = {}
        for topic, words in topics.items():
            filtered = self._filter_words(words)
            new_topics[topic] = filtered
        return new_topics

    def _crude_root(self, word: str) -> str:
        """Basic root detector: lowercase, strip s', and remove final 's'."""
        word = word.lower().strip()
        if word.startswith("s'"):
            word = word[2:]
        if word.endswith("s") and len(word) > 3:
            word = word[:-1]
        return word

    def _filter_words(self, words: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        word_roots = {w: [self._crude_root(token) for token in w.split()] for w, _ in words}
        kept_words = []

        for word, score in words:
            roots = word_roots[word]
            n = len(roots)

            # Keep only if not fully contained in any longer n-gram's roots
            contained_in_longer = any(
                self._is_sublist(roots, other_roots)
                for other, other_roots in word_roots.items()
                if other != word and len(other_roots) > n
            )
            if contained_in_longer:
                continue

            kept_words.append((word, score))
        if len(kept_words) == 0 :
            print('no words left')
        return kept_words[:self.top_n_words]

    def _is_sublist(self, small: List[str], big: List[str]) -> bool:
        """Check if all elements of `small` appear in order in `big` (as a contiguous sublist)."""
        for i in range(len(big) - len(small) + 1):
            if big[i:i + len(small)] == small:
                return True
        return False


custom_repr = StrictLongestFormNgramRepresentation(top_n_words=25)
model_48.update_topics(
    docs,
    vectorizer_model = vectorizer_model,
    representation_model = custom_repr,
    top_n_words=100  # must be higher than 25 to allow filtering
)
model_78.update_topics(
   docs,
   vectorizer_model = vectorizer_model,
    representation_model = custom_repr,
    top_n_words=100  # must be higher than 25 to allow filtering
)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
#### 2.3.1 Topic representative words
The barcharts below describe the top 10 words for each topic, ranked by C-TF-IDF (specificity) score.
<!-- #endregion -->

```python editable=true jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} slideshow={"slide_type": ""} tags=["figure-topicbarchart78HTML-*"]
p2=78
barchart_78 = model_78.visualize_barchart(title=f'Salient words per topic <br>'+
                                            f'<span style="font-size: 12pt;">V78 / {len(model_78.get_topics().keys())} topics </span>',
                                            topics=[t for t in model_78.get_topics().keys()],
                                              n_words=10,
                                              autoscale=True)
barchart_78.update_layout(margin_t=150)

barchart_78.write_html(f'./figs/topics_barchart_V{p2}.html', include_plotlyjs="cdn")
barchart_78.write_image(f'./figs/topics_barchart_V{p2}.png')
barchart_78.show()
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicbarchart78PNG-*"]
# Fallback to the png version in case HTML fails to display
display(Image('./figs/topics_barchart_V78.png'))
```

<!-- #region tags=["hermeneutics"] -->
#### On soft clustering and the “outlier topic”
Note the first topic `#-1` represents the top words from passages that remained unassigned by HDBSCAN. Out of approximately 14,500 passages, about 10,000 belong to this generic "non-cluster", while only 4,500 are assigned to actual clusters/topics. 

Unlike a common approach, we do not interpret the assignment of a large number of documents to this “generic” topic as evidence of poor model quality. Instead, we hypothesize that this high proportion of generic passages is consistent with a corpus that was originally selected based on a theme already highly specific within the French language. In other words, contrary to the “outlier topic” label used by the library’s author, topic `#-1` does not represent corpus anomalies in our case, but rather the common discourse shared across the entire corpus.
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
Some irrelevant themes are indeed assigned a cluster, as evidenced by topic `#20` in the model below (advertisements appearing in some of the material).
<!-- #endregion -->

```python editable=true jdh={"object": {"source": ["Topics and top 10 words for each of the 28 topics in Bertopic model V48, ranked by C-TF-IDF"]}} slideshow={"slide_type": ""} tags=["figure-topicbarchart48HTML-*"]
p2=48

barchart_48 = model_48.visualize_barchart(title=f'Salient words per topic <br>'+
                                            f'<span style="font-size: 12pt;">V48 / {len(model_48.get_topics().keys())} topics </span>',
                                            topics=[t for t in model_48.get_topics().keys()],
                                              n_words=10,
                                              autoscale=True)
barchart_48.update_layout(margin_t=150)

barchart_48.write_html(f'./figs/topics_barchart_V{p2}.html', include_plotlyjs="cdn")
barchart_48.write_image(f'./figs/topics_barchart_V{p2}.png', include_plotlyjs="cdn")
barchart_48.show()
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicbarchart48PNG-*"]
# Fallback to the png version in case HTML fails to display
display(Image('./figs/topics_barchart_V48.png'))
```

### 2.3.2 Cluster map

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hermeneutics"] -->
BERTopic's visualization tools allows to perform 2D reduction of embeddings, so as to map the clusters onto the semantic space.
<!-- #endregion -->

<!-- #region tags=["hermeneutics"] -->
We will pass a custom list of `docs` to the `visualize_documents` function. These `docs` will include html-formatted metadata (title, publication date) so it displays in the hover tooltip.
<!-- #endregion -->

```python
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

docs_tooltip =[]
for _, row in chunk_df.iterrows() : # one row per chunk
  row = row.fillna('')
  doc_mots_liste=[]
  mots = row.chunk.replace('\n', ' ').split() 
  for chunk in chunks(mots, 12) :
    doc_mots_liste.append(' '.join(chunk))
    this_tooltip = '<br>'.join(
    [
    '<b>'+ row.channel + ' - ' +  row.title + ' - ' + pd.to_datetime(row.publication_date).strftime('%d/%m/%Y') + '</b>',
    ] + 
    doc_mots_liste
    )
  docs_tooltip.append(this_tooltip)

```

<!-- #region tags=["hermeneutics"] -->
Optionnaly reload and reduce the embeddings.
<!-- #endregion -->

```python
import pickle
with open("./script/data/TM/chunk_lgc_token/dangvantuan_sentence-camembert-large_embeddings.pkl", 'rb') as infile :
    embeddings = pickle.load(infile)
```

```python
print('re-running umap reduction model...')
reduced_embeddings = UMAP(n_neighbors=p0, n_components=2, min_dist=0.0, metric='cosine', random_state=42).fit_transform(embeddings)
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicsmap78HTML-*"]
from script.utils import get_fig_safe

# generate map for model 48
title=("Cluster map<br>" f'<span style="font-size: 12pt;">V48 / {len(model_48.get_topics().keys())} topics </span>')
map48 = model_48.visualize_documents(title=title, docs=docs_tooltip, topics=[t for t in model_48.get_topics().keys()], reduced_embeddings=reduced_embeddings)
map48.write_image(f'./figs/topics_map_V48.png')
safe48 = get_fig_safe(map48)
safe48.write_html(f'./figs/topics_map_V48.html')

# generate map for model 78
title=("Cluster map<br>" f'<span style="font-size: 12pt;">V78 / {len(model_78.get_topics().keys())} topics </span>')
map78 = model_78.visualize_documents(title=title, docs=docs_tooltip, topics=[t for t in model_78.get_topics().keys()], reduced_embeddings=reduced_embeddings)
safe78 = get_fig_safe(map78)
safe78.write_html(f'./figs/topics_map_V78.html')

# generate a light version of the map
map78_light = model_78.visualize_documents(title=title, docs=docs_tooltip, topics=[t for t in model_78.get_topics().keys()], sample=0.1, reduced_embeddings=reduced_embeddings)

map78_light.show()
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicsmap78PNG-*"]
# Fallback to PNG in case the HTML fails to display
Image('./figs/topics_map_V78.png')
```

To reduce the visualization footprint and facilitate model comparison, the code below generates a single interactive figure, where the user can switch between the two models.

```python
from script.utils import merge_bertopic_visualizations

merged_map = merge_bertopic_visualizations(
    map78,
    map48,
    label1="19 topics",
    label2=f"28 topics",
    legend_char_limit=27 # manually tune this parameter until the fig stops resizing when toggled
)

safe_merged = get_fig_safe(merged_map)
safe_merged.write_html('./figs/topics_map_merged.html')

# Uncomment below to show directly in the notebook
safe_merged.show() 
```

### 2.3.3 Dynamic topic modeling



Finally, we examine the evolution of the topic's underlying themes over time. For each year, the following plot aims to visualize and compare the proportion of passages assigned to the selected topics.


Once again, we slightly modify a native BERTopic visualization function, `topics_per_class`, and define classes as yearly bins in our data. 

```python editable=true slideshow={"slide_type": ""}
import numpy as np
import plotly.graph_objects as go

# Plot the topics for V48
topics_per_class = topics_per_class_48

title=("Topic proportion per year<br>"
+ f'<span style="font-size: 12pt;">(Percentage of documents assigned to topic T for year Y)</span><br>'
+ f'<span style="font-size: 12pt;">V48 / {len(model_48.get_topics().keys())} topics </span>')
fig_48 = model_48.visualize_topics_per_class(topics_per_class.sort_values('Class'), title=title, topics=range(-1, len(model_48.get_topics())-1), top_n_topics=35)

# Get the categories (y-axis for horizontal bars) -- these are the known x-ticks
categories = fig_48.data[0].y

# Initialize a dictionary to store x values for each trace, with 0 for missing x-ticks
trace_values = {}

# Iterate through each trace and align x values with known categories
for trace in fig_48.data:
    if isinstance(trace, go.Bar):
        # Create a dictionary of categories with 0 values for each trace
        trace_dict = dict(zip(trace.y, trace.x))  # Map categories to x-values
        # For categories that are not in the trace, set the value to 0
        aligned_values = [trace_dict.get(cat, 0) for cat in categories]
        trace_values[trace.name] = aligned_values

# Convert trace values to a numpy array for normalization
value_matrix = np.array(list(trace_values.values()))

# Sum across traces for each category (column-wise sum)
totals = value_matrix.sum(axis=0)

# Normalize each trace's values so they sum to 100 for each x-tick
for i, (trace, trace_name) in enumerate(zip(fig_48.data, trace_values.keys())):
    if isinstance(trace, go.Bar):
        # Normalize the trace values
        trace.x, trace.y = categories, (value_matrix[i] / totals) * 100
        trace.orientation = 'v'  # Set orientation to vertical

# Update layout to stack the bars                  
fig_48.update_layout(dict(yaxis=dict(automargin=True)))
fig_48.update_xaxes(dtick='Y1', tickangle=45)

# Add generous right margin for hoverlabels
fig_48.update_layout(
    margin=dict(t=120, l=80, r=320, b=60),  # r increased significantly
    xaxis_title="Year",
    yaxis_title="Topic proportion",
)
fig_48.write_html('./figs/topics_dynamic_V48.html')
fig_48.write_image('./figs/topics_dynamic_V48.png')
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicsdyn78HTML-*"]
import numpy as np
import plotly.graph_objects as go

# Plot the topics for V78
topics_per_class = topics_per_class_78

title=("Topic proportion per year<br>"
+ f'<span style="font-size: 12pt;">(Percentage of documents assigned to topic T for year Y)</span><br>'
+ f'<span style="font-size: 12pt;">V78 / {len(model_78.get_topics().keys())} topics </span>')
fig_78 = model_78.visualize_topics_per_class(topics_per_class.sort_values('Class'), title=title, topics=range(-1, len(model_78.get_topics())-1), top_n_topics=35)

# Get the categories (y-axis for horizontal bars) -- these are the known x-ticks
categories = fig_78.data[0].y

# Initialize a dictionary to store x values for each trace, with 0 for missing x-ticks
trace_values = {}

# Iterate through each trace and align x values with known categories
for trace in fig_78.data:
    if isinstance(trace, go.Bar):
        # Create a dictionary of categories with 0 values for each trace
        trace_dict = dict(zip(trace.y, trace.x))  # Map categories to x-values
        # For categories that are not in the trace, set the value to 0
        aligned_values = [trace_dict.get(cat, 0) for cat in categories]
        trace_values[trace.name] = aligned_values

# Convert trace values to a numpy array for normalization
value_matrix = np.array(list(trace_values.values()))

# Sum across traces for each category (column-wise sum)
totals = value_matrix.sum(axis=0)

# Normalize each trace's values so they sum to 100 for each x-tick
for i, (trace, trace_name) in enumerate(zip(fig_78.data, trace_values.keys())):
    if isinstance(trace, go.Bar):
        # Normalize the trace values
        trace.x, trace.y = categories, (value_matrix[i] / totals) * 100
        trace.orientation = 'v'  # Set orientation to vertical

# Update layout to stack the bars                  
fig_78.update_layout(dict(yaxis=dict(automargin=True)))
fig_78.update_xaxes(dtick='Y1', tickangle=45)

# Add generous right margin for hoverlabels
fig_78.update_layout(
    margin=dict(t=120, l=80, r=320, b=60),  # r increased significantly
    xaxis_title="Year",
    yaxis_title="Topic proportion",
)
fig_78.write_html('./figs/topics_dynamic_V78.html')
fig_78.write_image('./figs/topics_dynamic_V78.png')
fig_78.show()
```

```python jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} tags=["figure-topicsdyn78PNG-*"]
# Fallback to PNG in case HTML fails to display
display(Image('./figs/topics_dynamic_V78.png'))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
To facilitate a back‑and‑forth between distant reading and the source material, and to address what we see as a shortcoming of BERTopic’s native visualizations, we provide code to adapt BERTopic figures so as to display representative documents in a table below the chart. Because this code might fail to display in some environments, the improved version is visible at [this url](https://inalelab.github.io/marche83/figs/dynamic_explorer.html){:target="_blank"}.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import pandas as pd
import re
import plotly.graph_objects as go
import ipywidgets as widgets
from IPython.display import display, HTML
import traceback
import time

# STEP 1: Prepare document info with metadata

# Get BERTopic's document info aligned to your docs
doc_info = model_48.get_document_info(docs)

# Attach year from your precomputed années (same order as docs)
doc_info['year'] = years.values

# Merge in your metadata from chunk_df
chunk_df['chunk_url'] = chunk_df['url'] + "/?type=stream&redirect=1" + "&additional_params=start=" + chunk_df['chunk_start_approx'].astype(int).astype(str)
from lab_pytools.asr import sec_to_tc
chunk_df['chunk_tc'] = chunk_df.chunk_start_approx.apply(sec_to_tc)
meta_cols = ['title', 'url','channel','chunk_url', "chunk_tc", 'publication_date']  # adjust to your actual columns
doc_info = doc_info.join(chunk_df.reset_index()[meta_cols])

# Helper to pick top-k representative docs
def pick_representative(group: pd.DataFrame, k=5):
    # Prefer probability if available
    if 'Probability' in group.columns and not group['Probability'].isna().all():
        return group.sort_values('Probability', ascending=False).head(k)
    else:
        return group.head(k)

# Build (topic_id, year) -> list of dicts with selected metadata
repr_map = {}
for (t, y), g in doc_info.groupby(['Topic', 'year']):
    reps = pick_representative(g, k=5)
    # Convert to list of dicts for easier access
    repr_map[(t, y)] = reps[[
        'Document',
        'Topic',
        'Probability',
        'title',
        'url',
        'channel',
        'chunk_url',
        'chunk_tc',
        'publication_date'
    ]].to_dict('records')


# STEP 2: Convert figure to FigureWidget and attach metadata

# Convert to FigureWidget for interactive callbacks
fw = go.FigureWidget(fig_48)

fw.update_layout(
       margin=dict(t=120, l=80, r=320, b=60),  # same generous right margin
    xaxis_title="Year",
    yaxis_title="Topic proportion"
)
# Extract categories (years) from the first trace
categories = fw.data[0].x if hasattr(fw.data[0], 'x') and fw.data[0].x is not None else fw.data[0].y
# print(f"Categories (years): {categories}")

# Helper to extract numeric topic id from trace name
def extract_topic_id(name: str):
    m = re.search(r'-?\d+', str(name))
    return int(m.group()) if m else None

# Attach metadata and customdata to each bar trace
for trace in fw.data:
    if isinstance(trace, go.Bar):
        t_id = extract_topic_id(trace.name)
        trace.meta = {'topic_id': t_id}
        
        # Build customdata: one entry per bar (year)
        payload = []
        for year in categories:
            # Convert year to int, handling both string and numeric types
            try:
                year_int = int(year)
            except (ValueError, TypeError):
                year_int = year
            
            docs_list = repr_map.get((t_id, year_int), [])
            payload.append(docs_list)
        
        trace.customdata = payload

# STEP 3: Create output widget and click handler (fixed)

out = widgets.Output()

_last_click = {'topic': None, 'year': None, 'time': 0.0}
_is_rendering = False  # Lock to prevent concurrent rendering

def render_docs(docs_list, topic_id, year):
    """Render representative documents as an HTML table."""
    if not docs_list:
        display(HTML(f"<b>Year {year} — Topic {topic_id}:</b> Aucun document représentatif trouvé."))
        return
    
    rows = []
    for d in docs_list:
        title = d.get('title', '(Sans titre)')
        url = d.get('chunk_url')
        pub = d.get('publication_date', '')
        prob = d.get('Probability')
        doc = d.get('Document')
        tc = d.get('chunk_tc')
        channel = d.get('channel')
        
        # Make title clickable if URL exists
        if url:
            title_html = f'<a href="{url}" target="_blank">{title}</a>'
        else:
            title_html = title
        
        prob_str = f"{prob:.4f}" if prob is not None and not pd.isna(prob) else ''

        rows.append(f"""
        <tr>
          <td style="padding:4px; border-bottom:1px solid #ddd">{doc}</td>
          <td style="padding:4px; border-bottom:1px solid #ddd">{title_html}</td>
          <td style="padding:4px; border-bottom:1px solid #ddd">{channel}</td>
          <td style="padding:4px; border-bottom:1px solid #ddd">{tc}</td>
          <td style="padding:4px; border-bottom:1px solid #ddd">{pub}</td>
        </tr>
        """)

        #   <td style="padding:4px; border-bottom:1px solid #ddd">{prob_str}</td> 
    
        table_html = f"""
        <div style="margin-top:10px; font-family:sans-serif">
        <b style="font-size:14px">Year {year} — Topic {topic_id}: representative documents</b>
        <div style="max-height:380px; overflow:auto; border:1px solid #ddd; margin-top:6px">
            <table style="border-collapse:collapse; width:100%; font-size:13px; line-height:1.25">
            <thead style="position:sticky; top:0; background:#f7f7f7; z-index:1">

        <thead>
          <tr style="background-color:#f0f0f0">
            <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Passage</th>
            <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Title</th>
            <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Channel</th>
            <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Start (approx)</th>
            <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Publication date</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
            # <th style="text-align:left; padding:6px; border-bottom:2px solid #333">Probability</th> # add this above if we want the probability
        
    display(HTML(table_html))


def handle_click(trace, points, state):
    """Handle bar click events."""
    global _is_rendering, _last_click

    # Ignore if this trace wasn't actually clicked
    if not points.point_inds:
        return

    # Re-entrancy guard
    if _is_rendering:
        return

    # Which bar?
    idx = points.point_inds[0]

    # Determine year from orientation/x/y
    orientation = getattr(trace, "orientation", "v")
    if orientation == "v":
        year = trace.x[idx] if trace.x is not None else None
    else:
        year = trace.y[idx] if trace.y is not None else None

    # Topic ID from meta
    t_id = (getattr(trace, "meta", None) or {}).get("topic_id")

    # Debounce duplicates
    now = time.time()
    if (_last_click['topic'] == t_id and
        _last_click['year'] == year and
        now - _last_click['time'] < 0.2):
        return

    # Update last-click info
    _last_click['topic'] = t_id
    _last_click['year'] = year
    _last_click['time'] = now

    # LOCK
    _is_rendering = True
    try:
        # Resolve docs_list from customdata
        docs_list = None
        if hasattr(trace, "customdata") and trace.customdata is not None:
            if idx < len(trace.customdata):
                docs_list = trace.customdata[idx]

        with out:
            out.clear_output(wait=True)

            if not docs_list:
                # Still render a friendly message so users know the click worked
                display(HTML(f"<b>Year {year} — Topic {t_id}:</b> Aucun document représentatif trouvé."))
                return

            render_docs(docs_list, t_id, year)

    except Exception:
        # Make sure errors are visible and don't kill the lock release
        with out:
            print("Exception in handle_click:")
            print(traceback.format_exc())
    finally:
        # ALWAYS release the lock so subsequent clicks work
        _is_rendering = False


# Attach the callback to all bar traces (unchanged)
callback_count = 0
for tr in fw.data:
    if isinstance(tr, go.Bar):
        tr.on_click(handle_click)
        callback_count += 1

# display(fw)
# display(out)
print("\n✓ Interactive figure ready. Click on any bar to see representative documents below.")

```

## Part 3 - Findings, Limitations, and Epistemological Reflections on AI Integration in Historical Research



### 3.1. Results and limitations 

<!-- #region citation-manager={"citations": {"ze50b": [{"id": "10163145/HVXDTLZJ", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
One of the principal outcomes of this work is the development of a reproducible, AI‑assisted method for exploring historical corpora. 
From the perspective of computational information and archival sciences, this work illustrates the potential of AI technologies applied to automatic speech recognition (ASR) for the historical study of French audiovisual heritage. Using a medium‑sized corpus—conceived as a testbed yet capable of informing analyses of large‑scale collections—we proposed a methodological adaptation of tools for textual synthesis, in particular BERTopic, traditionally used for the ‘distant reading’ of written corpora. This adaptation was driven by the historian’s fundamental questions: What are the sources under analysis? What do we know about their authors? Is each item an original, a copy, or a version republished or re‑editorialised online? In this sense, our approach contributes to reflections on the socio‑technical construction of historical facts (<cite id="ze50b"><a href="#zotero%7C10163145%2FHVXDTLZJ">(Ben-David and Amram 2018)</a></cite>) and paves the way for renewed practices in the thematic and critical exploration of audiovisual archives.  
From the historian’s standpoint, the researcher benefits from a transmedia audiovisual exploration environment that enables the combination of distant reading with a semio‑discursive approach, as close to the archive as possible. Specifically, HTML‑based visualisations make it possible to identify semantic proximities (bar chart, map, hierarchy), as well as the temporal distribution of themes—both in absolute terms and as proportions relative to the number of documents per year. The map then allows retrieval of verbatim quotations and references from the transcribed video. Each item can be located precisely in the spreadsheet, which also contains the full transcription text. Ultimately, the researcher can consult the original video or web page in the archives of the Institut national de l’audiovisuel (INA). This transforms how audiovisual archives are searched and systematically reviewed, without confining the research to majority trends surfaced by AI, since the original archive can always be viewed. The tools therefore fully satisfy disciplinary imperatives aimed at interpreting ruptures and continuities in media coverage, while simultaneously engaging in source (documentary) criticism. At the same time, collaboration and dialogue with the data scientist leads to a better grasp of data and AI biases. Together, we selected the level of granularity and are able to account for it transparently. This is also an open‑science approach, which is crucial at a time when the reliability of science is being questioned.

<!-- #endregion -->

### 3.2. Improve the study of the audiovisual mediatisation of the March 

<!-- #region editable=true slideshow={"slide_type": ""} -->
This methodology sheds light on the audiovisual mediatisation of the March on television and on the web since 1983. 
Automated thematisation (topic modelling) reveals the interpretive polyvocality of the March, and making the outputs available at different levels of granularity enables a more fine‑grained analysis (`V48` & `V78`). We observe continuities in how the March is characterised across the entire period: the description of the event and its historical context is a constant running through the corpus (topic `#0` in `V78`; topics `#0` and `#3` in `V48`). This encompasses social conditions (also topic `#8` in `V48`), demands and the gains associated with the March—notably the award of the ten‑year long‑term residence permit (topic `#15` in `V48` and `V78`)—as well as universal models and aspirations (topic `#23` in `V48`) and the anchoring of the March within the history of anti‑racism (topic `#7` in `V78`). Within this descriptive frame, immigration and the specifically Algerian dimension are recurring themes. The corpus is also marked by questions of memory and transmission of the March through commemorations and cultural productions. We note the strong presence of the ‘Muslim’ theme (topic `#2` in `V48` and `V78`). Finally, with regard to references to the past, colonisation in Algeria is explicitly mentioned (former colony; `"indigène"` in topic `#5` in `V48`), as are decolonisation and the immigration of Maghrebi workers—the ‘parents’ of the marchers (topic `#5` in `V48`). However, Martin Luther King’s 1963 March (topic `#4`, dedicated to the genealogy of the March), taken as a model by the marchers of 1983, eclipses the victory in 1962 that gave birth to Algeria.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
While it is unsurprising to see periodic historical reminders of an event receding over the decades, our analysis makes visible the increasing visibility of the marchers, thereby giving greater space to their demands and to the references that were theirs in 1983. Notably, Toumi Djaïdja is explicitly identifiable in topic `#2`,`V48`, and his presence (mention by name or direct testimony) in the videos is associated with the incident of police violence he suffered in 1983, which led him to conceive the March. Ultimately, the thematisation shows that, despite the marchers’ genuine visibility, the reach of the March is constrained within the governance of Islam and the integration of Muslims in France (topic `#2`: `"Islam"`, `"laïcité"`), and by the threat that the far right poses to anti‑racism (topic `#−1`, `"Front National"`).

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
To examine how thematic coverage evolves over time, we focus here on two salient findings: on the one hand, a clearer visibility of the marchers and their demands; on the other, forty years of denunciations of systemic discrimination and, more specifically, the racialisation of police violence—one of the triggering factors of the March in 1983.

<!-- #endregion -->

### a. Rising prominence of the marchers and their demands

<!-- #region citation-manager={"citations": {"2sfvq": [{"id": "10163145/ZHQWCEM3", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Text analysis captures the growing mediatisation of the marchers—most notably Toumi Djaïdja—where visual analysis had failed. His name appears in the description of topic `#0` (V48), and consultation of the transcripts shows that references to him are systematically enriched with narrative details. We also find Farid Lahoua and Christian Delorme. Chronological navigation helps to clarify the mediatisation of T. Djaïdja: he is cited in 1983 as the instigator, after being the victim of a police shooting; in 1984, however, his name is mentioned not to celebrate him but to report his conviction to fifteen months’ imprisonment for robbery with violence. According to (<cite id="2sfvq"><a href="#zotero%7C10163145%2FZHQWCEM3">(Hajjat 2013)</a></cite>), this conviction resulted from police harassment and gave rise to multiple protests, culminating in a presidential pardon granted by François Mitterrand in December 1984. Thereafter his name fades. 

It is briefly mentioned in 1993 in ambiguous terms: ‘Like many others, Toumi Djaïdja, their leader, who today shies away from the camera, has found in Islam a reason to live’ (1993‑12‑02, CAA93078428). His name is not mentioned at the twentieth anniversary in 2003, but it returns in a significant way only from 2009, thereafter becoming firmly established as that of a principal instigator of the March and an indispensable witness to its mediatisation.

<!-- #endregion -->

This renewed recognition is accompanied by an improved articulation of the marchers’ demands—including the denunciation of the term “beur”—and by better contextualisation of the event, which has been the subject of several historiographical studies since 2012. The demands are recalled and stated in 1983 and 1984 in the television archives; they resurface at the ten‑ and twenty‑year commemorations; but it is chiefly with the re‑examination of the March in 2009 that they become information that thereafter almost systematically accompanies media coverage of the event, with a peak in 2013, of course, but also in 2018 and 2024 (topic `#15`, `V78`, by proportion). This translates into a return to the initial demands, notably justice for the families of victims of racist crimes and the call for a ten‑year long‑term residence permit.



This movement is accompanied by a reappraisal of the role of SOS Racisme as an actor in—or emanation of—the 1983 March, originating in SOS Avenir Minguettes, led by Toumi Djaïdja. Contested as early as 1984, the association’s filiation with the March is sporadically called into question from 2009, before becoming the dominant view in the archives from 2013 (topic `#7`, `V78`). The following passage from the transcript of a programme broadcast on the television channel Public Sénat and posted on Dailymotion illustrates the rejection of the movement’s co‑optation: _‘SOS Racisme is, after all, a way—with the slogan touche pas à mon pote—of infantilising those who march. It’s saying: we will stand in front; don’t move, lads, we’ll speak for you’_ (2013/1/02, x17wt18).



Alongside the greater prominence accorded to the marchers, the promotion of Nabil Ben Yadir’s film La Marche also contributed in 2013 to this re‑examination, seeking to return to the origins of the March against the grain of the media myth. However, the film keeps its distance (see the ‘residence‑permit’ cluster) from a powerful driver of the March’s mediatisation: the denunciation of racist crimes and police violence against children and young men racialised as ‘black’ or ‘Maghrebi’ from working‑class neighbourhoods (map, e.g., far from theme 13).



### b. Police violence and the question of institutional racism



_“Why are we being shot like rabbits? (…) The March was born from the tears of mothers who lost their children,”_ recounts a witness when recalling his participation in the 1983 March (excerpt from INA archive PAC03002355).


<!-- #region citation-manager={"citations": {"8zbfc": [{"id": "10163145/EWSSBAJP", "source": "zotero"}]}} -->
These children, adolescents and young adults who were killed lie at the heart of the March’s genesis: it was during his hospitalisation following a gunshot wound inflicted by the police that Toumi Djaïdja conceived the idea of crossing France to denounce these violences. An analysis that cross‑references the relevant topics and their temporal distribution offers a more fine‑grained view of the discursive modalities and of the factors invoked to explain the persistence of mechanisms leading to the killing of racialised young men from working‑class neighbourhoods—resulting from police shootings, but also from violence perpetrated by drug traffickers.
Whether caused by individual firearms or by police weapons, these tragedies embody the apex of social and territorial inequalities. Their mediatisation is accompanied by denunciations of police violence and, more broadly, of the effects of institutional racism, understood as practices or policies that generate discriminatory outcomes even in the absence of explicit intent (<cite id="8zbfc"><a href="#zotero%7C10163145%2FEWSSBAJP">(Wihtol de Wenden 2016)</a></cite>). 

<!-- #endregion -->

The analysis traces four decades of struggle against police violence, with the colonial legacy operating as a contributing factor to the persistence of systemic racism in society and, consequently, within the police. Racist violence is initially described as emanating from society at large, sustained by movements nostalgic for French Algeria and supported by the far right—recalling that a wave of racist crimes helped trigger the March in the 1980s, such as the fatal shooting of Toufik Ouanes on 9 July 1983 in La Courneuve (the racist motive was dismissed at trial), and, of course, the killing of Habib Grimzy on the Bordeaux–Ventimiglia train during the March in November 1983. Over the years—likely aided by the decline in racist crimes in the 1990s—the scope of references to the March increasingly centres on the persistence of police violence, more specifically shootings, targeting young men from working‑class neighbourhoods, notably those perceived as Maghrebi/Muslim (topic `#13`, `V78`).



The issue first came to the fore in 1984, with the presidential pardon granted to Toumi Djaïdja and during the Marche Convergence 84, when journalists filmed the accounts of bereaved mothers whose sons had been shot dead. Since the March, victims have organised as associations, denouncing the impunity of the perpetrators and the injustice of merely symbolic sentences. They also call for the recognition of specific guidelines and for their associations to be admitted as civil parties, while insisting on the need to alert public opinion to counter the banalisation of crimes rooted in ‘racist hatred’ against young people from working‑class neighbourhoods (VDC17001181, 1984‑11‑04, FR3).



Above all, however, the theme becomes embedded in the corpus from the 2010s, with a significant presence in 2013 and 2023, punctuated by a peak in 2020. Beyond the thirtieth anniversary of March, 2013 is marked by a lawsuit brought by thirteen French youths racialised as Black and Arab, who sued the State and the Interior Ministry for discrimination in abusive identity checks. This led to a historic hearing at the Paris Tribunal de grande instance on 3 July 2013, at a time when one of President François Hollande’s campaign pledges was to introduce a stop‑and‑search receipt to combat délit de faciès (racial profiling). The consulted archives contain no explicit mention of the July hearing, but they do emphasise the broken promise—a measure ‘postponed indefinitely’ (repoussée aux calendes grecques) (421874.047, La Chaîne Info, 2013‑10‑16).



The mobilisation of victims’ families contributed to the mediatisation of discriminatory police violence during Hollande’s term, exemplified by the march of 31 October 2015 launched by the sister of Amin Bentounsi, who died in 2012—one year before the commemorations of the 1983 March and the creation of the Black Lives Matter movement in the United States. However, the connection between Black Lives Matter and the March only materialised in the data in 2020, with the repercussions of George Floyd’s killing in the United States, which triggered the appearance of references to Black Lives Matter in the corpus, even though it originated in 2013. At that point, the systemic dimension was fully denounced by protesters gathering in June in Paris and several other French cities in homage to Adama Traoré (killed in 2016) and George Floyd, despite the restrictions imposed by Covid‑19.


<!-- #region editable=true slideshow={"slide_type": ""} -->
In 2023, the death of Nahel Marzouk, shot by a police officer on 27 June at the age of 17, resonated painfully with the fortieth anniversary of the March. Several former marchers saw it as symptomatic of the persistence of a specific form of violence by law‑enforcement towards young people perceived as Maghrebi. The discourse is bitter, describing the weakness of mechanisms to combat police violence since 1983 and Hollande’s unfulfilled promise. The systemic nature of these discriminations was not acknowledged by the government in 2023, and was vigorously contested by right‑wing and far‑right parties. Such justifications are frequently associated with the necessary fight against organised crime, fuelled by the pauperisation of working‑class neighbourhoods. Audiovisual archives dealing with the March also give voice to residents and to victims’ families, who deplore the growing surveillance of these areas, without addressing institutional discrimination or the protection of families. For instance, the corpus mentions the death of Fayed, a ten‑year‑old boy shot dead in Nîmes (Provence) by traffickers on 21 August 2023, which prompted a visit by Gérald Darmanin, then Interior Minister. 

A video from the online media Blast, devoted to Madani Marzuk—who has stood alongside Fayed’s family—denounces the minister’s on‑site visit and documents the unsanitary conditions and the institutions’ inability to guarantee residents’ safety. Madani Marzuk, a neighbourhood resident active in social work, meets Tarek Kawtari, the founder of the Mouvement Immigration Banlieue, whose words neatly capture the sense of abandonment and bitterness: _‘We were the flowers in the concrete. You see what I mean? Like the song says, you know? That’s what we were. But they massacred us, in fact. They shut the door on us. We knocked at the door of the Republic—just like the lyrics say. We knocked at the door of the Republic. It slammed shut on us, you see? We won battles, but they’re killing us with time.’_ (translated by author, video archive was available online in 2025, see below).

<!-- #endregion -->

```python editable=true jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} slideshow={"slide_type": ""} tags=["video-blast-*"]
# Load video from YT
from IPython.display import YouTubeVideo
# Extract video ID from URL
# URL: https://youtu.be/SQmOpyL7PMA?t=954
# Video ID: SQmOpyL7PMA
YouTubeVideo('SQmOpyL7PMA', start=954)
```

```python editable=true jdh={"module": "object", "object": {"source": ["LABEL TO ADD"]}} slideshow={"slide_type": ""} tags=["table-blast-*"]
chunk_df[chunk_df.id=='SQmOpyL7PMA'].reset_index().iloc[39:41]["chunk"].rename('Tarek Kawtari')
```

<!-- #region citation-manager={"citations": {"tmwee": [{"id": "10163145/KCYNM7TH", "source": "zotero"}]}} editable=true slideshow={"slide_type": ""} -->
Policing and surveillance of the neighbourhoods, in the name of combating organised crime yet without succeeding in protecting residents, serve here to heighten their vulnerability. Fayed’s death, initially attributed to drug trafficking, is thus analysed as the result of systemic discrimination that goes beyond individual acts to denote the indirect, structural effects of policies and practices (<cite id="tmwee"><a href="#zotero%7C10163145%2FKCYNM7TH">(Dhume 2016)</a></cite>).

<!-- #endregion -->

<!-- #region citation-manager={"citations": {"7sqhm": [{"id": "10163145/HGXGKTED", "source": "zotero"}]}} -->
Finally, the data analysis shows an improvement in the qualitative mediatisation of the event, which in turn makes visible the disappointments expressed by former marchers, who observe the persistence of systemic discrimination despite a decline in racist crimes over the past forty years. However, scholars specialising in the March, residents of working‑class neighbourhoods, anti‑racist activists and former marchers speaking on camera all point to the role of the French far right in disseminating anti‑Arab and anti‑Muslim discourse that renders their demands inaudible. Indeed, in a previous study of the March on web media, we observed how the March was systematically appropriated and drowned out by a mass of hostile discourse targeting Muslims - notably around the far-right editorialist Eric Zemmour, triggering chain reactions and a semantic storm that undermined the visibility of content relating to the March itself (<cite id="7sqhm"><a href="#zotero%7C10163145%2FHGXGKTED">(Rendina et al. 2024)</a></cite>). 
<!-- #endregion -->

### 3.3. Epistemological implications of integrating AI into historical research



This work on the March required substantial time for training and discussion concerning both television and digital‑media archives, as well as the appropriate use of computational methods. This dimension cannot simply be overlooked; in the tradition of the digital humanities, it deserves full consideration in order to reflect more effectively on the epistemological transformations underway and to recognise the value of exploratory approaches in history—a discipline in which scholarly contribution has traditionally rested on the novelty of interpretations, the themes and past events brought to light, or the recovery of previously unknown sources. Beyond the singularities of a corpus combining web archives and television archives, our exploration of the mediatisation of the 1983 March offers a case study through which to consider the implications of AI for the ways in which its use transforms the conditions under which historical knowledge is produced.


#### a. Historian in front of bias: Behind AI, the Data


It will particularly address the challenges posed by heterogeneous audiovisual corpora, highlighting the risks of cultural and linguistic bias—such as orthographic inconsistencies and the underrepresentation of minority voices in language models.

<!-- #region citation-manager={"citations": {"dxa27": [{"id": "10163145/CQ79HRL9", "source": "zotero"}]}} -->
We have seen how the aim of bringing marginalised bodies, faces, and voices back into view has quickly come up against the challenges posed by cultural bias. From this perspective, the use of AI multiplies the risks inherent to data-driven biases. In this "history through data" as put by (<cite id="dxa27"><a href="#zotero%7C10163145%2FCQ79HRL9">(Clavert 2016)</a></cite>), the corpus itself bears the marks of racial hegemony. Our case study, which concerns minoritised populations, fully illustrates the need to account for the power relations embedded within data—a requirement that resonates strongly with the methodological rigour of traditional historical source criticism. This serves to draw the attention of the historical research community to the importance of vigilance regarding bias. Indeed, while we have focused here on groups marginalised in the media sphere, one can easily imagine the harmful consequences that a similar approach might produce if socio-cultural biases were neglected in distant analyses of the media coverage of any contemporary event.

<!-- #endregion -->

These biases were quickly observed in textual data, from video metadata through to automatic transcriptions. The orthography of names of Arabic origin, such as Toumi Djaïdja, is heterogeneous in television archive metadata and on online video platforms. Although Whisper employs a multilingual AI model, its French training corpus contains few Arabic‑transliterated names, which results in transcription errors that reinforce biases already present in the archival data (e.g., Toumi, Tomi, Tony appear side by side). In a facial‑recognition approach, this training bias is also present, although one may hypothesise that the marchers are somewhat more visible on screen.


In response to these biases, two avenues can be pursued: intervening at the level of the AI model itself—either by adopting a different pre‑trained model with fewer undesirable biases, or by re‑training the model on corpus‑specific data; or de‑biasing the input data, here the texts, to improve the numerical representation of words (embeddings). In the present case, we primarily adopted the data‑cleaning approach. First, we developed linguistic rules to standardise orthography (e.g., Tony, Toumy, Tomi are normalised to Toumi). Secondly, we compiled a corrected dictionary prepared by student interns. This task is extremely time‑consuming but valuable. It invites us to invert the conventional view of AI as a time‑saving tool and, instead, to invest time in producing clean data, even if this requires dedicated resources. The dictionary can then be used to improve representations derived from pre‑trained models such as BERT and CamemBERT. It is also conceivable to fine-tune a small model on a list of proper names. This would follow a logic of domain adaptation: adapting the embedding model to the data drawn from the corpora assembled within the project.


Another contribution of our case study lies in the indispensable transparency of the approach adopted. This aspect is not new, even though it has become imperative with the widespread use of generative AI (GenAI). The digital‑humanities tradition advocates making methodological choices explicit, clarifying technical trade‑offs, acknowledging tool limitations, and detailing data biases. It underwrites evidentiary standards in digital history and is essential for understanding how historical knowledge is produced in the age of AI. Transparency is no longer merely a mark of good practice; here it functions as a critical instrument for uncovering power relations and data biases prior to their historical interpretation.


Moreover, methodological openness underpins the possibilities of reproducibility within an open‑science framework. In this instance, the data themselves are not shared—because they are subject to rights—but rather the entire methodological and technical protocol, articulated in a reflexive manner to encourage the circulation of know‑how. By clearly laying out the development process, we enable other researchers to adapt the methods to their own corpora, to test alternative technical options, and to identify persistent blind spots, thereby contributing to the collective refinement of computational approaches applied to born‑digital history.



#### b. Prospects for enhancing corpus analysis by leveraging advances in AI, including GenAI : Better embeddings (LLMs), Fine‑tuning, Generative capabilities


A representation better suited to topic modelling could be obtained from models trained on passage ranking or question answering tasks (see, e.g., [https://huggingface.co/antoinelouis/biencoder-camembert-base-mmarcoFR](https://huggingface.co/antoinelouis/biencoder-camembert-base-mmarcoFR){:target="_blank"}), or from large language models (LLMs).

<!-- #region citation-manager={"citations": {"1b0ol": [{"id": "10163145/447J7MJ6", "source": "zotero"}]}} tags=["hermeneutics"] -->
The SBERT model used to produce lexical embeddings of passages was trained on a large general‑purpose French corpus, then fine‑tuned on the Semantic Textual Similarity Benchmark  (STSB, <cite id="1b0ol"><a href="#zotero%7C10163145%2F447J7MJ6">(Cer et al. 2017)</a></cite>), in which the similarity between two sentences takes into account stance (polarity) and the actors mentioned as much as the topic itself. For instance, the sentence _“Iran’s president condemns use of chemical weapons in Syria”_ is assigned below‑average (0.3) similarity to _“Probe alleged use of chemical weapons in Syria”_ because a different actor is mentioned in the former, even though the underlying theme/event is the same.
<!-- #endregion -->

<!-- #region citation-manager={"citations": {"dvrkc": [{"id": "10163145/N7PEIRSL", "source": "zotero"}]}} -->
The intermediate layers of LLMs could be used to produce higher‑quality embeddings (<cite id="dvrkc"><a href="#zotero%7C10163145%2FN7PEIRSL">(BehnamGhader et al. 2024)</a></cite>). LLMs offer two advantages over SBERT: an extended context window, allowing embeddings of longer passages, and above all the potential to yield more nuanced representations, given training data that are orders of magnitude larger than those of BERT. Multimodal LLMs, trained on text–image pairs, could ultimately enable joint analysis across these two modalities.

<!-- #endregion -->

A more ambitious avenue would be to re‑train the models (fine‑tuning) so as to adapt their representations to the specificities of the corpus’s textual, visual, and audio‑visual data. Fine‑tuning a language model on the corpus texts could, for example, learn contextual similarity between orthographic variants of Arabic names and enforce proximity between Toumi and Tomi in the vector space.


Finally, generative AI (GenAI)—with or without fine‑tuning—could be used to obtain natural‑language descriptions of topics, or even direct thematic analyses of the corpus, by means of prompts that incorporate a broad context and, where appropriate, de‑biasing instructions, all while embedding safeguards (human verification, transparency).


#### c. Ethical positioning and use‑cases for GenAI in historical research and archival institutions


<!-- #region citation-manager={"citations": {"jms3k": [{"id": "10163145/UBB3ITVZ", "source": "zotero"}]}} -->
The rapid pace of deployment, coupled with the voracity of major GenAI multinationals, compels the historical research community and archival institutions that steward heritage data to take a position on the prospects and ethical questions associated with using GenAI for history.
Archival institutions such as the Institut national de l’audiovisuel (INA) host vast quantities of digitised data and media heritage, as well as born‑digital heritage (the archived web), which constitutes a tremendous reservoir of data for historians. At the same time, historians cannot remain indifferent to the claims extolling the power of GenAI. They need to understand and explore these tools critically and appropriately, while safeguarding scientific integrity and reproducibility. Considering the relationship between heritage data and the functions assigned to different AI models, three types of use can be envisaged for the study of historical sources:
1) Direct use of pre‑trained models for analysis or information retrieval (RAG without retraining).
AI models can be adapted to serve as tools for indexing, internal search, classification, detection, and metadata enhancement. This supports two objectives: discoverability (facilitating retrieval, increasing the likelihood that relevant archives are found) and exploratory analysis (summarising trends, characteristics, topics within a subset of the collection). A first GenAI configuration meeting these needs employs off‑the‑shelf models, trained on general‑purpose corpora (e.g., general French language data). Archival data are indexed using the model to enable improved answers to natural‑language queries. These retrieval‑augmented generation (RAG) systems can be deployed over an archival corpus without additional training (<cite id="jms3k"><a href="#zotero%7C10163145%2FUBB3ITVZ">(Pouyllau et al. 2024)</a></cite>). This would extend the approach proposed in our study of the March, yet it is limited by lower quality on specific or minoritised data, since the employed AI model would be trained only on general‑purpose corpora and not retrained on the research corpus. Legally, this configuration is relatively comfortable, but it is less compelling in qualitative terms.
2) Adaptation or fine‑tuning for enhanced discoverability.
Here, we still target analytical applications and collection discoverability, but with adapted AI models—retrained, or even trained from scratch—on the institution’s archival corpora. Retraining, formerly prohibitive for large models and unfeasible with closed models from Big Tech, is now facilitated by technologies such as Low‑Rank Adaptation (LoRA), which reduce memory costs and thus support local fine‑tuning, experimentation, and prototyping. For example, a model trained on the archived web of France TV could power a chatbot enabling researchers to navigate the site’s history differently and to generate automated data treatments (topic modelling, sentiment analysis, visual similarity, etc.).
3) Heritage‑aware generative modelling (trained on archives): generating synthetic content from archives.
In this configuration, an AI model is adapted/fine‑tuned/retrained not for content analysis but for synthetic content generation. For instance, a model fine‑tuned on the web archive of the France TV site from 1998 to the present could generate synthetic pages in the style of France TV at different periods (e.g., a prompt generates a France TV page dated 14 July 2003). For research, this may help visualise large narrative or visual trends, illustrate biases by extrapolating the corpus, test data quality, or explore art‑science relations. However, for the depositing institution, this raises legal questions: rights holders alone can authorise such uses. Neighbouring rights, together with the institution’s public‑service mandate, may prohibit large‑scale extraction for training purposes. Moreover, there is a trade‑off between the energy cost of such endeavours and their heuristic value, with the risk of a mere “gadget” effect.

<!-- #endregion -->

## Conclusion

<!-- #region editable=true slideshow={"slide_type": ""} -->
Our study of the 1983 March approaches the media coverage from the margins while contributing to the development of contemporary born‑digital web history—without disavowing continuities with media history (television here). In doing so, it both renews Memory Studies and advances a critical understanding of AI uses in the discipline. The focus on marginalised groups, the adoption of transmedia corpora, and the maintenance of a diachronic imperative together constitute a threefold challenge for integrating computational methods, including AI, into digital history. We have shown how pre‑existing historiographical biases around the Marche—notably the slow recognition of the marchers themselves in scholarship and the media—are further amplified by AI models operating on textual and visual data. Attending to socio‑cultural bias therefore guided the pipeline’s design at every point where AI was introduced, making the procedure reproducible and sustaining critical reflection. The experiment foregrounds virtuous modes of interdisciplinary knowledge production. A step‑by‑step approach requires close collaboration between archivists, who know heritage data and their biases; data scientists, who master technological developments; and historians, who cannot sustain a critical stance alone. For digital history, such collaboration presupposes baseline digital literacy, which both preserves critical distance and forms a foundation for upskilling alongside data scientists. In this respect, our approach constitutes a substantive response to the challenges GenAI poses to research in history and, more broadly, the humanities and social sciences.

<!-- #endregion -->

The AI‑assisted pipeline was conceived within a framework of transparency and continuous alignment between historical imperatives and advances in data science. We regularly evaluated the impact of different parameters on thematic modelling and the visualisation of topic models, and systematically explored multiple scenarios and tested several language models in light of the corpus’s specificities. A major contribution lies in transforming visualisation so that primary sources are re‑attached at each stage—from ingestion and cleaning through embedding, retrieval, and thematic exploration—closing a persistent gap between computation and historical evidence. Likewise, automatic speech recognition turns television archives into queryable text corpora, enabling historians to mobilise the full repertoire of textual methods (from concordances to topic modelling and narrative analysis) on audiovisual holdings. Taken together, these elements deliver practical gains (discoverability, reproducibility, auditability) and, above all, a methodological renewal for media history, making large‑scale, source‑aware, and critically interpretable analyses feasible across both born‑digital and digitised audiovisual collections. Conceived in this way, the AI‑assisted pipeline renews modes of interrogating corpora and casts the media coverage of the Marche in a new light. The interface articulates the identification of major themes across the corpus while enabling diachronic exploration by date and systematic linkage to each archived video. The analysis attests to the growing visibility of the marchers, despite ambivalent media treatment, and maps the associated societal debates, showing the principal trends and their evolution. Some themes already recognised in the literature re‑emerge—anti‑racism, police violence, and memorial cultural productions (e.g., the 2013 film)—while others surface only through processing the corpus as a whole, thereby enriching qualitative approaches based on sampled viewing. 


<!-- #region editable=true slideshow={"slide_type": ""} -->
More broadly, the study clarifies both the prospects and risks of AI in historical data work: our use of BERTopic illustrates the tension between the clustering power of contemporary tools and the risks of bias introduced by corpus composition and underlying models. A critical, transparent approach is therefore indispensable to prevent AI from reproducing or amplifying cultural and linguistic asymmetries. Building on these findings, we outline GenAI‑based avenues to improve the study and mitigate sociocultural bias—most notably by improving embeddings and re‑training models—and we distinguish three families of use to which research communities and GLAM institutions (galleries, libraries, archives and museums) will soon need to respond: Heritage‑aware Generative Modelling (trained on archives), AI‑enhanced Discoverability (fine‑tuning, embeddings, RAG), and Direct Use of General Models (RAG without training). Beyond technical potential, these paths raise legal and ethical questions specific to heritage data; from this perspective, pursuing source work “from the margins” sustains a critical approach to AI in which the historian retains a central role.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Bibliography
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} tags=["hidden"] -->
<!-- BIBLIOGRAPHY START -->
<div class="csl-bib-body">
  <div class="csl-entry"><i id="zotero|10163145/UXQWLN6C"></i>Bain, Max, Jaesung Huh, Tengda Han, and Andrew Zisserman. 2023. “WhisperX: Time-Accurate Speech Transcription of Long-Form Audio.” arXiv. <a href="https://doi.org/10.48550/arXiv.2303.00747">https://doi.org/10.48550/arXiv.2303.00747</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/N7PEIRSL"></i>BehnamGhader, Parishad, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. 2024. “LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders.” arXiv. <a href="https://doi.org/10.48550/arXiv.2404.05961">https://doi.org/10.48550/arXiv.2404.05961</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/HVXDTLZJ"></i>Ben-David, Anat, and Adam Amram. 2018. “The Internet Archive and the Socio-Technical Construction of Historical Facts.” <i>Internet Histories</i> 2 (1–2): 179–201. <a href="https://doi.org/10.1080/24701475.2018.1455412">https://doi.org/10.1080/24701475.2018.1455412</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/I2H36MMV"></i>Blei, David M. 2003. “Latent Dirichlet Allocation,” 30.</div>
  <div class="csl-entry"><i id="zotero|10163145/447J7MJ6"></i>Cer, D., M. Diab, E. Agirre, I. Lopez-Gazpio, and L. Specia. 2017. “Semeval-2017 Task 1: Semantic Textual Similarity-Multilingual and Cross-Lingual Focused Evaluation.” <a href="https://doi.org/arXiv:1708.00055">arXiv:1708.00055</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/CQ79HRL9"></i>Clavert, Frédéric. 2016. “Une Histoire Par Les Données ? Le Futur Très Proche de l’histoire Des Relations Internationales.” <i>Bulletin de l’Institut Pierre Renouvin</i> 44 (2): 119–30. <a href="https://doi.org/10.3917/bipr1.044.0119">https://doi.org/10.3917/bipr1.044.0119</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/KCYNM7TH"></i>Dhume, Fabrice. 2016. “Du Racisme Institutionnel à La Discrimination Systémique ? Reformuler l’approche Critique.” <i>Migrations Société</i> 163 (1): 33–46. <a href="https://doi.org/10.3917/migra.163.0033">https://doi.org/10.3917/migra.163.0033</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/3AE44DD7"></i>EURECOM, Sophia Antipolis, France, Ismail Harrando, Pasquale Lisena, EURECOM, Sophia Antipolis, France, Raphaël Troncy, and EURECOM, Sophia Antipolis, France. 2021. “Apples to Apples: A Systematic Evaluation of Topic Models.” In <i>Proceedings of the Conference Recent Advances in Natural Language Processing - Deep Learning for Natural Language Processing Methods and Applications</i>, 483–93. INCOMA Ltd. Shoumen, BULGARIA. <a href="https://doi.org/10.26615/978-954-452-072-4_055">https://doi.org/10.26615/978-954-452-072-4_055</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/3DVB7YXJ"></i>Eyries, Alexandre. 2020. “Claire Blandin, François Robinet, Valérie Schafer, Penser l’histoire des médias.” <i>Questions de communication</i>, no. 38 (déc): 663–6635. <a href="https://doi.org/10.4000/questionsdecommunication.24699">https://doi.org/10.4000/questionsdecommunication.24699</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/VXYHFRSB"></i>Fuica, B. T., and Arthur Lezer. 2023. “From Still to Moving Images and Vice Versa.” <i>NECSUS – European Journal of Media Studies</i> 12 (2): 150–75. <a href="https://doi.org/10.25969/mediarep/21721">https://doi.org/10.25969/mediarep/21721</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/CSHMTCCS"></i>Gebeil, Sophie. 2016. “Les Mémoires de La Marche Pour l’égalité et Contre Le Racisme.” <i>Hommes &#38; Migrations</i>, no. 1313. <a href="https://doi.org/10.4000/hommesmigrations.3574">https://doi.org/10.4000/hommesmigrations.3574</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/ZME5VVZN"></i>———. 2021. <i>Website Story. Histoire, Mémoires et Archives Du Web</i>. Ina Éditions.</div>
  <div class="csl-entry"><i id="zotero|10163145/WS6BRI54"></i>Grootendorst, Maarten. 2022. “BERTopic: Neural Topic Modeling with a Class-Based TF-IDF Procedure.” arXiv. <a href="https://doi.org/10.48550/arXiv.2203.05794">https://doi.org/10.48550/arXiv.2203.05794</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/ZHQWCEM3"></i>Hajjat, Abdellali. 2013. <i>La Marche Pour l’égalité et Contre Le Racisme</i>. Amsterdam Éditions.</div>
  <div class="csl-entry"><i id="zotero|10163145/YPJA74PZ"></i>———. 2022. <i>The Wretched of France: The 1983 March for Equality and against Racism</i>. Indiana University Press.</div>
  <div class="csl-entry"><i id="zotero|10163145/MRWCSDZ4"></i>Martin, L., B. Muller, P. O. Suarez, Y. Dupont, L. Romary, É. de La Clergerie, D. Seddah, and B. Sagot. 2020. “CamemBERT: A Tasty French Language Model.” In <i>ACL Proceedings</i>, 7203–19.</div>
  <div class="csl-entry"><i id="zotero|10163145/7IB85MLJ"></i>Ory, Pascal. 2007. <i>L’Histoire Culturelle (2e Éd.)</i>. Presses Universitaires de France.</div>
  <div class="csl-entry"><i id="zotero|10163145/UBB3ITVZ"></i>Pouyllau, S., A. Faci, A. Silvestre de Sacy, and L. Maronet. 2024. “ISIDORE 2030.”</div>
  <div class="csl-entry"><i id="zotero|10163145/ZECEJYDL"></i>Reimers, N., and I. Gurevych. 2019. “Sentence-BERT: Sentence Embeddings Using Siamese BERT-Networks.” <a href="https://doi.org/arXiv:1908.10084">arXiv:1908.10084</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/HGXGKTED"></i>Rendina, Davide, Sophie Gebeil, Mathieu Génois, and Patrice Bellot. 2024. “Semantic Analysis of Web Archive Historical Data.” In <i>Exploring the Archived Web during a Highly Transformative Age Proceedings of the 5th International RESAW Conference, Marseille, June 2023</i>.</div>
  <div class="csl-entry"><i id="zotero|10163145/ZPL7888X"></i>Röder, Michael, Andreas Both, and Alexander Hinneburg. 2015. “Exploring the Space of Topic Coherence Measures.” In <i>Proceedings of the Eighth ACM International Conference on Web Search and Data Mining</i>, 399–408. Shanghai China: ACM. <a href="https://doi.org/10.1145/2684822.2685324">https://doi.org/10.1145/2684822.2685324</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/EUF5F7AZ"></i>Srivastav, Vaibhav, Steven Zheng, Eric Bezzam, Eustache Le Bihan, Adel Moumen, and Sanchit Gandhi. 2025. “Open ASR Leaderboard: Towards Reproducible and Transparent Multilingual Speech Recognition Evaluation.” arXiv. <a href="https://doi.org/10.48550/arXiv.2510.06961">https://doi.org/10.48550/arXiv.2510.06961</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/Q6SU9JHQ"></i>Terragni, Silvia, Elisabetta Fersini, Bruno Giovanni Galuzzi, Pietro Tropeano, and Antonio Candelieri. 2021. “OCTIS: Comparing and Optimizing Topic Models Is Simple!” In <i>Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations</i>, 263–70. Online: Association for Computational Linguistics. <a href="https://doi.org/10.18653/v1/2021.eacl-demos.31">https://doi.org/10.18653/v1/2021.eacl-demos.31</a>.</div>
  <div class="csl-entry"><i id="zotero|10163145/EWSSBAJP"></i>Wihtol de Wenden, C. 2016. “Police et Discriminations Institutionnelles.” <i>Cliniques Méditerranéennes</i> 94 (2): 83–92. <a href="https://doi.org/10.3917/cm.094.0083">https://doi.org/10.3917/cm.094.0083</a>.</div>
</div>
<!-- BIBLIOGRAPHY END -->
<!-- #endregion -->
