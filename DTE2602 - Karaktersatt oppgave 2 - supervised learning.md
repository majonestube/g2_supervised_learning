# Karaktersatt oppgave 2 | DTE-2602 | H23

_Trykk Ctrl+K og deretter V for å vise dokumentet som formatert Markdown i VSCode._

## Introduksjon
I denne oppgaven skal du bruke to ulike varianter av "supervised learning" for å prøve artsbestemme pingviner basert på ulike målinger. Datasettet som skal brukes er "Palmer penguins" som du antakelig allerede kjenner fra før fra innlevering 6. Du finner datasettet i fila ``palmer_penguins.csv`` i repository for oppgaven.

<img src="https://github.com/allisonhorst/palmerpenguins/blob/main/man/figures/lter_penguins.png?raw=true" alt="Drawing" style="width: 600px;"/>

Figuren under viser ulike varianter av "scatterplott" for de fire numeriske størrelsene i datasettet:
- bill_length_mm
- bill_depth_mm
- flipper_length_mm
- body_mass_g

Plottene viser alle parvise kombinasjoner av disse fire. Diagonalen viser histogram for enkelt-størrelser. Legg merke til hvordan noen kombinasjoner gir punktskyer der de tre klassene (pingvinartene) overlapper mye, mens andre kombinasjoner gir bedre separasjon.

![Scatter-plott av Palmer penguins](https://seaborn.pydata.org/_images/pairplot_3_0.png)
_Figur hentet fra https://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot_

## Vurdering av oppgaven
Besvarelsen på oppgaven består av to deler som skal leveres **via opplasting til repository på GitHub Classroom**:
- En Python-fil med implementasjon. Implementasjonen kan bruke biblioteker som er inkludert i Python 3.10, samt ``binarytree``, ``numpy``, ``matplotlib`` og ``scipy.stats``. _Ingen andre eksterne biblioteker er tillatt._ Teller 60% .
- En rapport i form av et PDF-dokument (f.eks. generert fra MS Word eller LaTeX) med beskrivelse av din egen løsning. Rapporten bør være minst 4 sider lang (A4, skriftstørrelse 11, enkel linjeavstand), men du kan skrive den så lang som du vil. Rapporten kan skrives på norsk eller engelsk. Teller 40 %

**Implementasjonen** vurderes etter følgende kriterier:
- **Korrekthet**: Koden svarer på oppgaven og fungerer som den skal. Koden skal være kjørbar. Merk at noen av funksjonene som skal implementeres har tilhørende enhetstester i fila supervised_learning_test.py. 
- **Effektivitet**: Man har gjort gode valg i implementasjonen. Koden trenger ikke være lynrask, men man skal unngå unødvendig tungvinte løsninger.   
- **Lesbarhet**: Koden har 
    - Beskrivende docstrings for alle funksjoner / metoder 
    - En passe mengde kommentarer som tydeliggjør hensikten med koden. Mest relevant for kodelinjer der det ikke er åpenbart hva koden gjør.  
    - Beskrivende variabelnavn.
    - Formatering som følger anbefalinger gitt i [video om kodestandard](https://uit.instructure.com/courses/30442/modules/items/862106) (trenger ikke alltid følge absolutt alle regler, men bør ha god og «standard» formatering)

**Rapporten** vurderes etter følgende kriterier:
- **Korrekthet**: Svarer på konkrete punkter etterspurt i oppgaven. Bruker riktig teori og ligninger der dette er relevant. 
- **Referanse til kode**: Det skal være en tydelig sammenheng mellom innholdet i rapporten og koden. Vis f.eks. hvordan evt. matematiske ligninger i rapporten er implementert i koden. 
- **Lesbarhet og struktur**: Språket er korrekt og tydelig, og skrevet i fullstendige setninger. Dokumentet er tydelig og konsistent formatert, med overskrifter og inndeling som gjør det lett å finne fram i. Overskriftene i rapporten bør ikke være "oppgave 1", "oppgave 2", osv. Den skal heller ikke være en "dagbok". Rapporten skal beskrive _helheten_ i løsningen av oppgaven, som om du skulle forklare den til en student på ditt nivå som aldri har hørt om temaet for oppgaven før. Overskriftene bør være _noe sånt som_ "Problemstilling", "Teori", "Implementasjon", "Eksperimenter", "Resultater", og så videre - men du står fritt til å velge overskriftene selv. 
- **Selvrefleksjon**: I rapporten skal du også vurdere ditt eget arbeid. Fungerer løsningen din? Hvis ikke, hva tror du er årsaken? Er det noe du ville ha gjort annerledes? 

## Oppgaver
Oppgaven er delt opp i underoppgaver, med et maksimalt antall poeng for hver. Man kan få 100 poeng totalt.

### Forberede data og vurdere resultater
1. Les inn datasettet fra ``palmer_penguins.csv``. Fjern alle rader som mangler data (rader som inneholder "NA"). Legg data for de fire numeriske størrelsene (bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g) i en matrise X. Normaliser verdiene i hver kolonne i X gjennom å trekke fra gjennomsnittsverdi for kolonnen og dele på standardavviket (["Z-score"](https://en.wikipedia.org/wiki/Standard_score)). Konverter informasjon om art ("species") til heltall 0, 1 og 2 for henholdsvis Adelie, Chinstrap og Gentoo, og legg dette i en vektor y. Implementeres som funksjon read_data(). **5 poeng**
2. Del opp datasettet ditt i et trenings-datasett (X_train,y_train) og et test-datasett (X_test,y_test). Gi en begrunnelse for hvor mange observasjoner (samples) du legger i hvert av disse. Implementeres som funksjon ``train_test_split()``. **5 poeng**
3. Lag en funksjon ``accuracy()`` som tar en vektor $y_{\text{true}}$ med "sanne" verdier og en vektor $y_{\text{pred}}$ med estimerte verdier som input. Funksjonen skal beregne hvor stor andel av elementene som er _like_ i de to vektorene. Vi skal bruke dette senere for å vurdere samsvar mellom "fasit" og output fra våre maskinlæringsmodeller.  **5 poeng**


### Perceptron
4. Implementer en klasse ``Perceptron``. Klassen skal ha metodene ``.train()`` og ``.predict()``, som indikert i startkoden. Metodene brukes til henholdsvis trening av modellen og anvendelse av modellen på nye data. **15 poeng**
5. Bruk datasettet (X_train,y_train) til å trene en Perceptron-modell til å gjenkjenne pingvinarten **Gentoo**. Du skal kun bruke kolonnene _bill_depth_mm_ og _flipper_length_mm_ fra X-matrisa. Output fra modellen ($y_{\text{pred}}$) skal være 1 hvis arten er Gentoo, og 0 hvis den ikke er det (dvs. en av de andre to artene). Du kan bruke funksjonen ``convert_y_to_binary()`` for å konvertere til binære verdier. Bruk datasettet (X_test,y_test) sammen med funksjonen ``accuracy()`` fra oppgave 3 for å måle nøyaktigheten til modellen. **10 poeng**
6. Visualiser "decision boundary" for modellen i oppgave 5. Lag et plott som viser hver pingvin som et punkt (bruk ``matplotlib.pyplot.scatter()``). Plott så decision boundary for modellen som en rett linje (bruk hjelpemetoden ``get_line_x_y()``, og ``matplotlib.pyplot.plot()``). Hvis alt er riktig bør linja ligge mellom Gentoo-punktene og punktene for de to andre artene. **5 poeng**
7. Lag et nytt perceptron som skal skille arten **Chinstrap** fra de to andre. Du står fritt til å bruke hele X-matrisa eller et utvalg av kolonnene fra X. Gi en begrunnelse for hvilke features du velger. Merk at det ikke er sikkert at modellen konvergerer. Forklar i så fall hvorfor. Mål nøyaktigheten til modellen med (X_test,y_test) og ``accuracy()``. **10 poeng**

### Decision tree
8. Implementer funksjonene ``gini_impurity()``, ``gini_impurity_reduction()`` og ``best_split_feature_value()`` som vist i startkoden. **15 poeng**
9. Implementer en klasse ``DecisionTree``. Deler av implementasjonen er allerede ferdig og er oppgitt i startkoden. Du skal implementere den rekursive metoden ``._predict()``, som bruker et ferdig trent beslutningstre for å klassifisere nye data. **10 poeng**
10. Bruk datasettet (X_train,y_train) til å trene en modell for å gjenkjenne arten **Gentoo**, på samme måte som i oppgave 5. Mål nøyaktigheten med (X_test,y_test) og ``accuracy()``. **5 poeng**
11. Bruk datasettet (X_train,y_train) til å trene en modell for å gjenkjenne arten **Chinstrap**, på samme måte som i oppgave 7. Mål nøyaktigheten med (X_test,y_test) og ``accuracy()``. **5 poeng**
12. Bruk datasettet (X_train,y_train) til å trene en modell som skal skille mellom **alle tre pingvinartene**. Kommentér i rapporten: Hvorfor er klassifisering av mer enn to kategorier mulig med et decision tree, og ikke med et perceptron? Mål nøyaktigheten med (X_test,y_test) og ``accuracy()``.  **5 poeng**.
 
### Oppsummering
13. Sammenlign resultatene for de ulike modellene du har trent og testet. Er det forskjeller mellom resultatene? Hva er sannsynlige grunner til det, i så fall? **5 poeng**