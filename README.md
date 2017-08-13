To run, clone the repository and run ```pip install``` if needed. Then run ```python ff_task.py``` from the terminal. 

Example run:

```
>>> python ff_task.py
Smallest group: 41; trimming others to 102 items or fewer  
Basic training size      420	F1=0.707  
Enahnced training size 10859	F1=0.987  

Claim 'Many workers arrived from Eastern Europe' assigned to topic: Immigration  
Claim 'EU farm subsidies are for EU farms' assigned to topic: Europe
Claim 'Rise in smoking linked to straight bananas' assigned to topic: Health
Claim 'British workers put in longer hours for lower pay than German workers.' assigned to topic: Law 
```

The first time the program runs, documents will be downloaded and stored in a local cache; subsequent runs will therefore be faster.

New sentences can be classified by editing the last part of the code and re-running it; e.g.:

```classify_sentences(classifier, ["This is a claim about education"])```


