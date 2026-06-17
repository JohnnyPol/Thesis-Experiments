# TODO

group all exp1.* 
- table: experiment | accuracy | inference_time | throughput | model | exit0 exit1 exit2 exit3 - experiments 1.1 1.2 1.3 
- utilization (worker id - utilization) throughput,
exp2:
- avg inference time
- node utilization for each, same as 1.3

exp3.1 3.2 3.3:
barplot utilization
- compare the experiments, find the best (subsection)
- avg inference time

## Writing / Thesis
- [*] fix all the overfull \hbox warnings
- [ ] Shift the focus to the distribution of the multi-model setup.
- [ ] Write down that we trained the models in a Jupyter Notebook
- [ ] δωσε λογο γιατι χρησιμοποιεις την καθε μετρικη που θα χρησιμοποιησεις οντως
- [ ] make diagrams and tables for cifar10.1 as well
- [ ] grafana monitoring for future work
- [ ] προσθεσε την περιλιψη και τα keywords στα ελληνικα και στα αγγλικα
- [ ] 3 Main συμπεράσματα, 
MIXED PRECISION

## Writing / Thesis (Status of Chapters)
- 1.1 Checked 
- 1.2 Revision at the end
- 1.3 Revision at the end
- 2.1 Checked
- 2.2 Checked
- 2.3 Checked
- 2.4 Checked
- 3.1 Read it one more time
- 3.2 Read it one more time
- 3.3 Read it one more time
- 3.4 Read it one more time
- 4.1 Read it one more time
- 4.2 Read it one more time
- 4.3 Read it one more time
- 4.4 Read it one more time
- 4.5 Read it one more time
- 4.6 Talk with Grigoris and make sure everything is factual
- 4.7 Read it one more time
- 4.8 Read it one more time
- 4.9 Read it one more time
- 4.10 Read it one more time
- 4.11 Read it one more time

## ΟΡΟΛΟΓΙΑ ΝΑ ΤΣΕΚΑΡΕΙΣ ΑΝ ΕΙΝΑΙ ΙΔΙΑ ΠΑΝΤΟΥ
- latency (CHECK)
- threshold (CHECK)
- throughput (CHECK)

## ΠΡΑΓΜΑΤΑ ΝΑ ΡΩΤΗΣΕΙΣ
- Επεξηγηση κωδικα για το training επειδη τα βαρη τα πηρα ετοιμα (CHECK)
- Γιατί έχουμε βάλει τα δύο πρώτα early exits στο πρώτο partition? Είναι αρκετά μεγάλο το ποσοστό που τερματίζουν στο πρώτο partition. NA ΔΕΙΞΟΥΜΕ ΠΟΣΟ ΔΥΝΑΛΟΓΟ ΕΙΝΑΙ ΤΟ ΦΟΡΤΙΟ ΣΤΑ ΠΡΩΤΑ EXIT BRANCHES. ΠΩΣ ΜΠΟΡΟΥΜΕ ΝΑ ΑΞΙΟΠΟΙΗΣΟΥΜΕ ΑΥΤΗ ΤΗΝ ΑΝΙΣΟΡΟΠΙΑ; 
FUTURE WORK -> PARTITION 0 (EXIT0) ... PARTITION 2 (EXIT 2 EXIT 3)

Παρά την πρόοδο στα early-exit DNNs, δεν υπάρχει ακόμη πλήρως τυποποιημένη μέθοδος για το πώς συνδυάζονται βέλτιστα early exits, split computing, offloading decisions και resource-aware inference σε δυναμικά edge-cloud περιβάλλοντα

## Cleanup
- [ ] Update README files and the experiments descriptions on the repo
- [ ] Change the repo name into something more appropriate