import pandas as pd
df = pd.read_csv("evaluation_results/resultados_granulares_Config_A.csv")
cols = ['bleu_4', 'rouge_l', 'radgraph_f1', 'chexpert_precision', 'chexpert_recall', 'chexpert_f1']
print(df[cols].mean())