import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import random

path_to_txt = 'MAPP_Corpus/English_sanitized_policies'
path_to_csv = 'MAPP_Corpus/English_consolidation'

# Считываем содержимое политик безопасности
txt_files = [f for f in listdir(path_to_txt) if isfile(join(path_to_txt, f))]
txt_policies = []
csv_files = [f for f in listdir(path_to_csv) if isfile(join(path_to_csv, f))]
csv_annotations = []
# Читаем текст политик безопасности
for txt in txt_files:
    with open(path_to_txt + '/' + txt, 'r', encoding='utf-8') as f:
        buf = f.read()
        txt_policies.append(buf.split('\n\n')[:-1])
data_types_1d = []
# Читаем аннотации к политикам
for csv in csv_files:
    df = pd.read_csv(path_to_csv + '/' + csv, encoding='utf-8')
    data_types = df['value name']
    new_data_type = []
    for data_type in data_types:
        if data_type == '[]':
            new_data_type.append([])
            continue
        new_data_type.append(data_type[1:-1].replace("'", '').split(','))
    data_type_1d = list(set([j for sub in new_data_type for j in sub]))
    data_types_1d.append(data_type_1d)
    csv_annotations.append(new_data_type)
# Создаем список со всеми встретившимися аннотациями
all_annotations = list(set([j for sub in data_types_1d for j in sub]))
all_annotations = [j.strip() for j in all_annotations]

# Создаем датасет на основе считанных данных
paragraphs = []
annotations = []
text = []
label = []
# Читаем по абзацам и аннотациям, соответствующим этим абзацам
for path, text, annotation in zip(csv_files, txt_policies, csv_annotations):
    i = 1
    for para, annot in zip(text, annotation):
        # Если нет аннотации, можно пропускать абзац
        if annot == []:
            for k in range(random.randint(1, 3)):
                a = random.choice(all_annotations)
                paragraphs.append(para.replace(',', '').replace(';', ''))
                annotations.append(a.replace(',', '').strip().replace(' ', '_'))
                label.append('No')
            i += 1
            continue
        j = 1
        tmp_annot = []
        # Читаем все аннотации, существующие в текущем абзаце
        for a in annot:
            paragraphs.append(para.replace(',', '').replace(';', ''))
            annotations.append(a.replace(',', '').strip().replace(' ', '_'))
            tmp_annot.append(a.replace(',', '').strip().replace(' ', '_'))
            label.append('Yes')
            j += 1
        # Добавляем несколько отсутствующих аннотаций
        if len(tmp_annot) > 2:
            for i in range(len(tmp_annot) - 1):
                a = random.choice(all_annotations)
                if a not in tmp_annot:
                    paragraphs.append(para.replace(',', '').replace(';', ''))
                    annotations.append(a.replace(',', '').strip().replace(' ', '_'))
                    label.append('No')
        i += 1

# Сохраняем датасет
new_df = pd.DataFrame({'Text' : paragraphs, 'Annotaion' : annotations, 'Result' : label}).sample(frac=1)
new_df.to_csv('dataset_1.csv', sep=',', index=None, encoding='utf-8')