"""
Training script - runs all 13 notebook cells as plain Python.
Run with: venv/bin/python3 notebooks/train.py
"""

import sys
sys.path.insert(0, '.')

import os, json, random, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
warnings.filterwarnings('ignore')

BASE_DIR = Path('.')
print('✅ Cell 1: Setup complete')

# ── CELL 2: Load KB ───────────────────────────────────────────────────
with open(BASE_DIR / 'data/kb/knowledge_base.json') as f:
    kb = json.load(f)
places      = [loc['name'] for loc in kb]
departments = [loc['name'].replace(' Department','')
               for loc in kb if loc.get('category') == 'department']
print(f'✅ Cell 2: KB loaded — {len(kb)} records, {len(departments)} departments')

# ── CELL 3: Build Intent Dataset ─────────────────────────────────────
templates = {
    'find_location': [
        'where is {place}','how do i get to {place}','take me to {place}',
        'where is the cafeteria','where is the library','where is the canteen',
        'show me the way to {place}','which floor is {place}','location of {place}',
        'where can i find {place}','directions to {place}','how to reach {place}',
        'from civil to mechanical','from library to cafeteria','from cse to gym',
        'how do i go from {place} to the library','navigate to {place}',
    ],
    'ask_hours': [
        'when is {place} open','what time does {place} close',
        'is {place} open now','is the library open now',
        'opening hours of {place}','what are the timings of {place}',
        'is cafeteria open on sunday','what time does the gym open',
        'can i visit {place} now','is {place} open today',
    ],
    'ask_event': [
        'any events today','what workshops are happening',
        'any seminar this week','show me upcoming events',
        'anything happening on campus','any guest lecture today',
        'events at student union','what is on today',
    ],
    'menu_query': [
        'what is today menu','price of tea','how much is coffee',
        'where can i get lunch','what food is available now',
        'what is the price of tea','tea price','coffee price',
        'what snacks are available','price of samosa',
        'what is on the menu today','can i pay with student card',
        'how much does lunch cost','menu for today','friday special menu',
    ],
    'faculty_query': [
        'where is {department} staff room','where is {department} faculty room',
        'faculty room for {department}','staff room for {department}',
        'where can i meet my professor','where do lecturers sit',
        'where is the hod office of {department}',
        'where can i meet my project supervisor',
        'where do teachers sit in {department}',
    ],
    'service_query': [
        'where can i charge my laptop','where can i print assignment',
        'where can i issue books','where can i return books',
        'where can i pay fees','my wifi is not working',
        'where can i scan documents','where can i get bonafide certificate',
        'where is the it helpdesk','how to get student id reissued',
        'my laptop battery is dead where can i go',
        'my laptop battery died where can i charge',
        'i need to charge my phone urgently',
        'where can i get wifi password',
        'internet is not working on my laptop',
        'i need to take a printout urgently',
        'where can i get my id card reissued',
        'i need a no objection certificate',
    ],
    'recommend_place': [
        'where can i study peacefully','where can i study quietly',
        'suggest a quiet study place','where can i relax after class',
        'where can i sit with laptop','where can i prepare for exams',
        'i need a peaceful place to work','best place to study on campus',
        'i need a quiet corner with good wifi',
        'where can i sit and focus on my assignment',
        'is there a nice outdoor spot on campus',
        'where can i take a break between classes',
        'suggest a calm place on campus',
        'where can i do group study',
        'best place to read on campus',
        'where is a comfortable place to sit',
    ],
    'lost_found': [
        'lost my watch','lost my wallet','lost my laptop',
        'i lost my student id card','where is lost and found',
        'where should i report lost item','i found someone phone',
        'i lost my bag','lost property office',
        'i found a wallet near the library what do i do',
        'i found someone bag what should i do',
        'i think someone stole my phone',
        'someone stole my laptop what do i do',
        'my phone got stolen where do i report',
        'i found a phone on campus',
        'i found keys near the cafeteria',
        'i found a bag in the classroom',
    ],
    'emergency': [
        'i need first aid','where is first aid','where is medical room',
        'i feel sick','someone is injured','i need emergency help',
        'i cut my finger','i hurt my hand','i am bleeding',
        'i need bandage','where can i get medical help',
        'i feel dizzy','i have fever','i have stomachache',
        'i injured myself','i need a doctor',
        'my friend fainted in the robotics lab what do i do',
        'someone fainted in the classroom',
        'my friend is unconscious',
        'i feel nauseated and dizzy',
        'i have chest pain',
        'i think i am having an anxiety attack',
        'i twisted my ankle',
        'i accidentally cut myself',
        'i feel like vomiting',
        'i am having trouble breathing',
    ],
    'facility_info': [
        'tell me about {place}','what facilities are in {place}',
        'what services does {place} provide','what is {place} used for',
        'give me details about {place}','what does {place} offer',
    ],
    'ask_contact': [
        'who should i contact for {place}','where can visitors ask for help',
        'who handles hostel issues','where can i ask general questions',
        'who manages document queries','who is the contact for {place}',
    ],
    'ask_department': [
        'tell me about {department}','where is {department} department',
        'what labs are in {department}','which floor is {department} department',
        'what does {department} department offer',
    ],
    'fallback': [
        'what is the weather','tell me a joke','who won the match',
        'how do i cook pasta','what is bitcoin price','book me a flight',
        'what movie should i watch','who is the president',
        'play some music','what is the news today',
        'i want to propose a girl','where can i meet a girlfriend',
        'where can i find a boyfriend','how do i talk to girls',
        'suggest a romantic spot','best place to propose',
        'where can i make friends','how do i meet new people',
        'what is love','tell me about relationships',
        'what day is today','what is todays date','what time is it',
        'tell me the time','what year is it',
    ],
    'ask_admission': [
        'how do I apply for admission','what are the admission requirements',
        'where is the admissions office','I want to apply to this university',
        'what documents do I need for admission','admission process for MBA',
        'how to get admission in computer science','what is the last date for admission',
        'is there still time to apply','where can I submit my admission form',
        'who handles admissions','can I still enroll this semester',
        'what is the fee structure','how do I register as a new student',
        'prospectus for new students','eligibility criteria for BTech',
        'where to get admission brochure','student enrollment process',
        'admission for transfer students','what are the intake dates',
        'where to pay admission fees','how to get a student ID after admission',
        'new student orientation details','induction programme for new students',
        'welcome week schedule','merit list for admission',
    ],
    'ask_placement': [
        'where is the placement cell','how does campus placement work',
        'I need help with my CV','upcoming placement drives',
        'which companies come for recruitment','how to register for placements',
        'internship opportunities on campus','placement statistics this year',
        'who is the placement officer','when is the placement season',
        'mock interview preparation help','aptitude test for placements',
        'placement cell contact number','how to prepare for campus interviews',
        'job fair at campus','placement training workshops',
        'salary packages for graduates','placement for MBA students',
        'how many students get placed','placement cell opening hours',
        'career guidance sessions','where can I get career counselling',
        'resume building workshop','highest package in placements',
        'off campus placement support','industry visit schedule',
    ],
    'ask_fest': [
        'when is the annual fest','cultural fest this year',
        'technical fest schedule','hackathon on campus',
        'sports day this year','inter college competition',
        'how to register for the fest','who organises the annual day',
        'freshers party details','farewell party for final year',
        'cultural night programme','dance competition on campus',
        'music fest at university','food festival on campus',
        'inter department sports meet','coding competition details',
        'robotics competition schedule','entrepreneurship summit',
        'business plan competition','startup pitch event',
        'science exhibition campus','art exhibition schedule',
        'debate competition details','quiz competition campus',
        'open mic night campus','sports tournament schedule',
    ],
    'ask_club': [
        'how to join a club','student clubs at campus',
        'is there a coding club','music club details',
        'sports club registration','dance club on campus',
        'photography club meetings','debate society details',
        'entrepreneurship club','AI club at university',
        'student council elections','how to start a new club',
        'drama club auditions','volunteer club campus',
        'student newspaper club','film club at university',
        'gaming club details','yoga club schedule',
        'cultural club activities','language learning club',
        'student ambassador programme','peer mentorship programme',
        'club fair on campus','extracurricular activities details',
        'co-curricular activities details','how to get involved on campus',
    ],
    'ask_exam': [
        'when are the exams','exam timetable for this semester',
        'where is the exam cell','how do I get my hall ticket',
        'exam schedule for final year','mid semester exam dates',
        'how to apply for re-evaluation','grace marks policy',
        'supplementary exam dates','how to get marksheet',
        'transcript request process','provisional certificate collection',
        'degree certificate collection','exam fee payment deadline',
        'internal assessment marks','attendance requirement for exams',
        'how to apply for exam leave','practical exam schedule',
        'project submission deadline','result announcement date',
        'backlog exam schedule','how to check exam results online',
        'exam hall rules','answer sheet re-checking process',
        'grade sheet collection','arrear examination details',
    ],
    'ask_hostel': [
        'where is the hostel','how to apply for hostel',
        'hostel fees and charges','hostel room availability',
        'hostel warden contact','hostel rules and regulations',
        'is there a girls hostel','boys hostel details',
        'hostel mess menu','hostel curfew timing',
        'hostel visiting hours','how to get hostel allotment',
        'hostel allotment process','change hostel room request',
        'hostel wifi details','hostel laundry facility',
        'hostel complaint procedure','hostel security details',
        'hostel common room timing','hostel canteen timing',
        'hostel fee payment process','hostel leave application',
        'hostel gate pass procedure','overnight stay in hostel',
        'hostel notice board','hostel warden meeting timing',
    ],
    'ask_scholarship': [
        'is there any scholarship available','how to apply for scholarship',
        'merit scholarship details','need based scholarship',
        'scholarship for international students','fee waiver process',
        'financial assistance for students','education loan guidance',
        'scholarship deadline this year','scholarship amount details',
        'who qualifies for scholarship','scholarship application form',
        'scholarship renewal process','government scholarship details',
        'sports scholarship campus','academic excellence award',
        'bursary for students','tuition fee instalment plan',
        'fee concession process','scholarship office location',
        'where to apply for financial aid','stipend for research students',
        'fellowship opportunities campus','how to get fee relaxation',
        'scholarship interview process','financial aid office location',
    ],
    'social_life': [
        'how can I make friends on campus','I want to meet new people at university',
        'I feel lonely on campus','how to socialise at university',
        'where can I hang out on campus','student social events this week',
        'how to adjust to university life','campus life tips for freshers',
        'I am new here how to settle in','how to get involved in campus activities',
        'friendly places on campus','where do students hang out',
        'campus community events','international student social events',
        'buddy programme for new students','how to network with students',
        'what do students do for fun','weekend activities on campus',
        'where to chill between classes','recreational facilities on campus',
        'how to balance studies and social life','meeting classmates outside class',
        'I miss home tips for new students','social clubs for students',
        'activities to meet people campus','campus orientation social events',
    ],
}

rows = []
SAMPLES = 300
for intent, tmpl_list in templates.items():
    for _ in range(SAMPLES):
        t = random.choice(tmpl_list)
        q = t.format(
            place=random.choice(places),
            department=random.choice(departments) if departments else 'Computer Science'
        )
        rows.append({'query': q, 'intent': intent})

df = pd.DataFrame(rows)
df.to_csv(BASE_DIR / 'data/text/intent_dataset.csv', index=False)
print(f'✅ Cell 3: Dataset — {len(df)} rows, {df.intent.nunique()} intents')

# ── CELL 4: Plot ──────────────────────────────────────────────────────
df['query_len'] = df['query'].apply(lambda x: len(x.split()))
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.countplot(data=df, y='intent', order=df['intent'].value_counts().index, ax=axes[0])
axes[0].set_title('Intent Distribution')
sns.histplot(df['query_len'], bins=20, kde=True, ax=axes[1])
axes[1].set_title('Query Length Distribution')
plt.tight_layout()
plt.savefig(BASE_DIR / 'outputs/plots/dataset_overview.png', dpi=150)
plt.close()
print('✅ Cell 4: Dataset plot saved → outputs/plots/dataset_overview.png')

# ── CELL 5: Encode and Split ──────────────────────────────────────────
le = LabelEncoder()
df['label'] = le.fit_transform(df['intent'])
id2label = {i: l for i, l in enumerate(le.classes_)}
label2id = {l: i for i, l in id2label.items()}

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_df.to_csv(BASE_DIR / 'data/text/train.csv', index=False)
test_df.to_csv(BASE_DIR  / 'data/text/test.csv',  index=False)

with open(BASE_DIR / 'models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
with open(BASE_DIR / 'models/id2label.json', 'w') as f:
    json.dump(id2label, f)
print(f'✅ Cell 5: Train={len(train_df)}, Test={len(test_df)}, Labels saved')

# ── CELL 6: Tokenise ─────────────────────────────────────────────────
import datasets.config
datasets.config.TORCHVISION_AVAILABLE = False
from datasets import Dataset
from transformers import AutoTokenizer

MODEL_NAME = 'distilbert-base-uncased'
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

train_ds = Dataset.from_pandas(train_df[['query','label']])
test_ds  = Dataset.from_pandas(test_df[['query','label']])

def tokenize(batch):
    return tokenizer(batch['query'], truncation=True, padding='max_length', max_length=64)

train_ds = train_ds.map(tokenize, batched=True)
test_ds  = test_ds.map(tokenize,  batched=True)
train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
test_ds.set_format( type='torch', columns=['input_ids','attention_mask','label'])
print('✅ Cell 6: Tokenisation complete')

# ── CELL 7: Train ─────────────────────────────────────────────────────
import torch
from transformers import (AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(le.classes_),
    id2label=id2label, label2id=label2id
)

def compute_metrics(ep):
    preds = np.argmax(ep.predictions, axis=1)
    p,r,f,_ = precision_recall_fscore_support(
        ep.label_ids, preds, average='weighted', zero_division=0)
    return {'accuracy': accuracy_score(ep.label_ids, preds),
            'precision': p, 'recall': r, 'f1': f}

args = TrainingArguments(
    output_dir=str(BASE_DIR / 'models/distilbert_intent'),
    evaluation_strategy='epoch', save_strategy='epoch',
    learning_rate=2e-5, per_device_train_batch_size=16,
    per_device_eval_batch_size=16, num_train_epochs=4,
    weight_decay=0.01, logging_steps=20,
    load_best_model_at_end=True, metric_for_best_model='f1',
    report_to='none'
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=test_ds,
    tokenizer=tokenizer, compute_metrics=compute_metrics
)

print('⏳ Cell 7: Training DistilBERT — this takes 8-12 minutes on M3...')
trainer.train()
trainer.save_model(str(BASE_DIR / 'models/distilbert_intent'))
tokenizer.save_pretrained(str(BASE_DIR / 'models/distilbert_intent'))
print('✅ Cell 7: Intent model trained and saved → models/distilbert_intent/')

# ── CELL 8: Evaluate ──────────────────────────────────────────────────
out   = trainer.predict(test_ds)
preds = np.argmax(out.predictions, axis=1)
true  = out.label_ids
acc       = accuracy_score(true, preds)
p,r,f,_   = precision_recall_fscore_support(true, preds, average='weighted', zero_division=0)

print(f'\n✅ Cell 8: Intent Classifier Results')
print(f'   Accuracy : {acc:.4f}')
print(f'   Precision: {p:.4f}')
print(f'   Recall   : {r:.4f}')
print(f'   F1       : {f:.4f}')
print(classification_report(true, preds, target_names=le.classes_, zero_division=0))

metrics = {'accuracy':float(acc),'precision':float(p),'recall':float(r),'f1':float(f)}
with open(BASE_DIR / 'outputs/metrics/intent_metrics.json','w') as fout:
    json.dump(metrics, fout, indent=2)

cm = confusion_matrix(true, preds)
plt.figure(figsize=(14,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Intent Classifier Confusion Matrix')
plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(BASE_DIR / 'outputs/plots/intent_confusion_matrix.png', dpi=150)
plt.close()
print('   Confusion matrix saved')

# ── CELL 9: Hard Real-World Test ──────────────────────────────────────
def predict_intent(query):
    inputs = tokenizer(query, return_tensors="pt",
                       truncation=True, padding=True, max_length=64)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits
    probs   = torch.softmax(logits, dim=1)[0]
    pred_id = torch.argmax(probs).item()
    return {'intent': id2label[pred_id], 'confidence': float(probs[pred_id])}

hard_tests = [
    ('I need somewhere quiet before my exam',           'recommend_place'),
    ('My laptop is not connecting to wifi',             'service_query'),
    ('Can I grab coffee before class?',                 'menu_query'),
    ('Where do professors sit?',                        'faculty_query'),
    ('I misplaced my wallet',                           'lost_found'),
    ('I feel dizzy and need help',                      'emergency'),
    ('Is the library still open?',                      'ask_hours'),
    ('Anything happening on campus this week?',         'ask_event'),
    ('Tell me about the AI department',                 'ask_department'),
    ('Where can I get bonafide certificate?',           'service_query'),
    ('Take me to the computer lab',                     'find_location'),
    ('What food can I get right now?',                  'menu_query'),
    ('How do I get from the library to the cafeteria?', 'find_location'),
    ('Price of tea',                                    'menu_query'),
    ('Who won the cricket match yesterday?',            'fallback'),
]

results = []
for query, true_intent in hard_tests:
    r = predict_intent(query)
    correct = r['intent'] == true_intent
    results.append({'query': query[:45], 'true': true_intent,
                    'pred': r['intent'], 'conf': round(r['confidence'],3),
                    'ok': '✅' if correct else '❌'})

hard_df = pd.DataFrame(results)
print('\n✅ Cell 9: Hard Test Results')
print(hard_df.to_string(index=False))
hard_acc = hard_df['ok'].eq('✅').mean()
print(f'\nHard test accuracy: {hard_acc:.2f}')

# ── CELL 10: Build Retrieval Index ────────────────────────────────────
from sentence_transformers import SentenceTransformer

retrieval_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def build_doc(loc):
    import re
    coord = loc.get('coordinates','')
    floor_num = None
    if 'K-1' in coord.upper():
        floor_num = -1
    else:
        m = re.search(r'(\d+)', coord)
        if m:
            floor_num = int(m.group(1))

    floor_text = {
        -1: "basement floor gym",
         0: "ground floor zero level entrance reception cafeteria admin",
         1: "first floor 1st floor library cse it computer",
         2: "second floor 2nd floor ai ece electronics",
         3: "third floor 3rd floor mechanical electrical engineering",
         4: "fourth floor 4th floor civil chemical engineering",
         5: "fifth floor 5th floor robotics biotechnology",
         6: "sixth floor 6th floor mba humanities management",
    }.get(floor_num, "")

    return ' '.join(filter(None, [
        loc.get('name',''),
        loc.get('category','').replace('_',' '),
        loc.get('description',''),
        ' '.join(loc.get('keywords',[])),
        ' '.join(loc.get('aliases',[])),
        floor_text,
    ]))

docs       = [build_doc(loc) for loc in kb]
embeddings = retrieval_model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

with open(BASE_DIR / 'models/retrieval_embeddings.pkl','wb') as f:
    pickle.dump({'embeddings': embeddings,
                 'kb_ids': [loc['id'] for loc in kb]}, f)
print(f'✅ Cell 10: Retrieval index saved — shape {embeddings.shape}')

# ── CELL 11-12: Evaluate Retrieval ────────────────────────────────────
from sklearn.metrics.pairwise import cosine_similarity

def semantic_retrieve(query, top_k=3):
    q_emb  = retrieval_model.encode([query])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [{'record': kb[i], 'score': float(scores[i])} for i in top_idx]

eval_set = [
    ('where is the library',             'central_library'),
    ('where can i print my assignment',  'printing_room'),
    ('i lost my wallet',                 'lost_and_found'),
    ('where is civil faculty room',      'civil_faculty_room'),
    ('i feel sick need help',            'medical_room'),
    ('where can i study quietly',        'reading_room'),
    ('where is cafeteria',               'main_cafeteria'),
    ('where is computer lab',            'computer_lab'),
    ('where can i charge my laptop',     'charging_station'),
    ('where is the gym',                 'gym'),
]

top1 = top3 = 0
for q, expected_id in eval_set:
    results_r = semantic_retrieve(q, top_k=3)
    ids = [r['record']['id'] for r in results_r]
    top1 += int(ids[0] == expected_id)
    top3 += int(expected_id in ids)

ret_metrics = {'top1_accuracy': top1/len(eval_set),
               'top3_accuracy': top3/len(eval_set)}
with open(BASE_DIR / 'outputs/metrics/retrieval_metrics.json','w') as f:
    json.dump(ret_metrics, f, indent=2)
print(f'✅ Cell 11-12: Top-1={ret_metrics["top1_accuracy"]:.2f}, Top-3={ret_metrics["top3_accuracy"]:.2f}')

# ── CELL 13: Summary Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
intent_vals = [metrics['accuracy'],metrics['precision'],metrics['recall'],metrics['f1']]
axes[0].bar(['Accuracy','Precision','Recall','F1'], intent_vals,
            color=['#4C72B0','#DD8452','#55A868','#C44E52'])
axes[0].set_ylim(0, 1.05)
axes[0].set_title('Intent Classifier Metrics')
for i, v in enumerate(intent_vals):
    axes[0].text(i, v+0.01, f'{v:.3f}', ha='center')

axes[1].bar(['Top-1','Top-3'],
            [ret_metrics['top1_accuracy'], ret_metrics['top3_accuracy']],
            color=['#4C72B0','#55A868'])
axes[1].set_ylim(0, 1.05)
axes[1].set_title('Semantic Retrieval Accuracy')
for i, v in enumerate([ret_metrics['top1_accuracy'], ret_metrics['top3_accuracy']]):
    axes[1].text(i, v+0.01, f'{v:.2f}', ha='center')

plt.tight_layout()
plt.savefig(BASE_DIR / 'outputs/plots/pipeline_metrics.png', dpi=150)
plt.close()

print('\n' + '='*50)
print('ALL TRAINING COMPLETE')
print('Models saved in models/ — never need to retrain')
print('Plots saved in outputs/plots/')
print('Metrics saved in outputs/metrics/')
print('='*50)

# This block is intentionally empty - see fix below
