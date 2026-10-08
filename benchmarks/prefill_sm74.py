import json,time,urllib.request,statistics,concurrent.futures,sys,base64,io,re
from pathlib import Path
import argparse
parser=argparse.ArgumentParser(description="Cold-prefill A/B and small quality smoke; allocated test server only.")
parser.add_argument('--run',action='store_true');parser.add_argument('--base-url',required=True)
parser.add_argument('--model',required=True);parser.add_argument('--label',required=True)
parser.add_argument('--output-directory',type=Path,required=True);args=parser.parse_args()
if not args.run:parser.exit(0,'Plan only. Pass --run to send requests to your allocated server.\n')
root=args.output_directory;root.mkdir(parents=True,exist_ok=True)
label=args.label;assert label and all(c.isalnum() or c in '-_' for c in label)
base=args.base_url.rstrip('/');model=args.model

def request(prompt,max_tokens=256,stream=True,extra=None):
 body={'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':max_tokens,'stream':stream,'chat_template_kwargs':{'enable_thinking':False},**(extra or {})}
 if stream:body.update(stream_options={'include_usage':True})
 start=time.perf_counter();first=None;last=None;usage=None;parts=[]
 with urllib.request.urlopen(urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=360) as response:
  if not stream:return json.load(response)
  for raw in response:
   if not raw.startswith(b'data:'):continue
   data=raw[5:].strip()
   if data==b'[DONE]':break
   d=json.loads(data)
   if d.get('usage'):usage=d['usage']
   for x in d.get('choices',[]):
    delta=x.get('delta',{});text=(delta.get('content') or '')+(delta.get('reasoning_content') or '')
    if text:
     last=time.perf_counter();first=first or last;parts.append(text)
 end=time.perf_counter();assert usage and first and last
 n=usage['completion_tokens'];return {'tokens':n,'ttft_s':first-start,'wall_s':end-start,'decode_tps':(n-1)/max(last-first,1e-9),'prompt_tokens':usage['prompt_tokens'],'cached':usage.get('prompt_tokens_details',{}).get('cached_tokens',0),'text':''.join(parts)}

report={'variant':label,'cold':[],'quality':[]}
request('Warm-up: reply ACK.',32)
# Warm the kernel configurations first. These samples are excluded from measurements.
for count in [240,900,1800]:
 request(f'Warmup {count}.\n'+''.join(f'Record {i}: key alpha holds value {i%997} and status active.\n' for i in range(count))+'\nReply ACK.',16)
for count,concurrency,repeats in [(240,1,2),(240,2,2),(240,4,2),(240,6,2),(900,1,3),(1800,1,3)]:
 for rep in range(repeats):
  prompts=[f'Cold trial {count}/{concurrency}/{rep}/{i}.\n'+''.join(f'Record {j}: key alpha holds value {j%997} and status active.\n' for j in range(count))+'\nReply ACK.' for i in range(concurrency)]
  started=time.perf_counter()
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:rows=list(pool.map(lambda x:request(x,16),prompts))
  wall=time.perf_counter()-started
  assert all(x['cached']==0 for x in rows),rows
  r={'rows':count,'concurrency':concurrency,'rep':rep,'input_tokens':sum(x['prompt_tokens'] for x in rows),'median_ttft_s':statistics.median(x['ttft_s'] for x in rows),'max_ttft_s':max(x['ttft_s'] for x in rows),'batch_wall_s':wall,'batch_input_tps':sum(x['prompt_tokens'] for x in rows)/wall,'cached_tokens':0};report['cold'].append(r);print(json.dumps(r),flush=True);(root/(label+'-prefill-results.json')).write_text(json.dumps(report,indent=2))
prefix='Cache qualification: secret value ALPHA is 7319.\n'+''.join(f'Line {i} is an ordinary document row with no replacement secret.\n' for i in range(1000))
a=request(prefix+'\nReturn the ALPHA value only.',64);b=request(prefix+'\nReturn the ALPHA value only.',64);c=request(prefix+'\nReturn the ALPHA value only. Then print DONE.',64)
assert all('7319' in x['text'] for x in [a,b,c]);assert b['cached']>1000 and c['cached']>1000
report['cache']={name:{k:v for k,v in x.items() if k!='text'} for name,x in [('cold',a),('repeat',b),('append',c)]}
questions=[('What is 17*23? Reply only with the number.','391'),('What is the value of sum(i*i for i in range(5)) in Python? Reply only the number.','30'),('Convert hexadecimal 2A to decimal. Reply only the number.','42'),('A farmer has 17 sheep. All but 9 die. How many remain? Reply only the number.','9'),('A snail climbs a 30-foot wall. Each day it climbs 5 feet and each night slides down 4 feet. How many days to reach the top? Reply only the number.','26'),('What does Python len(set([1,1,2,3,3])) return? Reply only the number.','3'),('A list has [3, 1, 4, 1, 5]. What is the sum? Reply only the number.','14'),('Decode JSON {"items":[{"n":4},{"n":7}]}. Return the sum of the n fields only.','11'),('What does (13 ^ 7) evaluate to in Python? Reply only the number.','10'),('Find the missing term: 2, 6, 12, 20, 30, __. Reply only the number.','42')]
for q,want in questions:
 r=request(q,128);answer=r['text'].strip();ok=bool(re.search(r'(?<!\d)'+want+r'(?!\d)',answer));report['quality'].append({'question':q,'answer':answer,'expected':want,'passed':ok})
report['quality_correct']=sum(x['passed'] for x in report['quality']);report['done']=True;(root/(label+'-prefill-results.json')).write_text(json.dumps(report,indent=2));print(json.dumps({'quality':report['quality_correct'],'cached_repeat':b['cached'],'cached_append':c['cached']}),flush=True)
