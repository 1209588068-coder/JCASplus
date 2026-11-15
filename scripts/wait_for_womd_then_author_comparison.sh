#!/usr/bin/env bash
set -euo pipefail

RUN=/home/mizhou/JCAS/results/author_comparison_v3_20261113_000558
PLAN=$RUN/plan.json
WOMD_STATUS=/home/mizhou/JCAS_vehicle_lane/artifacts/womd_formal_main_corrected_fullcorpus_streaming_v2_20261112/status.json
STATE=$RUN/womd_dependency.json

python3 - "$STATE" "$PLAN" "$WOMD_STATUS" <<'PY'
import datetime, hashlib, json, sys
def sha(p):
 h=hashlib.sha256(); h.update(open(p,'rb').read()); return h.hexdigest()
state,plan,womd=sys.argv[1:]
json.dump({'status':'WAITING_FOR_WOMD','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'comparison_plan':plan,'comparison_plan_sha256':sha(plan),'womd_status':womd,
 'womd_plan':'/home/mizhou/JCAS_vehicle_lane/artifacts/womd_formal_main_corrected_fullcorpus_streaming_v2_20261112/plan.json'},
 open(state,'w'),indent=2)
PY

while true; do
  state=$(python3 - "$WOMD_STATUS" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get('status','MISSING'))
except Exception: print('MISSING')
PY
)
  if [ "$state" = "COMPLETE" ]; then break; fi
  if [ "$state" = "FAILED" ] || [ "$state" = "FINISHED_WITH_FAILURES" ]; then
    python3 - "$STATE" "$state" <<'PY'
import json,sys,datetime
p,state=sys.argv[1:]
json.dump({'status':'BLOCKED_WOMD_FAILED','womd_status':state,'at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),indent=2)
PY
    exit 1
  fi
  sleep 60
done

python3 - "$STATE" <<'PY'
import json,sys,datetime
p=sys.argv[1]
json.dump({'status':'WOMD_COMPLETE_ENQUEUEING','at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),indent=2)
PY

cd /home/mizhou/JCAS
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python3 -B -m author_comparisons.runner --enqueue --plan "$PLAN"

python3 - "$STATE" <<'PY'
import json,sys,datetime
p=sys.argv[1]
json.dump({'status':'QUEUED_AFTER_WOMD','at':datetime.datetime.now(datetime.timezone.utc).isoformat()},open(p,'w'),indent=2)
PY
